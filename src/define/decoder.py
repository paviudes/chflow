import numpy as np
from define.QECCLfid import utils as ut
from define.QECCLfid.minwt import ComputeResiduals
from define.randchans import CreateIIDPauli
from define.chanreps import PauliConvertToTransfer
from define.qcode import ComputeAdaptiveDecoder
from define import fnames as fn

def GetLeadingPaulis(lead_frac, qcode, chan_probs):
    # Get the leading Pauli probabilities in the iid model.
    # To get the indices of the k-largest elements: https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    nPaulis = max(1, int(lead_frac * (4 ** qcode.N)))
    if chan_probs.ndim == 1:
        iid_chan_probs = chan_probs
    else:
        iid_chan_probs = ut.GetErrorProbabilities(qcode.PauliOperatorsLST, chan_probs, 0)
    leading_paulis = np.argsort(iid_chan_probs)[-nPaulis:]
    return (1 - iid_chan_probs[0], leading_paulis, iid_chan_probs[leading_paulis])

def CompleteDecoderKnowledge(leading_fraction, chan_probs, qcode, level):
    # Complete the probabilities given to a ML decoder.
    (infid, known_paulis, known_probs) = GetLeadingPaulis(leading_fraction, qcode, chan_probs)
    decoder_probs = CreateIIDPauli(infid, qcode)
    decoder_probs[known_paulis] = known_probs
    total_unknown = 1 - np.sum(known_probs)
    # Normalize the unknown Paulis
    # https://stackoverflow.com/questions/27824075/accessing-numpy-array-elements-not-in-a-given-index-list
    mask = np.ones(decoder_probs.shape[0], dtype = bool)
    mask[known_paulis] = False
    decoder_probs[mask] = total_unknown * decoder_probs[mask]/np.sum(decoder_probs[mask])
    return decoder_probs

def PrepareChannelDecoder(submit, noise, sample):
    # Generate the decoder input for a given channel in the submission.
    decoder_knowledge = [None for __ in range(submit.levels)]
    for l in range(submit.levels):
        if l == 0:
            qcode = submit.eccs[l]
            (nrows, ncols) = (4**submit.eccs[l].K, 4**submit.eccs[l].K)
            diagmask = [q * ncols * nrows + ncols * j + j for q in range(qcode.N) for j in range(nrows)]
            rawchan = np.load(fn.RawPhysicalChannel(submit, noise))[sample, :]
            if submit.iscorr == 0:
                chan_probs = np.reshape(np.tile(rawchan, qcode.N)[diagmask], [qcode.N, nrows])
            elif submit.iscorr == 2:
                chan_probs = np.reshape(rawchan[diagmask], [qcode.N, nrows])
            else:
                chan_probs = rawchan
        elif l == 1:
            ComputeAdaptiveDecoder(qcode, decoder_probs)
            chan_probs = np.tile([ComputeResiduals(p, decoder_probs, submit.eccs[0], lookup=qcode.tailored_lookup) for p in range(4)], [submit.eccs[1].N, 1])
        else:
            pass
        decoder_probs = CompleteDecoderKnowledge(submit.decoder_fraction, chan_probs, qcode, l)
        decoder_knowledge[l] = PauliConvertToTransfer(decoder_probs, qcode)
    return decoder_knowledge