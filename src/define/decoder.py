import numpy as np
from define.QECCLfid import utils as ut
from define.randchans import CreateIIDPauli
from define.chanreps import PauliConvertToTransfer
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
    decoder_probs_level = np.power(decoder_probs, (np.power(qcode.D, level) + 1)/2)
    decoder_knowledge = PauliConvertToTransfer(decoder_probs_level/np.sum(decoder_probs_level), qcode)
    return decoder_knowledge

def PrepareChannelDecoder(submit, noise, sample):
    # Generate the decoder input for a given channel in the submission.
    decoder_knowledge = [None for __ in range(submit.levels)]
    for l in range(submit.levels):
        qcode = submit.eccs[l]
        nrows = 4**submit.eccs[l].K
        ncols = 4**submit.eccs[l].K
        # decoder_knowledge = np.zeros((submit.samps, 2**(qcode.N + qcode.K)))
        diagmask = [q * ncols * nrows + ncols * j + j for q in range(qcode.N) for j in range(nrows)]
        rawchan = np.load(fn.RawPhysicalChannel(submit, noise))[sample, :]
        if submit.iscorr == 0:
            chan_probs = np.reshape(np.tile(rawchan, qcode.N)[diagmask], [qcode.N, nrows])
        elif submit.iscorr == 2:
            chan_probs = np.reshape(rawchan[diagmask], [qcode.N, nrows])
        else:
            chan_probs = rawchan
        decoder_knowledge[l] = CompleteDecoderKnowledge(submit.decoder_fraction, chan_probs, qcode, l)
    return decoder_knowledge