import numpy as np
from define.QECCLfid import utils as ut
from define.randchans import CreateIIDPauli
from define.chanreps import PauliConvertToTransfer
from define import fnames as fn


def GetLeadingPaulis(lead_frac, qcode, chan_probs, option, max_weight = None):
    # Get the leading Pauli probabilities in the iid model.
    # To get the indices of the k-largest elements: https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    # If the option is "full", it supplies top alpha fraction of entire chi diagonal
    # If the option is "weight", it supplies top alpha fraction of each weight until max weight
    if chan_probs.ndim == 1:
        iid_chan_probs = chan_probs
    else:
        iid_chan_probs = ut.GetErrorProbabilities(
            qcode.PauliOperatorsLST, chan_probs, 0
        )
    if option == "full":
        nPaulis = max(1, int(lead_frac * (4 ** qcode.N)))
        leading_paulis = np.argsort(iid_chan_probs)[-nPaulis:]
    elif option == "weight":
        if max_weight is None:
            max_weight = qcode.N//2 + 1
        if qcode.group_by_weight is None:
            PrepareSyndromeLookUp(qcode)
        n_errors_weight = [qcode.group_by_weight[w].size for w in range(max_weight + 1)]
        leading_paulis = np.array([],dtype=np.int)
        for w in range(max_weight+1):
            errors_wtw = qcode.group_by_weight[w]
            nPaulis = max(1, int(lead_frac * n_errors_weight[w]))
            indices_picked = errors_wtw[np.argsort(iid_chan_probs[errors_wtw])[-nPaulis:]]
            leading_paulis = np.concatenate((leading_paulis,indices_picked))
    return (1 - iid_chan_probs[0], leading_paulis, iid_chan_probs[leading_paulis])


def CompleteDecoderKnowledge(leading_fraction, chan_probs, qcode, option = "full"):
    # Complete the probabilities given to a ML decoder.
    (infid, known_paulis, known_probs) = GetLeadingPaulis(
        leading_fraction, qcode, chan_probs, option
    )
    # print(
    #     "Number of known paulis in decoder knowledge = {}".format(known_paulis.shape[0])
    # )
    # print("Total known probability = {}".format(np.sum(known_probs)))
    infid_qubit = 1 - np.power(1 - infid, 1 / qcode.N)
    decoder_probs = CreateIIDPauli(infid_qubit, qcode)
    decoder_probs[known_paulis] = known_probs
    total_unknown = 1 - np.sum(known_probs)
    # print("Total unknown probability = {}".format(total_unknown))
    # Normalize the unknown Paulis
    # https://stackoverflow.com/questions/27824075/accessing-numpy-array-elements-not-in-a-given-index-list
    mask = np.ones(decoder_probs.shape[0], dtype=bool)
    mask[known_paulis] = False
    decoder_probs[mask] = (
        total_unknown * decoder_probs[mask] / np.sum(decoder_probs[mask])
    )
    # print("Sum of decoder probs = {}".format(np.sum(decoder_probs)))
    # print("Sorted decoder probs = {}".format(np.sort(decoder_probs)))
    return decoder_probs


def PrepareChannelDecoder(submit, noise, sample):
    # Generate the reference channel for partial ML decoder
    # Reference channel at present is only created for level 1
    l = 0
    qcode = submit.eccs[l]
    (nrows, ncols) = (4 ** submit.eccs[l].K, 4 ** submit.eccs[l].K)
    diagmask = [
        q * ncols * nrows + ncols * j + j for q in range(qcode.N) for j in range(nrows)
    ]
    rawchan = np.load(fn.RawPhysicalChannel(submit, noise))[sample, :]
    if submit.iscorr == 0:
        chan_probs = np.reshape(np.tile(rawchan, qcode.N)[diagmask], [qcode.N, nrows])
    elif submit.iscorr == 2:
        chan_probs = np.reshape(rawchan[diagmask], [qcode.N, nrows])
    else:
        chan_probs = rawchan
    decoder_probs = CompleteDecoderKnowledge(submit.decoder_fraction, chan_probs, qcode)
    # elif l == 1:
    #     ComputeAdaptiveDecoder(qcode, decoder_probs)
    #     chan_probs = np.tile(
    #         [
    #             ComputeResiduals(
    #                 p, decoder_probs, qcode, lookup=qcode.tailored_lookup
    #             )
    #             for p in range(4)
    #         ],
    #         [submit.eccs[1].N, 1],
    #     )
    #     print("Chan probs at level 2 = {}".format(chan_probs[0]))
    #     print("Sum of chan probs at level 2 = {}".format(np.sum(chan_probs[0])))
    #     print(
    #         "Infidelity at level 2 from chan probs = {}".format(
    #             1 - chan_probs[0][0]
    #         )
    #     )
    #     process_mat = [
    #         np.sum(chan_probs[0]),
    #         chan_probs[0][0]
    #         + chan_probs[0][1]
    #         - chan_probs[0][3]
    #         - chan_probs[0][2],
    #         chan_probs[0][0]
    #         + chan_probs[0][2]
    #         - chan_probs[0][1]
    #         - chan_probs[0][3],
    #         chan_probs[0][0]
    #         + chan_probs[0][3]
    #         - chan_probs[0][1]
    #         - chan_probs[0][2],
    #     ]
    #     print(
    #         "Effective logical channel (level-1) from chan probs = {}".format(
    #             process_mat
    #         )
    #     )
    #     decoder_probs = ut.GetErrorProbabilities(
    #         qcode.PauliOperatorsLST, chan_probs, 0
    #     )
    # else:
    #     pass
    decoder_knowledge = PauliConvertToTransfer(decoder_probs, qcode)
    return decoder_knowledge
