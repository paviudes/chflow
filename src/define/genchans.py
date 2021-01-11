import sys
import os

try:
    from tqdm import tqdm
except:
    pass
try:
    import numpy as np
    from scipy import linalg as linalg
except:
    pass
import ctypes as ct
import multiprocessing as mp
from define import metrics as ml
from define import randchans as rchan
from define import chandefs as chdef
from define import chanreps as crep
from define import fnames as fn


def ChannelPair(chtype, rates, dim, method="qr"):
    # Generate process matrices for two channels that can be associated to the same "family".
    channels = np.zeros((2, 4, 4), dtype=np.longdouble)
    if chtype == "rand":
        # Generate process matrices for two random channels that originate from the same Hamiltonian.
        # If the rate already has a source, use it. When the source is fixed, there is no randomness in the channels.
        sourcefname = "physical/rand_source_%s.npy" % (method)
        if not os.path.isfile(sourcefname):
            print(
                "\033[2mCreating new random source with method = %s.\033[0m" % (method)
            )
            randH = rchan.RandomHermitian(dim, method=method)
            np.save(sourcefname, randH)
        randH = np.load(sourcefname)
        for i in range(2):
            randU = rchan.RandomUnitary(rates[i], dim, method=method, randH=randH)
            krauss = crep.ConvertRepresentations(randU, "stine", "krauss")
            channels[i, :, :] = crep.ConvertRepresentations(krauss, "krauss", "process")
    else:
        for i in range(2):
            krauss = chdef.GetKraussForChannel(chtype, rates[i])
            channels[i, :, :] = crep.ConvertRepresentations(krauss, "krauss", "process")
    return channels


def PreparePhysicalChannels(submit, nproc=None):
    # Prepare a file for each noise rate, that contains all single qubit channels, one for each sample.
    if nproc is None:
        nproc = mp.cpu_count()
    chunk = int(np.ceil(submit.samps / nproc))
    os.system("mkdir -p %s/physical" % (submit.outdir))
    # Create quantum channels for various noise parameters and store them in the process matrix formalism.
    if submit.iscorr == 0:
        nparams = 4 ** submit.eccs[0].K * 4 ** submit.eccs[0].K
        raw_params = nparams
    elif submit.iscorr == 1:
        nparams = 2 ** (submit.eccs[0].N + submit.eccs[0].K)
        raw_params = 4 ** submit.eccs[0].N
    elif submit.iscorr == 2:
        nparams = submit.eccs[0].N * 4 ** submit.eccs[0].K * 4 ** submit.eccs[0].K
        raw_params = nparams
    else:
        nparams = 4 ** (submit.eccs[0].N + submit.eccs[0].K)
        raw_params = 4 ** submit.eccs[0].N

    phychans = mp.Array(
        ct.c_double,
        np.zeros(
            (submit.noiserates.shape[0] * submit.samps * nparams), dtype=np.double
        ),
    )
    rawchans = mp.Array(
        ct.c_double,
        np.zeros(
            (submit.noiserates.shape[0] * submit.samps * raw_params), dtype=np.double
        ),
    )
    misc_info = [["None" for __ in range(submit.samps)] for __ in range(submit.noiserates.shape[0])]

    for i in tqdm(
        range(submit.noiserates.shape[0]),
        ascii=True,
        desc="\033[2mPreparing physical channels:",
    ):
        noise = np.zeros(submit.noiserates.shape[1], dtype=np.longdouble)
        for j in range(submit.noiserates.shape[1]):
            if submit.scales[j] == 1:
                noise[j] = submit.noiserates[i, j]
            else:
                noise[j] = np.power(submit.scales[j], submit.noiserates[i, j])
        # if submit.iscorr > 0:
        #     noise = np.insert(noise, 0, submit.eccs[0].N)
        misc = mp.Queue()
        processes = []
        for k in range(nproc):
            processes.append(
                mp.Process(
                    target=GenChannelSamples,
                    args=(
                        noise,
                        i,
                        [k * chunk, min(submit.samps, (k + 1) * chunk)],
                        submit,
                        nparams,
                        raw_params,
                        phychans,
                        rawchans,
                        misc
                    ),
                )
            )
        for k in range(nproc):
            processes[k].start()
        for k in range(nproc):
            processes[k].join()
        # Gathering the interactions results
        for s in range(submit.samps):
            (samp, info) = misc.get()
            if (info == 0):
                info = ([("N", "N")], 0, 0, 0)
            misc_info[i][samp] = info

    print("\033[0m")
    submit.phychans = np.reshape(
        phychans, [submit.noiserates.shape[0], submit.samps, nparams], order="c"
    )
    submit.rawchans = np.reshape(
        rawchans, [submit.noiserates.shape[0], submit.samps, raw_params], order="c"
    )
    # The miscellaneous info for correlated CPTP channels contains the interactions used to generate it.
    submit.misc = misc_info
    # print("Physical channels: {}".format(submit.phychans))
    # Prepare the weights of Pauli errors that will be supplied to the decoder: nr_weights.
    PrepareNRWeights(submit)
    return None


def PrepareNRWeights(submit):
    # Prepare the weights of Pauli errors that will be supplied to the decoder: nr_weights.
    # Use properties of submit to retrieve the mean and cutoff of the Poisson distribution: submit.noiserates[i, :] = (__, cutoff, __, mean)
    # Save the nr_weights to a file.
    qcode = submit.eccs[0]
    max_weight = qcode.N//2 + 1
    nr_weights = np.zeros((submit.samps, 4 ** qcode.N), dtype = np.int)
    for r in range(submit.noiserates.shape[0]):
        (__, cutoff, __, mean) = submit.noiserates[r, :]
        for s in range(submit.samps):
            nr_weights[s, :] = [SamplePoisson(1, cutoff=max_weight) for __ in range(4 ** qcode.N)]
        # Save the nr_weights to a file.
        fn.NRWeightsFile(submit, submit.noiserates[r, :])
    return None


def GenChannelSamples(
    noise, noiseidx, samps, submit, nparams, raw_params, phychans, rawchans, misc
):
    r"""
	Generate samples of various channels with a given noise rate.
	"""
    np.random.seed()
    nstabs = 2 ** (submit.eccs[0].N - submit.eccs[0].K)
    nlogs = 4 ** submit.eccs[0].K
    diagmask = np.array(
        [nlogs * nstabs * j + j for j in range(nlogs * nstabs)], dtype=np.int
    )
    for j in range(samps[0], samps[1]):
        # print(
        #     "Noise at {}, samp {} = {}".format(
        #         noiseidx, j, list(map(lambda num: "%g" % num, noise))
        #     )
        # )
        if submit.iscorr == 0:
            phychans[
                (noiseidx * submit.samps * nparams + j * nparams) : (
                    noiseidx * submit.samps * nparams + (j + 1) * nparams
                )
            ] = crep.ConvertRepresentations(
                chdef.GetKraussForChannel(submit.channel, *noise), "krauss", "process"
            ).ravel()
            rawchans[
                (noiseidx * submit.samps * raw_params + j * raw_params) : (
                    noiseidx * submit.samps * raw_params + (j + 1) * raw_params
                )
            ] = np.real(
                crep.ConvertRepresentations(
                    np.reshape(
                        phychans[
                            (noiseidx * submit.samps * nparams + j * nparams) : (
                                noiseidx * submit.samps * nparams + (j + 1) * nparams
                            )
                        ],
                        [4 ** submit.eccs[0].K * 4 ** submit.eccs[0].K],
                        order="c",
                    ),
                    "process",
                    "chi",
                )
            ).ravel()
            misc.put((j, 0))

        elif submit.iscorr == 1:
            rawchans[
                (noiseidx * submit.samps * raw_params + j * raw_params) : (
                    noiseidx * submit.samps * raw_params + (j + 1) * raw_params
                )
            ] = chdef.GetKraussForChannel(submit.channel, submit.eccs[0], *noise)
            phychans[
                (noiseidx * submit.samps * nparams + j * nparams) : (
                    noiseidx * submit.samps * nparams + (j + 1) * nparams
                )
            ] = crep.PauliConvertToTransfer(
                np.array(
                    rawchans[
                        (noiseidx * submit.samps * raw_params + j * raw_params) : (
                            noiseidx * submit.samps * raw_params + (j + 1) * raw_params
                        )
                    ],
                    dtype=np.double,
                ),
                submit.eccs[0],
            )
            misc.put((j, 0))

        elif submit.iscorr == 2:
            chans = chdef.GetKraussForChannel(submit.channel, submit.eccs[0].N, *noise)
            nentries = 4 ** submit.eccs[0].K * 4 ** submit.eccs[0].K
            for q in range(chans.shape[0]):
                phychans[
                    (noiseidx * submit.samps * nparams + j * nparams + q * nentries) : (
                        noiseidx * submit.samps * nparams
                        + j * nparams
                        + (q + 1) * nentries
                    )
                ] = crep.ConvertRepresentations(
                    chans[q, :, :, :], "krauss", "process"
                ).ravel()
                rawchans[
                    (
                        noiseidx * submit.samps * raw_params
                        + j * raw_params
                        + q * nentries
                    ) : (
                        noiseidx * submit.samps * raw_params
                        + j * raw_params
                        + (q + 1) * nentries
                    )
                ] = np.real(
                    crep.ConvertRepresentations(
                        np.reshape(
                            phychans[
                                (
                                    noiseidx * submit.samps * nparams
                                    + j * nparams
                                    + q * nentries
                                ) : (
                                    noiseidx * submit.samps * nparams
                                    + j * nparams
                                    + (q + 1) * nentries
                                )
                            ],
                            [4 ** submit.eccs[0].K * 4 ** submit.eccs[0].K],
                        ),
                        "process",
                        "chi",
                    )
                ).ravel()
            misc.put((j, 0))

        else:
            (
                phychans[
                    (noiseidx * submit.samps * nparams + j * nparams) : (
                        noiseidx * submit.samps * nparams + (j + 1) * nparams
                    )
                ],
                rawchans[
                    (noiseidx * submit.samps * raw_params + j * raw_params) : (
                        noiseidx * submit.samps * raw_params + (j + 1) * raw_params
                    )
                ],
                interactions # Information about the different interactions.
            ) = chdef.GetKraussForChannel(submit.channel, submit.eccs[0], *noise)
            misc.put((j, interactions))
    return None
