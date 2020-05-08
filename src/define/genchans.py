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


def PreparePhysicalChannels(submit, nproc=4):
    # Prepare a file for each noise rate, that contains all single qubit channels, one for each sample.
    nproc = min(nproc, mp.cpu_count())
    chunk = int(np.ceil(submit.samps / nproc))
    os.system("mkdir -p %s/physical" % (submit.outdir))
    # Create quantum channels for various noise parameters and store them in the process matrix formalism.
    if submit.iscorr == 0:
        nparams = 4 ** submit.eccs[0].K * 4 ** submit.eccs[0].K
        raw_params = params
    elif submit.iscorr == 1:
        # submit.rawchans = np.zeros(
        #     (submit.noiserates.shape[0], submit.samps, 4 ** submit.eccs[0].N),
        #     dtype=np.longdouble,
        # )
        nparams = 2 ** (submit.eccs[0].N + submit.eccs[0].K)
        raw_params = 4 ** submit.eccs[0].N
    else:
        nparams = submit.eccs[0].N * 4 ** submit.eccs[0].K * 4 ** submit.eccs[0].K
        raw_params = params
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
    # submit.phychans = np.zeros(
    #     (submit.noiserates.shape[0], submit.samps, nparams), dtype=np.longdouble
    # )
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
                    ),
                )
            )
        for k in range(nproc):
            processes[k].start()
        for k in range(nproc):
            processes[k].join()
    print("\033[0m")
    submit.phychans = np.reshape(
        phychans, [submit.noiserates.shape[0], submit.samps, nparams], order="c"
    )
    submit.rawchans = np.reshape(
        rawchans, [submit.noiserates.shape[0], submit.samps, raw_params], order="c"
    )
    if submit.nodes > 0:
        submit.cores[0] = int(
            np.ceil(
                submit.noiserates.shape[0] * submit.samps / np.longdouble(submit.nodes)
            )
        )
    submit.nodes = int(
        np.ceil(
            submit.noiserates.shape[0] * submit.samps / np.longdouble(submit.cores[0])
        )
    )
    return None


def GenChannelSamples(
    noise, noiseidx, samps, submit, nparams, raw_params, phychans, rawchans
):
    r"""
	Generate samples of various channels with a given noise rate.
	"""
    np.random.seed()
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
                chdef.GetKraussForChannel(submit.channel, submit.eccs[0].N, *noise),
                "krauss",
                "process",
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
        else:
            chans = chdef.GetKraussForChannel(submit.channel, *noise)
            nentries = 4 ** submit.eccs[0].K * 4 ** submit.eccs[0].K
            for q in range(chans.shape[0]):
                phychans[
                    (noiseidx * submit.samps * nparams + j * nparams + q * nentries) : (
                        noiseidx * submit.samps * nparams
                        + j * nparams
                        + (q + 1) * nentries
                    )
                ] = crep.ConvertRepresentations(
                    chans[np.newaxis, q, :, :], "krauss", "process"
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

    return None
