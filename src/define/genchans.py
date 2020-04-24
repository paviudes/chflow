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


def PreparePhysicalChannels(submit, nproc=1):
    # Prepare a file for each noise rate, that contains all single qubit channels, one for each sample.
    nproc = min(nproc, mp.cpu_count())
    chunk = int(np.ceil(submit.samps / nproc))
    os.system("mkdir -p %s/physical" % (submit.outdir))
    # Create quantum channels for various noise parameters and store them in the process matrix formalism.
    if submit.iscorr == 0:
        nparams = 4 ** submit.eccs[0].K * 4 ** submit.eccs[0].K
    else:
        # submit.rawchans = np.zeros(
        #     (submit.noiserates.shape[0], submit.samps, 4 ** submit.eccs[0].N),
        #     dtype=np.longdouble,
        # )
        nparams = 2 ** (submit.eccs[0].N + submit.eccs[0].K)
    phychans = mp.Array(
        ct.c_double,
        np.zeros(
            (submit.noiserates.shape[0] * submit.samps * nparams), dtype=np.double
        ),
    )
    rawchans = mp.Array(
        ct.c_double,
        np.zeros(
            (submit.noiserates.shape[0] * submit.samps * nparams), dtype=np.double
        ),
    )
    # submit.phychans = np.zeros(
    #     (submit.noiserates.shape[0], submit.samps, nparams), dtype=np.longdouble
    # )
    noise = np.zeros(submit.noiserates.shape[1], dtype=np.longdouble)
    for i in tqdm(
        range(submit.noiserates.shape[0]),
        ascii=True,
        desc="\033[2mPreparing physical channels:",
    ):
        for j in range(submit.noiserates.shape[1]):
            if submit.scales[j] == 1:
                noise[j] = submit.noiserates[i, j]
            else:
                noise[j] = np.power(submit.scales[j], submit.noiserates[i, j])
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
        rawchans, [submit.noiserates.shape[0], submit.samps, nparams], order="c"
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


def GenChannelSamples(noise, noiseidx, samps, submit, nparams, phychans, rawchans):
    r"""
	Generate samples of various channels with a given noise rate.
	"""
    for j in range(samps[0], samps[1]):
        if submit.iscorr == 0:
            phychans[
                (noiseidx * submit.samps * nparams + j * nparams) : (
                    noiseidx * submit.samps * nparams + (j + 1) * nparams
                )
            ] = crep.ConvertRepresentations(
                chdef.GetKraussForChannel(submit.channel, *noise), "krauss", "process"
            ).ravel()
            rawchans[
                (noiseidx * submit.samps * nparams + j * nparams) : (
                    noiseidx * submit.samps * nparams + (j + 1) * nparams
                )
            ] = np.real(
                crep.ConvertRepresentations(
                    np.reshape(
                        phychans[
                            (noiseidx * submit.samps * nparams + j * nparams) : (
                                noiseidx * submit.samps * nparams + (j + 1) * nparams
                            )
                        ],
                        [4, 4],
                        order="c",
                    ),
                    "process",
                    "chi",
                )
            ).ravel()
        else:
            rawchans[
                (noiseidx * submit.samps * nparams + j * nparams) : (
                    noiseidx * submit.samps * nparams + (j + 1) * nparams
                )
            ] = chdef.GetKraussForChannel(
                submit.channel, *np.concatenate((noise, [submit.eccs[0].N]))
            )
            phychans[
                (noiseidx * submit.samps * nparams + j * nparams) : (
                    noiseidx * submit.samps * nparams + (j + 1) * nparams
                )
            ] = crep.PauliConvertToTransfer(
                rawchans[
                    (noiseidx * submit.samps * nparams + j * nparams) : (
                        noiseidx * submit.samps * nparams + (j + 1) * nparams
                    )
                ],
                submit.eccs[0],
            )
    return None
