import os
import sys
import numpy as np
import ctypes as ct
from timeit import default_timer as timer
from tqdm import tqdm
from scipy import linalg as linalg
from define.chandefs import GetKraussForChannel
from define.QECCLfid.utils import SamplePoisson
from define.randchans import RandomHermitian, RandomUnitary
from define.chanreps import ConvertRepresentations, PauliConvertToTransfer


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
			randH = RandomHermitian(dim, method=method)
			np.save(sourcefname, randH)
		randH = np.load(sourcefname)
		for i in range(2):
			randU = RandomUnitary(rates[i], dim, method=method, randH=randH)
			krauss = ConvertRepresentations(randU, "stine", "krauss")
			channels[i, :, :] = ConvertRepresentations(krauss, "krauss", "process")
	else:
		for i in range(2):
			krauss = GetKraussForChannel(chtype, rates[i])
			channels[i, :, :] = ConvertRepresentations(krauss, "krauss", "process")
	return channels


def PreparePhysicalChannels(submit, nproc=1):
	# Prepare a file for each noise rate, that contains all single qubit channels, one for each sample.
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

	phychans = np.zeros((submit.noiserates.shape[0] * submit.samps * nparams), dtype=np.double)
	rawchans = np.zeros(2*(submit.noiserates.shape[0] * submit.samps * raw_params), dtype=np.double)

	misc_info = [["None" for __ in range(submit.samps)] for __ in range(submit.noiserates.shape[0])]

	for i in tqdm(range(submit.noiserates.shape[0]), ascii=True, desc="Preparing physical channels:", colour = "green"):
		noise = np.zeros(submit.noiserates.shape[1], dtype=np.longdouble)
		for j in range(submit.noiserates.shape[1]):
			if submit.scales[j] == 1:
				noise[j] = submit.noiserates[i, j]
			else:
				noise[j] = np.power(submit.scales[j], submit.noiserates[i, j])
		
		for k in range(nproc):
			GenChannelSamples(noise, i, [k * chunk, min(submit.samps, (k + 1) * chunk)], submit, nparams, raw_params, phychans, rawchans, misc_info[i])

	submit.phychans = np.reshape(
		phychans, [submit.noiserates.shape[0], submit.samps, nparams], order="c"
	)
	total_raw_params = raw_params*submit.noiserates.shape[0]*submit.samps
	rawchans_real = np.reshape(
		rawchans[:total_raw_params], [submit.noiserates.shape[0], submit.samps, raw_params], order="c"
	)
	rawchans_imag = np.reshape(
		rawchans[total_raw_params:], [submit.noiserates.shape[0], submit.samps, raw_params], order="c"
	)
	submit.rawchans = rawchans_real + 1j*rawchans_imag
	# The miscellaneous info for correlated CPTP channels contains the interactions used to generate it.
	if submit.iscorr == 3:
		submit.misc = misc_info
	# print("Physical channels: {}".format(submit.phychans))
	return None


def GenChannelSamples(noise, noiseidx, samps, submit, nparams, raw_params, phychans, rawchans, misc):
	r"""
	Generate samples of various channels with a given noise rate.
	"""
	np.random.seed()
	nstabs = 2 ** (submit.eccs[0].N - submit.eccs[0].K)
	nlogs = 4 ** submit.eccs[0].K
	diagmask = np.array(
		[nlogs * nstabs * j + j for j in range(nlogs * nstabs)], dtype=np.int64
	)
	for j in range(samps[0], samps[1]):
		# print(
		#     "Noise at {}, samp {} = {}".format(
		#         noiseidx, j, list(map(lambda num: "%g" % num, noise))
		#     )
		# )
		
		start = timer()

		if submit.iscorr == 0:
			phychans[
				(noiseidx * submit.samps * nparams + j * nparams) : (
					noiseidx * submit.samps * nparams + (j + 1) * nparams
				)
			] = ConvertRepresentations(
				GetKraussForChannel(submit.channel, *noise), "krauss", "process"
			).ravel()
			rawchans[
				(noiseidx * submit.samps * raw_params + j * raw_params) : (
					noiseidx * submit.samps * raw_params + (j + 1) * raw_params
				)
			] = np.real(
				ConvertRepresentations(
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
			misc[j] = ([("N", "N")], 0, 0, 0)

		elif submit.iscorr == 1:
			rawchans[
				(noiseidx * submit.samps * raw_params + j * raw_params) : (
					noiseidx * submit.samps * raw_params + (j + 1) * raw_params
				)
			] = GetKraussForChannel(submit.channel, submit.eccs[0], *noise)
			phychans[
				(noiseidx * submit.samps * nparams + j * nparams) : (
					noiseidx * submit.samps * nparams + (j + 1) * nparams
				)
			] = PauliConvertToTransfer(
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
			misc[j] = ([("N", "N")], 0, 0, 0)

		elif submit.iscorr == 2:
			chans = GetKraussForChannel(submit.channel, submit.eccs[0].N, *noise)
			nentries = 4 ** submit.eccs[0].K * 4 ** submit.eccs[0].K
			for q in range(chans.shape[0]):
				phychans[
					(noiseidx * submit.samps * nparams + j * nparams + q * nentries) : (
						noiseidx * submit.samps * nparams
						+ j * nparams
						+ (q + 1) * nentries
					)
				] = ConvertRepresentations(
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
					ConvertRepresentations(
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
			misc[j] = ([("N", "N")], 0, 0, 0)

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
			) = GetKraussForChannel(submit.channel, submit.eccs[0], *noise)
			misc[j] = interactions
	
		print("Noise rate {}, sample {} done in {} seconds.".format(noise, j, timer() - start))

	return None
