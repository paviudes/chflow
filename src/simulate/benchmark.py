import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from define import fnames as fn
from define.chanreps import ConvertRepresentations, ChangeOrdering
from define.decoder import CompleteDecoderKnowledge

DEBUG = 0

def GetBMOutput(nlevels, nmetrics, nlogs, nbins, nbreaks):
	# Create a class called BenchOut to mirror the C structure.
	class BMOutput(ctypes.Structure):
		"""
		Class to mirror the C structure BenchOut defined in benchmark.h
		"""

		"""
		_fields_ = [("logchans", ndpointer(dtype = np.float64, shape = ((nlevels + 1) * nlogs * nlogs,))),
					("chanvar", ndpointer(dtype = np.float64, shape = ((nlevels + 1) * nlogs * nlogs,))),
					("logerrs", ndpointer(dtype = np.float64, shape = ((nlevels + 1) * nmetrics,))),
					("logvars", ndpointer(dtype = np.float64, shape = ((nlevels + 1) * nmetrics,))),
					("bins", ndpointer(dtype = np.float64, shape = (nmetrics * (nlevels + 1) * nbins * nbins,))),
					("running", ndpointer(dtype = np.float64, shape = (nmetrics * nbreaks,)))]
		"""
		_fields_ = [
			(
				"logchans",
				ctypes.POINTER(ctypes.c_double * ((nlevels + 1) * nlogs * nlogs)),
			),
			(
				"chanvar",
				ctypes.POINTER(ctypes.c_double * ((nlevels + 1) * nlogs * nlogs)),
			),
			("logerrs", ctypes.POINTER(ctypes.c_double * ((nlevels + 1) * nmetrics))),
			("logvars", ctypes.POINTER(ctypes.c_double * ((nlevels + 1) * nmetrics))),
			(
				"bins",
				ctypes.POINTER(
					ctypes.c_int * (nmetrics * (nlevels + 1) * nbins * nbins)
				),
			),
			("running", ctypes.POINTER(ctypes.c_double * (nmetrics * nbreaks))),
		]

	return BMOutput


def Unpack(varname, pointer, nlevels, nmetrics, nlogs, nbins, nbreaks):
	# Convert the 1D pointer representation of an BMOutput variable to an array of the required shape.
	types = {
		"logchans": [
			ctypes.c_double,
			(nlevels + 1) * nlogs * nlogs,
			[nlevels + 1, nlogs, nlogs],
		],
		"chanvar": [
			ctypes.c_double,
			(nlevels + 1) * nlogs * nlogs,
			[nlevels + 1, nlogs, nlogs],
		],
		"logerrs": [ctypes.c_double, (nlevels + 1) * nmetrics, [nmetrics, nlevels + 1]],
		"logvars": [ctypes.c_double, (nlevels + 1) * nmetrics, [nmetrics, nlevels + 1]],
		"bins": [
			ctypes.c_int,
			(nlevels + 1) * nbins * nbins,
			[nmetrics, nlevels + 1, nbins, nbins],
		],
		"running": [ctypes.c_double, nmetrics * nbreaks, [nmetrics, nbreaks]],
	}
	ctvec = ctypes.cast(
		pointer, ctypes.POINTER(types[varname][0] * types[varname][1])
	).contents
	reshaped = np.ctypeslib.as_array(ctvec).reshape(types[varname][2])
	return reshaped


def Benchmark(submit, noise, sample, physical, refchan, infidelity, rawchan=None):
	# This is a wrapper function to the C function in benchmark.c
	# We will prepare the inputs to the C function as numpy arrays and use ctypes to convert them to C pointers.
	# After the benchmarking process is run, we will then transform the output form C pointers to numpy arrays.
	# print("Physical channel shape {}: {}.".format(physical.shape, list(physical)))
	iscorr = submit.iscorr
	nmetrics = len(submit.metrics)
	nlevels = submit.levels
	nlogs = 4 ** (submit.eccs[nlevels - 1].K)
	nkd = np.zeros(3 * nlevels, dtype=np.int32)
	nstabs = 0
	for l in range(nlevels):
		(nkd[3 * l], nkd[3 * l + 1], nkd[3 * l + 2]) = (
			submit.eccs[l].N,
			submit.eccs[l].K,
			submit.eccs[l].D,
		)
		nstabs = nstabs + 2 ** (submit.eccs[l].N - submit.eccs[l].K)
	SS = np.zeros(
		np.sum([4 ** (submit.eccs[l].N - submit.eccs[l].K) for l in range(nlevels)]),
		dtype=np.int32,
	)
	normalizer = np.zeros(
		np.sum(
			[
				2 ** (submit.eccs[l].N + submit.eccs[l].K) * submit.eccs[l].N
				for l in range(nlevels)
			]
		),
		dtype=np.int32,
	)
	normphases_real = np.zeros(
		np.sum([2 ** (submit.eccs[l].N + submit.eccs[l].K) for l in range(nlevels)]),
		dtype=np.float64,
	)
	normphases_imag = np.zeros(
		np.sum([2 ** (submit.eccs[l].N + submit.eccs[l].K) for l in range(nlevels)]),
		dtype=np.float64,
	)
	decoders = np.zeros(nlevels, dtype=np.int32)
	dclookups = np.zeros(np.sum([2 ** (submit.eccs[l].N - submit.eccs[l].K) for l in range(nlevels)]), dtype=np.int32)
	operators_LST = np.zeros(
		np.sum([4 ** (submit.eccs[l].N) * submit.eccs[l].N for l in range(nlevels)]),
		dtype=np.int32,
	)

	if ((submit.decoders)[0] == 3 or (submit.decoders)[0] == 4): # Introduced decoder 4 to capture top alpha grouped by weight
		if submit.iscorr == 0:
			chan_probs = np.tile(
				np.real(np.diag(ConvertRepresentations(physical, "process", "chi"))),
				[submit.eccs[0].N, 1],
			)
		elif submit.iscorr == 2:
			chans_ptm = np.reshape(physical, [submit.eccs[0].N, 4, 4])
			chan_probs = np.zeros((submit.eccs[0].N, 4), dtype=np.double)
			for q in range(submit.eccs[0].N):
				chan_probs[q, :] = np.real(
					np.diag(
						ConvertRepresentations(chans_ptm[q, :, :], "process", "chi")
					)
				)
		else:
			chan_probs = rawchan
		if (submit.decoders)[0] == 3:
			mpinfo = CompleteDecoderKnowledge(submit.decoder_fraction, chan_probs, submit.eccs[0], option="full", nr_weights = None).astype(np.float64)
		else:
			# decoder is 4 i.e distribute by weight guided by Poisson
			nr_weights = np.load(fn.NRWeightsFile(submit, noise))[sample, :]
			mpinfo = CompleteDecoderKnowledge(submit.decoder_fraction, chan_probs, submit.eccs[0], option="weight", nr_weights = nr_weights).astype(np.float64)
	else:
		mpinfo = np.zeros(4**submit.eccs[0].N, dtype=np.float64)

	s_count = 0
	ss_count = 0
	norm_count = 0
	normphase_count = 0
	lst_count = 0
	for l in range(nlevels):
		# print("normalizer for code at level {}\n{}".format(l, submit.eccs[l].normalizer))
		nstabs = 2 ** (submit.eccs[l].N - submit.eccs[l].K)
		SS[ss_count : (ss_count + nstabs * nstabs)] = submit.eccs[l].syndsigns.ravel()
		normalizer[
			norm_count : (
				norm_count
				+ 2 ** (submit.eccs[l].N + submit.eccs[l].K) * submit.eccs[l].N
			)
		] = submit.eccs[l].normalizer.ravel()
		normphases_real[
			normphase_count : (
				normphase_count + 2 ** (submit.eccs[l].N + submit.eccs[l].K)
			)
		] = np.real(submit.eccs[l].normphases.ravel()).astype(np.float64)
		normphases_imag[
			normphase_count : (
				normphase_count + 2 ** (submit.eccs[l].N + submit.eccs[l].K)
			)
		] = np.imag(submit.eccs[l].normphases.ravel()).astype(np.float64)

		# Decoding preferences
		# Pass 3 in place of 4 as well -- same for backend
		if submit.decoders[l] == 4:
			decoders[l] = 3
		else:
			decoders[l] = submit.decoders[l]
		dclookups[s_count : (s_count + nstabs)] = submit.eccs[l].lookup[:, 0].astype(np.int32)

		s_count = s_count + nstabs
		ss_count = ss_count + nstabs * nstabs
		norm_count = (
			norm_count + 2 ** (submit.eccs[l].N + submit.eccs[l].K) * submit.eccs[l].N
		)
		normphase_count = normphase_count + 2 ** (submit.eccs[l].N + submit.eccs[l].K)

		# List of LST operators
		operators_LST[
			lst_count : (lst_count + 4 ** submit.eccs[l].N * submit.eccs[l].N)
		] = submit.eccs[l].PauliOperatorsLST.ravel()

		lst_count = lst_count + 4 ** submit.eccs[l].N * submit.eccs[l].N

	# Hybrid decoding -- channels that must be averaged in the intermediate levels
	if submit.hybrid == 0:
		decoderbins = np.zeros(1, dtype=np.int32)
		ndecoderbins = np.zeros(1, dtype=np.int32)
	else:
		chans = [
			np.prod(
				[submit.eccs[nlevels - l - 1].N for l in range(inter)], dtype=np.int
			)
			for inter in range(nlevels + 1)
		][::-1]
		decoderbins = np.zeros(sum(chans), dtype=np.int32)
		ndecoderbins = np.zeros(nlevels, dtype=np.int32)
		chan_count = 0
		for l in range(nlevels):
			decoderbins[chan_count : (chan_count + chans[l])] = submit.decoderbins[l][:]
			chan_count = chan_count + chans[l]
			ndecoderbins[l] = np.unique(submit.decoderbins[l]).shape[0]

	# Error channel parameters
	metrics = (ctypes.c_char_p * nmetrics)()
	metrics[:] = list(map(lambda str: str.encode("utf-8"), submit.metrics))
	if DEBUG == 1:
		SaveBenchmarkInput(
			nlevels,
			nkd,
			SS,
			normalizer,
			normphases_real,
			normphases_imag,
			submit.channel,
			iscorr,
			physical.astype(np.float64).ravel(),
			submit.rc,
			nmetrics,
			metrics,
			decoders,
			dclookups,
			mpinfo,
			operators_LST,
			submit.hybrid,
			decoderbins,
			ndecoderbins,
			submit.frame,
			len(submit.stats),
			submit.stats.astype(np.long),
			submit.nbins,
			submit.maxbin,
			submit.importance,
			refchan.astype(np.float64).ravel(),
		)

	_bmark = ctypes.cdll.LoadLibrary(os.path.abspath("simulate/bmark.so"))
	_bmark.Benchmark.argtypes = (
		ctypes.c_int,  # nlevels
		ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # nkd
		ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # SS
		ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # normalizer
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # normphases_real
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # normphases_imag
		ctypes.POINTER(ctypes.c_char),  # chname
		ctypes.c_int,  # iscorr
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # physical
		ctypes.c_int,
		ctypes.c_int,  # nmetrics
		ctypes.c_char_p * nmetrics,  # metrics
		ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # decoders
		ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # dclookups
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # mpinfo
		ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # operators_LST
		ctypes.c_int,  # hybrid
		ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # decoderbins
		ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # ndecoderbins
		ctypes.c_int,  # frame
		ctypes.c_int,  # nbreaks
		ndpointer(dtype=np.long, ndim=1, flags="C_CONTIGUOUS"),  # stats
		ctypes.c_int,  # nbins
		ctypes.c_int,  # maxbin
		ctypes.c_int,  # importance
		ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # refchan
		ctypes.c_double,  # infidelity
	)
	_bmark.Benchmark.restype = GetBMOutput(
		nlevels, nmetrics, nlogs, submit.nbins, len(submit.stats)
	)

	# print("_bmark.argtypes: {}".format(_bmark.Benchmark.argtypes))
	bout = _bmark.Benchmark(
		nlevels,  # arg 1
		nkd,  # arg 2
		SS,  # arg 3
		normalizer,  # arg 4
		normphases_real,  # arg 5
		normphases_imag,  # arg 6
		ctypes.c_char_p(submit.channel.encode("utf-8")),  # arg 7
		iscorr,  # arg 8
		physical.astype(np.float64).ravel(),  # arg 9
		submit.rc,  # arg 10
		nmetrics,  # arg 11
		metrics,  # arg 12
		decoders,  # arg 13
		dclookups,  # arg 14
		mpinfo, # arg 15
		operators_LST,  # arg 15
		submit.hybrid,  # arg 16
		decoderbins,  # arg 17
		ndecoderbins,  # arg 18
		submit.frame,  # arg 19
		len(submit.stats),  # arg 20
		submit.stats.astype(np.long),  # arg 21
		submit.nbins,  # arg 22
		submit.maxbin,  # arg 23
		submit.importance,  # arg 24
		refchan.astype(np.float64).ravel(),  # arg 25
		infidelity,  # arg 26
	)
	# print("bout: {}\nfields: {}".format(bout, bout._fields_))
	# The output arrays are all vectorized. We need to reshape them.
	logchans = Unpack(
		"logchans",
		bout.logchans,
		nlevels,
		nmetrics,
		nlogs,
		submit.nbins,
		len(submit.stats),
	)
	chanvar = Unpack(
		"chanvar",
		bout.chanvar,
		nlevels,
		nmetrics,
		nlogs,
		submit.nbins,
		len(submit.stats),
	)
	logerrs = Unpack(
		"logerrs",
		bout.logerrs,
		nlevels,
		nmetrics,
		nlogs,
		submit.nbins,
		len(submit.stats),
	)
	logvars = Unpack(
		"logvars",
		bout.logvars,
		nlevels,
		nmetrics,
		nlogs,
		submit.nbins,
		len(submit.stats),
	)
	bins = Unpack(
		"bins", bout.bins, nlevels, nmetrics, nlogs, submit.nbins, len(submit.stats)
	)
	running = Unpack(
		"running",
		bout.running,
		nlevels,
		nmetrics,
		nlogs,
		submit.nbins,
		len(submit.stats),
	)

	# Save the output to files.
	SaveAndChangeOwnership(fn.LogicalChannel(submit, noise, sample), logchans)
	SaveAndChangeOwnership(fn.LogChanVariance(submit, noise, sample), chanvar)
	for m in range(nmetrics):
		SaveAndChangeOwnership(fn.LogicalErrorRate(submit, noise, sample, submit.metrics[m]), logerrs[m, :])
		SaveAndChangeOwnership(fn.LogErrVariance(submit, noise, sample, submit.metrics[m]), logvars[m, :])
		SaveAndChangeOwnership(fn.SyndromeBins(submit, noise, sample, submit.metrics[m]), bins[m, :, :, :])
		SaveAndChangeOwnership(fn.RunningAverageCh(submit, noise, sample, submit.metrics[m]), running[m, :])
	# Decoder bins
	if submit.hybrid > 0:
		SaveAndChangeOwnership(fn.DecoderBins(submit, noise, sample), arr=None)
		with open(fn.DecoderBins(submit, noise, sample), "w") as df:
			for l in range(nlevels):
				df.write("%s\n" % ",".join(list(map(lambda num: "%d" % num, submit.decoderbins[l]))))
	# Free the memory allocated to bout by callin FreeBenchOut() method.
	return None


def SaveAndChangeOwnership(fname, arr=None):
	# Change ownership from user to group.
	# chown -h -R $USER:def-jemerson -- /projects/def-jemerson/chbank
	os.system("touch %s" % (fname))
	os.system("chown $USER:def-jemerson %s" % (fname))
	if arr is not None:
		np.save(fname, arr)
	return None


def SaveBenchmarkInput(
	nlevels,
	nkd,
	SS,
	normalizer,
	normphases_real,
	normphases_imag,
	channel,
	iscorr,
	physical,
	rc,
	nmetrics,
	metrics,
	decoders,
	dclookups,
	mpinfo,
	operators_LST,
	hybrid,
	decoderbins,
	ndecoderbins,
	frame,
	nstats,
	stats,
	nbins,
	maxbin,
	importance,
	refchan,
):
	# Save the inputs to the C backend's Benchmark function.
	# The inputs must be saved to text files in ./../input/debug_testing.
	# This is only for testing purposes.
	outdir = "./../input/debug_testing"
	os.system("mkdir -p %s" % (outdir))
	np.savetxt("%s/nkd.txt" % (outdir), nkd, fmt="%d", delimiter=" ", newline=" ")
	np.savetxt("%s/SS.txt" % (outdir), SS, fmt="%d", delimiter=" ", newline=" ")
	np.savetxt(
		"%s/normalizer.txt" % (outdir), normalizer, fmt="%d", delimiter=" ", newline=" "
	)
	np.savetxt(
		"%s/normphases_real.txt" % (outdir),
		normphases_real,
		fmt="%g",
		delimiter=" ",
		newline=" ",
	)
	np.savetxt(
		"%s/normphases_imag.txt" % (outdir),
		normphases_imag,
		fmt="%g",
		delimiter=" ",
		newline=" ",
	)
	np.savetxt(
		"%s/lookup.txt" % (outdir), dclookups, fmt="%d", delimiter=" ", newline=" "
	)
	np.savetxt(
		"%s/lst.txt" % (outdir), operators_LST, fmt="%d", delimiter=" ", newline=" "
	)
	np.savetxt(
		"%s/mpinfo.txt" % (outdir), mpinfo, fmt="%.21f", delimiter=" ", newline=" "
	)
	np.savetxt(
		"%s/physical.txt" % (outdir), physical, fmt="%.21f", delimiter=" ", newline=" "
	)
	np.savetxt("%s/stats.txt" % (outdir), stats, fmt="%d", delimiter=" ", newline=" ")
	return None
