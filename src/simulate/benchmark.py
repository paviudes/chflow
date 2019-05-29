import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from define import fnames as fn

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
		_fields_ = [("logchans", ctypes.POINTER(ctypes.c_double * ((nlevels + 1) * nlogs * nlogs))),
					("chanvar", ctypes.POINTER(ctypes.c_double * ((nlevels + 1) * nlogs * nlogs))),
					("logerrs", ctypes.POINTER(ctypes.c_double * ((nlevels + 1) * nmetrics))),
					("logvars", ctypes.POINTER(ctypes.c_double * ((nlevels + 1) * nmetrics))),
					("bins", ctypes.POINTER(ctypes.c_int * (nmetrics * (nlevels + 1) * nbins * nbins))),
					("running", ctypes.POINTER(ctypes.c_double * (nmetrics * nbreaks)))]
	return BMOutput

def Unpack(varname, pointer, nlevels, nmetrics, nlogs, nbins, nbreaks):
	# Convert the 1D pointer representation of an BMOutput variable to an array of the required shape.
	types = {"logchans": [ctypes.c_double, (nlevels + 1) * nlogs * nlogs, [nlevels + 1, nlogs, nlogs]],
			 "chanvar": [ctypes.c_double, (nlevels + 1) * nlogs * nlogs, [nlevels + 1, nlogs, nlogs]],
			 "logerrs": [ctypes.c_double, (nlevels + 1) * nmetrics, [nmetrics, nlevels + 1]],
			 "logvars": [ctypes.c_double, (nlevels + 1) * nmetrics, [nmetrics, nlevels + 1]],
			 "bins": [ctypes.c_int, (nlevels + 1) * nbins * nbins, [nmetrics, nlevels + 1, nbins, nbins]],
			 "running": [ctypes.c_double, nmetrics * nbreaks, [nmetrics, nbreaks]]}
	ctvec = ctypes.cast(pointer, ctypes.POINTER(types[varname][0] * types[varname][1])).contents
	reshaped = np.ctypeslib.as_array(ctvec).reshape(types[varname][2])
	return reshaped

def Benchmark(submit, noise, sample, physical, refchan):
	# This is a wrapper function to the C function in benchmark.c
	# We will prepare the inputs to the C function as numpy arrays and use ctypes to convert them to C pointers.
	# After the benchmarking process is run, we will then transform the output form C pointers to numpy arrays.
	nmetrics = len(submit.metrics)
	nlevels = submit.levels
	nlogs = 4**(submit.eccs[nlevels - 1].K)
	nkd = np.zeros(3 * nlevels, dtype = np.int32)
	nstabs = 0
	for l in range(nlevels):
		(nkd[3 * l], nkd[3 * l + 1], nkd[3 * l + 2]) = (submit.eccs[l].N, submit.eccs[l].K, submit.eccs[l].D)
		nstabs = nstabs + 2**(submit.eccs[l].N - submit.eccs[l].K)
	SS = np.zeros(np.sum([4**(submit.eccs[l].N - submit.eccs[l].K) for l in range(nlevels)]), dtype = np.int32)
	normalizer = np.zeros(np.sum([2**(submit.eccs[l].N + submit.eccs[l].K) * submit.eccs[l].N for l in range(nlevels)]), dtype = np.int32)
	normphases_real = np.zeros(np.sum([2**(submit.eccs[l].N + submit.eccs[l].K) for l in range(nlevels)]), dtype = np.float64)
	normphases_imag = np.zeros(np.sum([2**(submit.eccs[l].N + submit.eccs[l].K) for l in range(nlevels)]), dtype = np.float64)
	ss_count = 0
	norm_count = 0
	normphase_count = 0
	for l in range(nlevels):
		nstabs = 2**(submit.eccs[l].N - submit.eccs[l].K)
		SS[ss_count:(ss_count + nstabs * nstabs)] = submit.eccs[l].syndsigns.ravel()
		ss_count = ss_count + nstabs * nstabs
		normalizer[norm_count:(norm_count + 2**(submit.eccs[l].N + submit.eccs[l].K) * submit.eccs[l].N)] = submit.eccs[l].normalizer.ravel()
		norm_count = norm_count + 2**(submit.eccs[l].N + submit.eccs[l].K) * submit.eccs[l].N
		normphases_real[normphase_count:(normphase_count + 2**(submit.eccs[l].N + submit.eccs[l].K))] = np.real(submit.eccs[l].normphases.ravel()).astype(np.float64)
		normphases_imag[normphase_count:(normphase_count + 2**(submit.eccs[l].N + submit.eccs[l].K))] = np.imag(submit.eccs[l].normphases.ravel()).astype(np.float64)
		normphase_count = normphase_count + 2**(submit.eccs[l].N + submit.eccs[l].K)

	# Hybrid decoding -- channels that must be averaged in the intermediate levels
	if (submit.hybrid == 0):
		decoderbins = np.zeros(1, dtype = np.int32)
		ndecoderbins = np.zeros(1, dtype = np.int32)
	else:
		chans = [np.prod([submit.eccs[nlevels-l-1].N for l in range(inter)], dtype=np.int) 
		for inter in range(nlevels+1)][::-1]
		decoderbins = np.zeros(sum(chans), dtype = np.int32)
		ndecoderbins = np.zeros(nlevels, dtype = np.int32)
		chan_count = 0
		for l in range(nlevels):
			decoderbins[chan_count:(chan_count + chans[l])] = submit.decoderbins[l][:]
			chan_count = chan_count + chans[l]
			ndecoderbins[l] = np.unique(submit.decoderbins[l]).shape[0]

	# Error channel parameters
	metrics = (ctypes.c_char_p * nmetrics)()
	metrics[:] = list(map(lambda str: str.encode('utf-8'), submit.metrics))

	_bmark = ctypes.cdll.LoadLibrary(os.path.abspath("simulate/bmark.so"))
	_bmark.Benchmark.argtypes = (ctypes.c_int, # nlevels
								 ndpointer(dtype = np.int32, ndim = 1, flags = 'C_CONTIGUOUS'), # nkd
								 ndpointer(dtype = np.int32, ndim = 1, flags = 'C_CONTIGUOUS'), # SS
								 ndpointer(dtype = np.int32, ndim = 1, flags = 'C_CONTIGUOUS'), # normalizer
								 ndpointer(dtype = np.float64, ndim = 1, flags = 'C_CONTIGUOUS'), # normphases_real
								 ndpointer(dtype = np.float64, ndim = 1, flags = 'C_CONTIGUOUS'), # normphases_imag
								 ctypes.POINTER(ctypes.c_char), # chname
								 ndpointer(dtype = np.float64, ndim = 1, flags = 'C_CONTIGUOUS'), # physical
								 ctypes.c_int, # nmetrics
								 ctypes.c_char_p * nmetrics, # metrics
								 ctypes.c_int, # hybrid
								 ndpointer(dtype = np.int32, ndim = 1, flags = 'C_CONTIGUOUS'), # decoderbins
								 ndpointer(dtype = np.int32, ndim = 1, flags = 'C_CONTIGUOUS'), # ndecoderbins
								 ctypes.c_int, # frame
								 ctypes.c_int, # nbreaks
								 ndpointer(dtype = np.long, ndim = 1, flags = 'C_CONTIGUOUS'), # stats
								 ctypes.c_int, # nbins
								 ctypes.c_int, # maxbin
								 ctypes.c_int, # importance
								 ndpointer(dtype = np.float64, ndim = 1, flags = 'C_CONTIGUOUS'), # refchan
								)
	_bmark.Benchmark.restype = GetBMOutput(nlevels, nmetrics, nlogs, submit.nbins, len(submit.stats))
	# print("_bmark.argtypes: {}".format(_bmark.Benchmark.argtypes))
	bout = _bmark.Benchmark(nlevels, # arg 1
							nkd, # arg 2
							SS, # arg 3
							normalizer, # arg 4
							normphases_real, # arg 5
							normphases_imag, # arg 6
							ctypes.c_char_p(submit.channel.encode('utf-8')), # arg 7
							physical.astype(np.float64).ravel(), # arg 8
							ctypes.c_int(nmetrics), # arg 9
							metrics, # arg 10
							submit.hybrid, # arg 11
							decoderbins, # arg 12
							ndecoderbins, # arg 13
							submit.frame, # arg 14
							len(submit.stats), # arg 15
							submit.stats.astype(np.long), # arg 16
							submit.nbins, # arg 17
							submit.maxbin, # arg 18
							submit.importance, # arg 19
							refchan.astype(np.float64).ravel()) # arg 20
	# print("bout: {}\nfields: {}".format(bout, bout._fields_))
	# The output arrays are all vectorized. We need to reshape them.
	logchans = Unpack("logchans", bout.logchans, nlevels, nmetrics, nlogs, submit.nbins, len(submit.stats))
	chanvar = Unpack("chanvar", bout.chanvar, nlevels, nmetrics, nlogs, submit.nbins, len(submit.stats))
	logerrs = Unpack("logerrs", bout.logerrs, nlevels, nmetrics, nlogs, submit.nbins, len(submit.stats))
	logvars = Unpack("logvars", bout.logvars, nlevels, nmetrics, nlogs, submit.nbins, len(submit.stats))
	bins = Unpack("bins", bout.bins, nlevels, nmetrics, nlogs, submit.nbins, len(submit.stats))
	running = Unpack("running", bout.running, nlevels, nmetrics, nlogs, submit.nbins, len(submit.stats))

	# Save the output to files.
	np.save(fn.LogicalChannel(submit, noise, sample), logchans)
	np.save(fn.LogChanVariance(submit, noise, sample), chanvar)
	for m in range(nmetrics):
		np.save(fn.LogicalErrorRate(submit, noise, sample, submit.metrics[m]), logerrs[m, :])
		np.save(fn.LogErrVariance(submit, noise, sample, submit.metrics[m]), logvars[m, :])
		np.save(fn.SyndromeBins(submit, noise, sample, submit.metrics[m]), bins[m, :, :, :])
		np.save(fn.RunningAverageCh(submit, noise, sample, submit.metrics[m]), running[m, :])

	# Free the memory allocated to bout by callin FreeBenchOut() method.
	return None
