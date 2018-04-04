#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False
from libc.stdlib cimport malloc, free
from cpython.string cimport PyString_AsString
from printfuns cimport PrintDoubleArray1D, PrintComplexArray2D, PrintDoubleArray2D, PrintIntArray2D, PrintIntArray1D
from constants cimport constants_t, InitConstants, FreeConstants
from memory cimport simul_t, AllocSimParams, FreeSimParams, AllocSimParamsQECC, FreeSimParamsQECC
from qecc cimport qecc_t, InitQECC, FreeQECC
from logmetrics cimport ComputeMetrics
from effective cimport Performance
from define import metrics as ml
import numpy as np
cimport numpy as np
from define import fnames as fn
from define import chanreps as crep

cpdef Benchmark(submit, rate, sample, physical, refchan):
	# This function is a wrapper for ComputeLogicalChannels(...)
	# The inputs to this function are class objects, just like a python function.
	# In this function, we will prepare the inputs to the pure C functions.
	# Then call the C functions.
	# Once done, we will write the output from the C structures, back to pure Python objects.
	# Initialize constants of the simulation
	cdef constants_t *consts = <constants_t *>malloc(sizeof(constants_t))
	InitConstants(consts)

	# Initialize values to the elements of the memory structure, depending on the inputs to the simulation
	# Parameters that depend only on the error correcting code
	cdef:
		int i, g, s, c, r, l
		qecc_t **qcode = <qecc_t **>malloc(sizeof(qecc_t *) * <int>(submit.levels))
	for l in range(<int>submit.levels):
		qcode[l] = <qecc_t *>malloc(sizeof(qecc_t))
		qcode[l][0].N = submit.eccs[l].N
		qcode[l][0].K = submit.eccs[l].K
		qcode[l][0].D = submit.eccs[l].D
		InitQECC(qcode[l], consts[0].nclifford)
		for s in range(qcode[l][0].nstabs):
			for g in range(qcode[l][0].nstabs):
				qcode[l][0].projector[s][g] = <int>(submit.eccs[l].syndsigns[s, g])
		for i in range(qcode[l][0].nlogs):
			for g in range(qcode[l][0].nstabs):
				for c in range(qcode[l].N):
					qcode[l][0].action[i][g][c] = submit.eccs[l].normalizer[i, g, c]
		for i in range(qcode[l][0].nlogs):
			for g in range(qcode[l][0].nstabs):
				qcode[l][0].phases[i][g] = submit.eccs[l].normphases[i, g]
		for i in range(2):
			for c in range(consts[0].nclifford):
				for r in range(qcode[l][0].nlogs):
					qcode[l][0].algebra[i][c][r] = submit.eccs[l].conjugations[i, c, r]

	# Parameters that are specific to the Montecarlo simulations to estimate the logical error rate
	# Place the physical channel in bank[0][0][0] and in logical[0]
	cdef simul_t **sims = <simul_t **>malloc(sizeof(simul_t *) * 2)
	for s in range(2):
		sims[s] = <simul_t *>malloc(sizeof(simul_t))
		# Read simulation constants from a file: const_<timestamp>.txt
		sims[s][0].nstats = <int>(submit.stats)
		sims[s][0].nlevels = <int>(submit.levels)
		sims[s][0].nmetrics = <int>(len(submit.metrics))
		sims[s][0].importance = <int>(submit.importance)
		sims[s][0].decoder = <int>submit.decoder
		sims[s][0].nbins = 50
		sims[s][0].maxbin = 30
		sims[s][0].threshold = 0.0 # Will be set once the level-0 metrics are computed.
		sims[s][0].cores = submit.cores[1]
		
		AllocSimParams(sims[s], qcode[0][0].nstabs, qcode[0][0].nlogs)
		AllocSimParamsQECC(sims[s], qcode[0][0].N, qcode[0][0].nstabs, qcode[0][0].nlogs)

		sims[s][0].chname = PyString_AsString(submit.channel)
		
		# Setting the metrics to be computed
		for i in range(sims[s][0].nmetrics):
			sims[s][0].metricsToCompute[i] = PyString_AsString(submit.metrics[i])

		# Error correcting frame
		# 0 -- Pauli frame, 1 -- Clifford frame, 2 -- Clifford last level
		if (submit.frame == 0):
			for i in range(sims[s].nlevels):
				sims[s][0].frames[i] = 4
		elif (submit.frame == 1):
			for i in range(sims[s].nlevels):
				sims[s][0].frames[i] = consts[0].nclifford
		else:
			for i in range(sims[s].nlevels):
				sims[s][0].frames[i] = 4
			sims[s][0].frames[sims[s][0].nlevels - 1] = consts[0].nclifford

	# sims[0] is the actual channel for which we want to compute logical error rates.
	# sims[1] contains a channel with higher noise rate, which is only used as a guide for sampling syndromes
	for r in range(qcode[0][0].nlogs):
		for c in range(qcode[0][0].nlogs):
			sims[0][0].physical[r][c] = <long double>(physical[r, c])
			sims[1][0].physical[r][c] = <long double>(refchan[r, c])

	# print("Going to start Performance")

	###################################

	Performance(qcode, sims, consts)
	
	###################################

	# print("Writing outputs on to files.")

	# Define the numpy objects to hold the outputs.
	cdef:
		# np.ndarray[np.long_t, ndim = 1] statsperlevel = np.zeros(sims[0][0].nlevels + 1, dtype = np.long)
		np.ndarray[np.longdouble_t, ndim = 2] metricValues = np.zeros((sims[0][0].nlevels + 1, sims[0][0].nmetrics), dtype = np.longdouble)
		# np.ndarray[np.longdouble_t, ndim = 2] variance = np.zeros((sims[0][0].nlevels + 1, sims[0][0].nmetrics), dtype = np.longdouble)
		np.ndarray[np.longdouble_t, ndim = 3] logical = np.zeros((sims[0][0].nlevels + 1, qcode[0][0].nlogs, qcode[0][0].nlogs), dtype = np.longdouble)
		np.ndarray[np.longdouble_t, ndim = 4] bins = np.zeros((sims[0][0].nmetrics, sims[0].nlevels + 1, sims[0][0].nbins, sims[0][0].nbins), dtype = np.longdouble)

	# Level-0 metrics and channels
	metricValues[0, :] = ml.ComputeNorms(crep.ConvertRepresentations(physical, "process", "choi"), submit.metrics)
	logical[0, :, :] = physical

	cdef int m, j
	for l in range(1, sims[0][0].nlevels + 1):
		for m in range(sims[0][0].nmetrics):
			metricValues[l, m] = sims[0][0].metricValues[l][m]
			# variance[1 + l, m] = sims[0][0].variance[l][m]
		for r in range(qcode[0][0].nlogs):
			for c in range(qcode[0][0].nlogs):
				logical[l, r, c] = sims[0][0].logical[l][r][c]
		# statsperlevel[l] = sims[0][0].statsperlevel[l]
		for m in range(sims[0].nmetrics):
			for i in range(sims[0][0].nbins):
				for j in range(sims[0][0].nbins):
					bins[m, l, i, j] = sims[0][0].bins[l][m][i][j]

	# print("Loaded data on to Python objects.")

	# Write all the data pertaining to the simulation
	# Record the logical channels
	path = fn.LogicalChannel(submit, rate, sample).split("/")
	fname = ("./../temp/channels/%s" % path[len(path) - 1])
	np.save(fname, logical)
	# Record the variance in the metric values
	# varfname = ("metrics/%s_var_%g_%d_%d_s%d.npy" % (submit.channel, rate, submit.stats, submit.levels, sample))
	# np.save(varfname, variance)
	# Metrics and syndrome bins
	for m in range(sims[0].nmetrics):
		# Save the metrics onto respective files
		path = fn.LogicalErrorRate(submit, rate, sample, submit.metrics[m]).split("/")
		fname = ("./../temp/metrics/%s" % path[len(path) - 1])
		np.save(fname, metricValues[:, m])
		# Record the syndrome bins
		path = fn.SyndromeBins(submit, rate, sample, submit.metrics[m]).split("/")
		fname = ("./../temp/metrics/%s" % path[len(path) - 1])
		np.save(fname, bins[m, :, :, :])

	###################################

	# print("Freeing memory.")

	for s in range(2):
		FreeSimParamsQECC(sims[s], qcode[0][0].N, qcode[0][0].nstabs, qcode[0][0].nlogs)
		FreeSimParams(sims[s], qcode[0][0].nstabs, qcode[0][0].nlogs)
		free(<void *>sims[s])
	free(<void *>sims)

	for l in range(<int>submit.levels):
		FreeQECC(qcode[l], consts[0].nclifford)
		free(<void *>qcode[l])
	free(<void *>qcode)

	FreeConstants(consts)
	free(<void *>consts)

	# print("Done benchmarking.")

	return 0
