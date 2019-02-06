#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False
from libc.stdlib cimport malloc, free
from cpython.string cimport PyString_AsString
from simulate.printfuns cimport PrintDoubleArray1D, PrintComplexArray2D, PrintDoubleArray2D, PrintIntArray2D, PrintIntArray1D
from simulate.constants cimport constants_t, InitConstants, FreeConstants
from simulate.memory cimport simul_t, AllocSimParams, FreeSimParams, AllocSimParamsQECC, FreeSimParamsQECC
from simulate.qecc cimport qecc_t, InitQECC, FreeQECC
from simulate.logmetrics cimport ComputeMetrics
from simulate.effective cimport Performance
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
		int i, j, g, s, c, r, l
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
	# print("Initializing simulation Parameters for submit.stats.shape[0] = %d" % (submit.stats.shape[0]))
	cdef simul_t **sims = <simul_t **>malloc(sizeof(simul_t *) * 2)
	for s in range(2):
		sims[s] = <simul_t *>malloc(sizeof(simul_t))
		
		sims[s][0].runstats = <int *>malloc(sizeof(int) * (1 + submit.stats.shape[0]))
		sims[s][0].runstats[0] = submit.stats.shape[0]
		for i in range(submit.stats.shape[0]):
			sims[s][0].runstats[i + 1] = <int>(submit.stats[i])

		sims[s][0].nstats = <int>(submit.stats[submit.stats.shape[0] - 1])

		# PrintIntArray1D(sims[s][0].runstats, "sims[s][0].runstats", sims[s][0].runstats[0] + 1)
		# print("sims[%d][0].nstats = %d" % (s, sims[s][0].nstats))

		sims[s][0].nlevels = <int>(submit.levels)
		sims[s][0].nmetrics = <int>(len(submit.metrics))

		# print("nmetrics = %d" % (sims[s][0].nmetrics))
		
		sims[s][0].runavg = <long double ***>malloc(sizeof(long double **) * 2)
		for j in range(2):
			sims[s][0].runavg[j] = <long double **>malloc(sizeof(long double *) * sims[s][0].nmetrics)
			for m in range(sims[s][0].nmetrics):
				sims[s][0].runavg[j][m] = <long double *>malloc(sizeof(long double) * (1 + submit.stats.shape[0]))
				for i in range(1 + submit.stats.shape[0]):
					sims[s][0].runavg[j][m][i + 1] = 0.0
		
		# PrintDoubleArray2D(sims[s][0].runavg, "sims[s][0].runavg", sims[s][0].nmetrics, submit.stats.shape[0])

		sims[s][0].importance = <int>(submit.importance)
		sims[s][0].decoder = <int>submit.decoder
		sims[s][0].nbins = 50
		sims[s][0].maxbin = 30
		# sims[s][0].threshold = 0.0 # Will be set once the level-0 metrics are computed.
		# sims[s][0].cores = submit.cores[1]
		
		# print("Going to call AllocSimParams")
		AllocSimParams(sims[s], qcode[0][0].nstabs, qcode[0][0].nlogs)
		# print("Going to call AllocSimParamsQECC with qcode[0][0].N = %d, qcode[0][0].nstabs = %d, qcode[0][0].nlogs = %d" % (qcode[0][0].N, qcode[0][0].nstabs, qcode[0][0].nlogs))
		AllocSimParamsQECC(sims[s], qcode[0][0].N, qcode[0][0].nstabs, qcode[0][0].nlogs)
		# print("allocated")
		# PrintDoubleArray2D(sims[s][0].variance, "sims[s][0].variance", sims[0][0].nlevels + 1, sims[0][0].nmetrics)

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
	
	print("Going to start Performance")

	###################################

	Performance(qcode, sims, consts)
	
	###################################

	print("Writing outputs on to files.")

	# Define the numpy objects to hold the outputs.
	cdef:
		# np.ndarray[np.long_t, ndim = 1] statsperlevel = np.zeros(sims[0][0].nlevels + 1, dtype = np.long)
		np.ndarray[np.longdouble_t, ndim = 2] metricValues = np.zeros((sims[0][0].nlevels + 1, sims[0][0].nmetrics), dtype = np.longdouble)
		np.ndarray[np.longdouble_t, ndim = 4] running = np.zeros((sims[0][0].nmetrics, 2, submit.stats.shape[0], 1 + <int>(sims[0][0].importance == 2)), dtype = np.longdouble)
		np.ndarray[np.longdouble_t, ndim = 2] variance = np.zeros((sims[0][0].nlevels + 1, sims[0][0].nmetrics + qcode[0][0].nlogs * qcode[0][0].nlogs), dtype = np.longdouble)
		np.ndarray[np.longdouble_t, ndim = 3] logical = np.zeros((sims[0][0].nlevels + 1, qcode[0][0].nlogs, qcode[0][0].nlogs), dtype = np.longdouble)
		np.ndarray[np.longdouble_t, ndim = 4] bins = np.zeros((sims[0][0].nmetrics, sims[0].nlevels + 1, sims[0][0].nbins, sims[0][0].nbins), dtype = np.longdouble)

	# PrintDoubleArray2D(sims[0][0].runavg[0], "running average[0]", sims[0][0].nmetrics, sims[0][0].runstats[0])
	# PrintDoubleArray2D(sims[0][0].runavg[1], "running average[1]", sims[0][0].nmetrics, sims[0][0].runstats[0])

	# Level-0 metrics and channels
	metricValues[0, :] = ml.ComputeNorms(crep.ConvertRepresentations(physical, "process", "choi"), submit.metrics, submit.channel)
	logical[0, :, :] = physical

	for s in range(1 + <int>(sims[0][0].importance == 2)):
		for m in range(sims[0][0].nmetrics):
			for i in range(submit.stats.shape[0]):
				for j in range(2):
					running[m, j, i, s] = sims[s][0].runavg[j][m][i + 1]
	
	# print("Copied the running averags")

	for l in range(1, sims[0][0].nlevels + 1):
		# print("Level %d" % (l))
		# PrintDoubleArray2D(sims[0][0].variance, "sims[0][0].variance", sims[0][0].nlevels + 1, sims[0][0].nmetrics)
		for m in range(sims[0][0].nmetrics):
			metricValues[l, m] = sims[0][0].metricValues[l][m]
			variance[l, m] = sims[0][0].variance[l][m]
		for r in range(qcode[0][0].nlogs):
			for c in range(qcode[0][0].nlogs):
				logical[l, r, c] = sims[0][0].logical[l][r][c]
				variance[l, sims[0][0].nmetrics + r * qcode[0][0].nlogs + c] = sims[0][0].variance[l][sims[0][0].nmetrics + r * qcode[0][0].nlogs + c]
		# print("level %d: copied metric values and associated variance" % (l))
		# statsperlevel[l] = sims[0][0].statsperlevel[l]
		for m in range(sims[0].nmetrics):
			for i in range(sims[0][0].nbins):
				for j in range(sims[0][0].nbins):
					bins[m, l, i, j] = sims[0][0].bins[l][m][i][j]

	# print("Loaded data on to Python objects.")

	# Write all the data pertaining to the simulation
	# Record the logical channels
	np.save(fn.LogicalChannel(submit, rate, sample), logical)
	# Metrics and syndrome bins
	for m in range(sims[0].nmetrics):
		# Save the metrics onto respective files
		np.save(fn.LogicalErrorRate(submit, rate, sample, submit.metrics[m]), metricValues[:, m])
		# Record the running averages
		np.save(fn.RunningAverageCh(submit, rate, sample, submit.metrics[m]), running[m, :, :, :])
		# Record the syndrome bins
		np.save(fn.SyndromeBins(submit, rate, sample, submit.metrics[m]), bins[m, :, :, :])
		# Record the variance in metric values
		np.save(fn.LogErrVariance(submit, rate, sample, submit.metrics[m]), variance[:, m])
	# Record the variance in logical channel
	np.save(fn.LogChanVariance(submit, rate, sample), variance[:, len(submit.metrics):])


	###################################

	# print("Freeing memory.")
	for s in range(2):
		# Free running average data structures
		for j in range(2):
			for m in range(sims[s][0].nmetrics):
				free(<void *>sims[s][0].runavg[j][m])
			free(<void *>sims[s][0].runavg[j])
		free(<void *>sims[s][0].runavg)
		free(<void *>sims[s][0].runstats)
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
