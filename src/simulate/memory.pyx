#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False
from libc.stdlib cimport malloc, free
from constants cimport constants_t
from printfuns cimport PrintDoubleArray1D, PrintComplexArray2D, PrintDoubleArray2D, PrintIntArray2D, PrintIntArray1D, PrintLongArray2D, PrintLongArray1D
from qecc cimport qecc_t

cdef extern from "math.h":
	int ceill(long double num)

cdef int AllocSimParamsQECC(simul_t *simul, int nqecc, int nstabs, int nlogs) nogil:
	# Allocate memory for parameters of the simulation structure that depend upon the QECC used.
	# This memeory will be reallocated everytime there is a new QECC, i.e, at a new concatenation level.
	# These parameters are
	# virtual, logical, syndprobs, cumulative, levelZeroSynds, levelZeroCumulative, levelZeroImpDist, levelZeroImpCumul, process, corrections, effprocess, effective, levelZeroChannels
	cdef int c, l, q, r, s
	## Logical channels at intermediate concatenation levels
	simul[0].virtual = <long double ***>malloc(nqecc * sizeof(long double **))
	for q in range(nqecc):
		simul[0].virtual[q] = <long double **>malloc(nlogs * sizeof(long double *))
		for r in range(nlogs):
			simul[0].virtual[q][r] = <long double *>malloc(nlogs * sizeof(long double))
			for c in range(nlogs):
				simul[0].virtual[q][r][c] = 0.0

	## Syndrome sampling
	simul[0].syndprobs = <long double *>malloc(nstabs * sizeof(long double))
	simul[0].cumulative = <long double *>malloc(nstabs * sizeof(long double))
	for s in range(nstabs):
		simul[0].syndprobs[s] = 0.0
		simul[0].cumulative[s] = 0.0
	## Quantum error correction
	simul[0].process = <long double ****>malloc(nlogs * sizeof(long double ***))
	for r in range(nlogs):
		simul[0].process[r] = <long double ***>malloc(nlogs * sizeof(long double **))
		for c in range(nlogs):
			simul[0].process[r][c] = <long double **>malloc(nstabs * sizeof(long double *))
			for i in range(nstabs):
				simul[0].process[r][c][i] = <long double *>malloc(nstabs * sizeof(long double))
				for s in range(nstabs):
					simul[0].process[r][c][i][s] = 0.0
	simul[0].corrections = <int *>malloc(nstabs * sizeof(int))
	for s in range(nstabs):
		simul[0].corrections[s] = 0
	simul[0].effective = <complex128_t ***>malloc(nstabs * sizeof(complex128_t **))
	simul[0].effprocess = <long double ***>malloc(nstabs * sizeof(long double **))
	for s in range(nstabs):
		simul[0].effective[s] = <complex128_t **>malloc(nlogs * sizeof(complex128_t *))
		simul[0].effprocess[s] = <long double **>malloc(nlogs * sizeof(long double *))
		for r in range(nlogs):
			simul[0].effective[s][r] = <complex128_t *>malloc(nlogs * sizeof(complex128_t))
			simul[0].effprocess[s][r] = <long double *>malloc(nlogs * sizeof(long double))
			for c in range(nlogs):
				simul[0].effective[s][r][c] = 0.0 + 0.0 * 1j
				simul[0].effprocess[s][r][c] = 0.0
	return 0

cdef int AllocSimParams(simul_t *simul, int nstabs, int nlogs) nogil:
	# initialize the elements that pertain to the montecarlo simulation of channels
	cdef int s, l, r, c, m
	## Physical channels
	simul[0].physical = <long double **>malloc(nlogs * sizeof(long double *))
	for r in range(nlogs):
		simul[0].physical[r] = <long double *>malloc(nlogs * sizeof(long double))
		for c in range(nlogs):
			simul[0].physical[r][c] = 0.0
	simul[0].chname = <char *>malloc(100 * sizeof(char))
	## Metrics to be computed at every level
	simul[0].metricsToCompute = <char **>malloc(simul[0].nmetrics * sizeof(char *))
	for m in range(simul[0].nmetrics):
		simul[0].metricsToCompute[m] = <char *>malloc(100 * sizeof(char))
	simul[0].metricValues = <long double **>malloc((simul[0].nlevels + 1) * sizeof(long double *))
	for l in range(simul[0].nlevels + 1):
		simul[0].metricValues[l] = <long double *>malloc(simul[0].nmetrics * sizeof(long double))
		for m in range(simul[0].nmetrics):
			simul[0].metricValues[l][m] = 0.0
	## Average logical channel at top level
	simul[0].logical = <long double ***>malloc((simul[0].nlevels + 1) * sizeof(long double **))
	for l in range(simul[0].nlevels + 1):
		simul[0].logical[l] = <long double **>malloc(nlogs * sizeof(long double *))
		for r in range(nlogs):
			simul[0].logical[l][r] = <long double *>malloc(nlogs * sizeof(long double))
			for c in range(nlogs):
				simul[0].logical[l][r][c] = 0.0
	## Syndrome sampling
	simul[0].levelOneSynds = <long double *>malloc(nstabs * sizeof(long double))
	simul[0].levelOneImpDist = <long double *>malloc(nstabs * sizeof(long double))
	simul[0].levelOneCumul = <long double *>malloc(nstabs * sizeof(long double))
	simul[0].levelOneImpCumul = <long double *>malloc(nstabs * sizeof(long double))
	for s in range(nstabs):
		simul[0].levelOneSynds[s] = 0.0
		simul[0].levelOneImpDist[s] = 0.0
		simul[0].levelOneCumul[s] = 0.0
		simul[0].levelOneImpCumul[s] = 0.0
	simul[0].statsperlevel = <long *>malloc((simul[0].nlevels + 1) * sizeof(long))
	for l in range(simul[0].nlevels + 1):
		simul[0].statsperlevel[l] = 0
	# Upper and lower limits for the probability of the outlier syndromes.
	simul[0].outlierprobs = <long double *>malloc(sizeof(long double) * 2)
	simul[0].outlierprobs[0] = 0.5
	simul[0].outlierprobs[1] = 0.6
	## Syndrome-metric bins
	simul[0].bins = <int ****>malloc((simul[0].nlevels + 1) * sizeof(int ***))
	for l in range(simul[0].nlevels + 1):
		simul[0].bins[l] = <int ***>malloc((simul[0].nmetrics) * sizeof(int **))
		for m in range(simul.nmetrics):
			simul[0].bins[l][m] = <int **>malloc(simul[0].nbins * sizeof(int *))
			for r in range(simul[0].nbins):
				simul[0].bins[l][m][r] = <int *>malloc(simul[0].nbins * sizeof(int))
				for c in range(simul[0].nbins):
					simul[0].bins[l][m][r][c] = 0
	## Variance measures
	simul[0].variance = <long double **>malloc((simul[0].nlevels + 1) * sizeof(long double *))
	for l in range(simul[0].nlevels + 1):
		simul[0].variance[l] = <long double *>malloc(simul[0].nmetrics * sizeof(long double))
		for m in range(simul[0].nmetrics):
			simul[0].variance[l][m] = 0.0
	## Quantum error correction
	simul[0].levelOneChannels = <long double ***>malloc(nstabs * sizeof(long double **))
	for s in range(nstabs):
		simul[0].levelOneChannels[s] = <long double **>malloc(nlogs * sizeof(long double *))
		for r in range(nlogs):
			simul[0].levelOneChannels[s][r] = <long double *>malloc(nlogs * sizeof(long double))
			for c in range(nlogs):
				simul[0].levelOneChannels[s][r][c] = 0.0
	simul[0].frames = <int *>malloc(simul[0].nlevels * sizeof(int))
	for l in range(simul[0].nlevels):
		simul[0].frames[l] = 0
	return 0


cdef int FreeSimParamsQECC(simul_t *simul, int nqecc, int nstabs, int nlogs) nogil:
	# free the memory allocated to simulation parameters that depend on the QECC
	# These parameters are
	# virtual, logical, syndprobs, cumulative, levelZeroSynds, levelZeroCumulative, levelZeroImpDist, levelZeroImpCumul, process, corrections, effprocess, effective, levelZeroChannels
	cdef int c, l, q, r, s
	## Logical channels at intermediate levels
	for q in range(nqecc):
		for r in range(nlogs):
			free(<void *>simul[0].virtual[q][r])
		free(<void *>simul[0].virtual[q])
	free(<void *>simul[0].virtual)	
	## Quantum error correction
	for r in range(nlogs):
		for c in range(nlogs):
			for s in range(nstabs):
				free(<void *>simul[0].process[r][c][s])
			free(<void *>simul[0].process[r][c])
		free(<void *>simul[0].process[r])
	free(<void *>simul[0].process)
	free(<void *>simul[0].corrections)
	for s in range(nstabs):
		for r in range(nlogs):
			free(<void *>simul[0].effective[s][r])
			free(<void *>simul[0].effprocess[s][r])
		free(<void *>simul[0].effective[s])
		free(<void *>simul[0].effprocess[s])
	free(<void *>simul[0].effective)
	free(<void *>simul[0].effprocess)
	## Syndrome sampling
	free(<void *>simul[0].syndprobs)
	free(<void *>simul[0].cumulative)
	return 0

cdef int FreeSimParams(simul_t *simul, int nstabs, int nlogs) nogil:
	# free memory allocated to the simulation structure
	cdef int l, i, s, r, c, g, m
	## Physical channels
	for r in range(nlogs):
		free(<void *>simul[0].physical[r])
	free(<void *>simul[0].physical)
	# free(<void *>simul[0].chname)
	## Metrics to be computed at logical levels
	free(<void *>simul[0].metricsToCompute)
	for i in range(simul[0].nlevels + 1):
		free(<void *>simul[0].metricValues[i])
	free(<void *>simul[0].metricValues)
	## Average logical channel at the top level
	for l in range(simul[0].nlevels + 1):
		for r in range(nlogs):
			free(<void *>simul[0].logical[l][r])
		free(<void *>simul[0].logical[l])
	free(<void *>simul[0].logical)
	## Syndrome sampling
	free(<void *>simul[0].statsperlevel)
	free(<void *>simul[0].outlierprobs)
	free(<void *>simul[0].levelOneSynds)
	free(<void *>simul[0].levelOneCumul)
	free(<void *>simul[0].levelOneImpDist)
	free(<void *>simul[0].levelOneImpCumul)
	## Variance measure
	for i in range(simul[0].nlevels + 1):
		free(<void *>simul[0].variance[i])
	free(<void *>simul[0].variance)
	## Syndrome metric bins
	for i in range(simul[0].nlevels + 1):
		for m in range(simul.nmetrics):
			for r in range(simul[0].nbins):
				free(<void *>simul[0].bins[i][m][r])
			free(<void *>simul[0].bins[i][m])
		free(<void *>simul[0].bins[i])
	free(<void *>simul[0].bins)
	## Quantum error correction
	for s in range(nstabs):
		for r in range(nlogs):
			free(<void *>simul[0].levelOneChannels[s][r])
		free(<void *>simul[0].levelOneChannels[s])
	free(<void *>simul[0].levelOneChannels)
	free(<void *>simul[0].frames)
	return 0


cdef int CopySimulation(simul_t *copyto, simul_t *copyfrom, qecc_t *qecc):
	# Copy the elements of a simulation structure from an old to a new one.
	cdef:
		int i, j
	copyto[0].nlevels = copyfrom[0].nlevels
	copyto[0].nstats = copyfrom[0].nstats
	copyto[0].nmetrics = copyfrom[0].nmetrics
	copyto[0].importance = copyfrom[0].importance
	copyto[0].decoder = copyfrom[0].decoder
	copyto[0].nbins = copyfrom[0].nbins
	copyto[0].maxbin = copyfrom[0].maxbin
	copyto[0].threshold = copyfrom[0].threshold
	AllocSimParams(copyto, qecc[0].nstabs, qecc[0].nlogs)
	AllocSimParamsQECC(copyto, qecc[0].N, qecc[0].nstabs, qecc[0].nlogs)
	copyto[0].chname = copyfrom[0].chname
	# Setting the metrics to be computed
	for i in range(copyto[0].nmetrics):
		copyto[0].metricsToCompute[i] = copyfrom[0].metricsToCompute[i]
	# Error correcting frame
	for i in range(copyto[0].nlevels):
		copyto[0].frames[i] = copyfrom[0].frames[i]
	# Physical channel
	for i in range(qecc[0].nlogs):
		for j in range(qecc[0].nlogs):
			copyto[0].physical[i][j] = copyfrom[0].physical[i][j]
	return 0


cdef int MergeSimulations(simul_t *parent, simul_t *child, int nlogs):
	# Merge the child simulation with the parent.
	# Add the average failure rates, bins, average logical channels, variance, statistics per level.
	cdef int l, m, i, j
	for l in range(parent[0].nlevels):
		# 1. statsperlevel
		parent[0].statsperlevel[l + 1] = (parent[0].statsperlevel[l + 1]) * (<int>(l > 0)) + child[0].statsperlevel[l + 1]
		# 2. average logical channels
		for i in range(nlogs):
			for j in range(nlogs):
				parent[0].logical[l + 1][i][j] = (parent[0].logical[l + 1][i][j]) * (<int>(l > 0)) + child[0].logical[l + 1][i][j]
		# 3. average metric values and variance
		for m in range(parent[0].nmetrics):
			parent[0].metricValues[l + 1][m] = (parent[0].metricValues[l + 1][m]) * (<int>(l > 0)) + child[0].metricValues[l + 1][m]
			parent[0].variance[l + 1][m] = (parent[0].variance[l + 1][m]) * (<int>(l > 0)) + child[0].variance[l + 1][m]
		# 4. Binning information
		for m in range(parent[0].nmetrics):
			for i in range(parent[0].nbins):
				for j in range(parent[0].nbins):
					parent[0].bins[l + 1][m][i][j] = (parent[0].bins[l + 1][m][i][j]) * (<int>(l > 0)) + child[0].bins[l + 1][m][i][j]
	# PrintLongArray1D(parent[0].statsperlevel, "statsperlevel", parent[0].nlevels + 1)
	return 0



cdef int CountIndepLogicalChannels(int *chans, qecc_t **qecc, int nlevels) nogil:
	# Determine the number of independent logical channels at every level, that determine the logical channels of higher levels.
	cdef:
		int i, l, nchans
	for l in range(nlevels):
		chans[l] = 1
		for i in range(l + 1, nlevels):
			chans[l] = chans[l] * qecc[i][0].N
	nchans = chans[0]
	return nchans


cdef int MemManageChannels(long double ******channels, int nbatches, qecc_t **qecc, int nlevels, int importance, int tofree):
	# Allocate and Free memory for the tree of lower-level channels which determine a logical channel.
	cdef:
		int b, i, l, s, sb, nchans
		int *chans = <int *>malloc(sizeof(int) * nlevels)
	nchans = CountIndepLogicalChannels(chans, qecc, nlevels)
	if (tofree == 0):
		# Allocate memory
		for b in range(nbatches):
			channels[b] = <long double *****>malloc(sizeof(long double ****) * nlevels)
			for l in range(nlevels):
				channels[b][l] = <long double ****>malloc(sizeof(long double ***) * chans[l])
				for sb in range(chans[l]):
					channels[b][l][sb] = <long double ***>malloc(sizeof(long double **) * (1 + <int>(importance == 2)))
					for s in range(1 + <int>(importance == 2)):
						channels[b][l][sb][s] = <long double **>malloc(sizeof(long double *) * (1 + qecc[l][0].nlogs))
						for i in range(qecc[l][0].nlogs):
							channels[b][l][sb][s][i] = <long double *>malloc(sizeof(long double) * qecc[l][0].nlogs)
						channels[b][l][sb][s][qecc[l][0].nlogs] = <long double *>malloc(sizeof(long double) * 3)
	else:
		# free memory
		for b in range(nbatches):
			for l in range(nlevels):
				for sb in range(<int> chans[l]):
					for s in range(1 + 2 * importance):
						for i in range(qecc[l][0].nlogs):
							free(<void *>channels[b][l][sb][s][i])
						free(<void *>channels[b][l][sb][s])
					free(<void *>channels[b][l][sb])
				free(<void *>channels[b][l])
			free(<void *>channels[b])
	free(<void *>chans)
	return nchans


cdef int MemManageInputChannels(long double ****inputchannels, int nqecc, int nlogs, int importance, int tofree) nogil:
	# Allocate and free memory for the input channels structure in ComputeLogicalChannels(...)
	cdef:
		int i, j, q, s
	if (tofree == 0):
		# Initialize the space required for the input channels
		for q in range(nqecc):
			inputchannels[q] = <long double ***>malloc(sizeof(long double **) * (1 + <int>(importance == 2)))
			for s in range(1 + <int>(importance == 2)):
				inputchannels[q][s] = <long double **>malloc(sizeof(long double *) * (nlogs + 1))
				for i in range(nlogs):
					inputchannels[q][s][i] = <long double *>malloc(sizeof(long double) * nlogs)
				inputchannels[q][s][nlogs] = <long double *>malloc(sizeof(long double) * 2)
	else:
		for q in range(nqecc):
			for i in range(1 + <int>(importance == 2)):
				for j in range(1 + nlogs):
					free(<void *>inputchannels[q][i][j])
				free(<void *>inputchannels[q][i])
			free(<void *>inputchannels[q])
		free(<void *>inputchannels)
	return 0