#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False
# import time
cimport cython
from libc.stdlib cimport malloc, free
from cython cimport parallel
from printfuns cimport PrintDoubleArray1D, PrintComplexArray2D, PrintDoubleArray2D, PrintIntArray2D, PrintIntArray1D
from constants cimport constants_t
from qecc cimport qecc_t, SingleShotErrorCorrection
from memory cimport simul_t, AllocSimParamsQECC, FreeSimParamsQECC, FreeSimParams, CountIndepLogicalChannels, MemManageInputChannels, MemManageChannels, CopySimulation, MergeSimulations
from sampling cimport PowerSearch, ConstructImportanceDistribution, ConstructCumulative, SampleCumulative
from checks cimport IsState, IsDiagonal
from logmetrics cimport ComputeMetrics
#, DiamondNorm, Entropy

cdef extern from "stdlib.h" nogil:
	void srand48(long seedval)

cdef extern from "time.h":
	long time(int)

cdef extern from "math.h" nogil:
	long double powl(long double base, long double expo)
	long double fabsl(long double num)
	long double sqrtl(long double num)
	long double log10(long double num)
	int floorl(long double num)

cdef extern from "complex.h" nogil:
	long double creall(long double complex cnum)
	long double cimagl(long double complex cnum)
	long double complex conjl(long double complex cnum)


cdef int ComputeLevelOneChannels(simul_t *sim, qecc_t *qcode, constants_t *consts):
	# Compute the effective channels and syndrome probabilities for all level-1 syndromes.
	# The pre-computation is to avoid re-computing level-1 syndromes for every new top-level syndrome.
	cdef:
		int s, q, i, j, isPauli = 0
	# Load the physical channels on to the simulation structure and perform qcode
	for q in range(qcode[0].N):
		for i in range(qcode[0].nlogs):
			for j in range(qcode[0].nlogs):
				sim[0].virtual[q][i][j] = sim[0].physical[i][j]
		if (isPauli > 0):
			isPauli = isPauli * IsDiagonal(sim[0].virtual[q], qcode[0].nlogs)
	# print("SingleShotErrorCorrection with physical channels.")
	SingleShotErrorCorrection(0, isPauli, sim[0].frames[0], qcode, sim, consts)
	# print("Going to update metrics for level-1 channels.")
	UpdateMetrics(0, 1.0, 1.0, 0, qcode, sim, consts)
	# print("Storing the level-1 channels for future use.")
	for s in range(qcode[0].nstabs):
		sim[0].levelOneSynds[s] = sim[0].syndprobs[s]
		for i in range(qcode[0].nlogs):
			for j in range(qcode[0].nlogs):
				sim[0].levelOneChannels[s][i][j] = sim[0].effprocess[s][i][j]

	ConstructCumulative(sim[0].levelOneSynds, sim[0].levelOneCumul, qcode[0].nstabs)
	# PrintDoubleArray1D(sim[0].levelOneCumul, "Level-1 cumulative distribution", qcode[0].nstabs)
	# Compute the importance distribution for level-1 if necessary.
	if (sim[0].importance == 1):
		expo = PowerSearch(sim[0].syndprobs, qcode[0].nstabs, sim[0].outlierprobs, NULL)
		ConstructImportanceDistribution(sim[0].syndprobs, sim[0].levelOneImpDist, qcode[0].nstabs, expo)
		ConstructCumulative(sim[0].levelOneImpDist, sim[0].levelOneImpCumul, qcode[0].nstabs)
	return 0


cdef int ComputeLogicalChannels(simul_t **sims, qecc_t **qcode, constants_t *consts, long double *****channels) nogil:
	# Compute a logical channel for the required concatenation level.
	# The logical channel at a concatenated level l depends on N channels from the previous concatenation level, and so on... until 7^l physical channels.
	# Here we will sample 7^(l-1) level-1 channels. Using blocks of 7 of them we will construct 7^(l-2) level-2 channels and so on... until 1 level-l channel.
	cdef:
		int i, j, level, b, q, s, randsynd = 0
		long double bias = 0.0, history = 0.0, expo = 0.0
		int *isPauli = <int *>malloc(sizeof(int) * 2)
		int *chans = <int *>malloc(sizeof(int) * sims[0].nlevels)
		long double *impdist = <long double*>malloc(sizeof(long double) * qcode[0].nstabs)
		long double *impcumul = <long double*>malloc(sizeof(long double) * qcode[0].nstabs)
		long double ****inputchannels = NULL
	CountIndepLogicalChannels(chans, qcode, sims[0].nlevels)
	# At every level, select a set of 7 channels, consider them as physical channels and perform qcode to output a logical channel.
	# Place this logical channel in the channels array, at the succeeding level.
	# To start with, we will only initialize the last level with samples of the level-1 channels.
	for level in range(1, sims[0].nlevels):
		# Allocate memory for the simulation parameters which depend on the error correcting code
		AllocSimParamsQECC(sims[0], qcode[level][0].N, qcode[level][0].nstabs, qcode[level][0].nlogs)
		# Allocate memory for inputchannels
		inputchannels = <long double ****>malloc(sizeof(long double ***) * qcode[level][0].N)
		MemManageInputChannels(inputchannels, qcode[level][0].N, qcode[level][0].nlogs, sims[0][0].importance, 0)
		for b in range(chans[level]):
			bias = 1.0
			history = 1.0
			for q in range(qcode[level][0].N):
				# inputchannels[q] = {channels[level][7*b], ..., channels[level][7*(b+1)]}
				for s in range(1 + <int>(sims[0].importance == 2)):
					for i in range(qcode[0].nlogs):
						for j in range(qcode[0].nlogs):
							inputchannels[q][s][i][j] = channels[level - 1][qcode[level][0].N * b + q][s][i][j]
					inputchannels[q][s][qcode[0].nlogs][0] = channels[level - 1][qcode[level][0].N * b + q][s][qcode[level][0].nlogs][0]
					inputchannels[q][s][qcode[0].nlogs][1] = channels[level - 1][qcode[level][0].N * b + q][s][qcode[level][0].nlogs][1]
				bias = bias * inputchannels[q][0][qcode[level][0].nlogs][0]
				history = history * inputchannels[q][0][qcode[level][0].nlogs][1]
			# Load the input channels on to the simulation structures and perform qcode.
			for s in range(1 + <int>(sims[0].importance == 2)):
				isPauli[s] = 1
				for q in range(qcode[level][0].N):	
					for i in range(qcode[level][0].nlogs):
						for j in range(qcode[level][0].nlogs):
							sims[s][0].virtual[q][i][j] = inputchannels[q][s][i][j]
					if (isPauli[s] > 0):
						isPauli[s] = isPauli[s] * IsDiagonal(sims[s][0].virtual[q], qcode[level][0].nlogs)
				SingleShotErrorCorrection(level, isPauli[s], sims[s].frames[level], qcode[level], sims[s], consts)
			UpdateMetrics(level, bias, history, 0, qcode[level], sims[0], consts)

			if (level < (sims[0].nlevels - 1)):
				if (sims[0][0].importance == 0):
					randsynd = SampleCumulative(sims[0][0].cumulative, qcode[level][0].nstabs)
					for i in range(qcode[0].nlogs):
						for j in range(qcode[0].nlogs):
							channels[level][b][0][i][j] = sims[0][0].effprocess[randsynd][i][j]
					channels[level][b][0][qcode[level][0].nlogs][0] = 1.0
					channels[level][b][0][qcode[level][0].nlogs][1] = history * sims[0][0].syndprobs[randsynd]
					channels[level][b][0][qcode[level][0].nlogs][2] = sims[0][0].syndprobs[randsynd]
				elif (sims[0][0].importance == 1):
					# Compute a probability distribution where the probability of every syndrome is given by a power of the original syndrome distribution.
					# The new distribution Q(s) is given by Eq. 6 of .
					# Sample a syndrome according to Q(s) and add a bias P(s)/Q(s).
					expo = PowerSearch(sims[0][0].syndprobs, qcode[level][0].nstabs, sims[0][0].outlierprobs, NULL)
					ConstructImportanceDistribution(sims[0][0].syndprobs, impdist, qcode[level][0].nstabs, expo)
					ConstructCumulative(impdist, impcumul, qcode[level][0].nstabs)
					randsynd = SampleCumulative(impcumul, qcode[level][0].nstabs)
					for i in range(qcode[0].nlogs):
						for j in range(qcode[0].nlogs):
							channels[level][b][1][i][j] = sims[0][0].effprocess[randsynd][i][j]
					channels[level][b][1][qcode[level][0].nlogs][0] = sims[0][0].syndprobs[randsynd]/impdist[randsynd]
					channels[level][b][1][qcode[level][0].nlogs][1] = history * sims[0][0].syndprobs[randsynd]
					channels[level][b][1][qcode[level][0].nlogs][2] = sims[0][0].syndprobs[randsynd]
				elif (sims[0][0].importance == 2):
					# Draw two logical channels.
					# 1. Noisy channel simulation itself.
					randsynd = SampleCumulative(sims[1][0].cumulative, qcode[level][0].nstabs)
					for i in range(qcode[level][0].nlogs):
						for j in range(qcode[level][0].nlogs):
							channels[level][b][1][i][j] = sims[1][0].effprocess[randsynd][i][j]
					channels[level][b][1][qcode[level][0].nlogs][0] = 1.0
					channels[level][b][1][qcode[level][0].nlogs][1] = 1.0
					channels[level][b][1][qcode[level][0].nlogs][2] = sims[1][0].syndprobs[randsynd]
					# 2. Drawing syndromes for the original channel according to the noisy channel syndrome distribution.
					for i in range(qcode[level][0].nlogs):
						for j in range(qcode[level][0].nlogs):
							channels[level][b][0][i][j] = sims[0][0].effprocess[randsynd][i][j]
					channels[level][b][0][qcode[level][0].nlogs][0] = bias * sims[0][0].syndprobs[randsynd]/sims[1][0].syndprobs[randsynd]
					channels[level][b][0][qcode[level][0].nlogs][1] = history * sims[0][0].syndprobs[randsynd]
					channels[level][b][0][qcode[level][0].nlogs][2] = sims[0][0].syndprobs[randsynd]
				else:
					pass
		
		# PrintDoubleArray1D(sims[0][0].metricValues[level + 1], "metrics at level %d" % (level + 1), sims[0][0].nmetrics)
		# Free memory for inputchannels
		MemManageInputChannels(inputchannels, qcode[level][0].N, qcode[level][0].nlogs, sims[0][0].importance, 1)
		# Free simulation parameters that depend on the qcode
		FreeSimParamsQECC(sims[0], qcode[level][0].N, qcode[level][0].nstabs, qcode[level][0].nlogs)
		
	# Free memory
	free(<void *>isPauli)
	free(<void *>impdist)
	free(<void *>impcumul)
	free(<void *>chans)
	return 0


cdef int Performance(qecc_t **qcode, simul_t **sims, constants_t *consts):
	# Compute logical error rates for a concatenation level.
	# Allocate memory required for channels
	# Allocate the space required for all the channels
	# channels = list of l arrays.
		# channels[i] = list of list of 7^(l-i-1) arrays.
			# channels[i][b] = list of 3 arrays -- each corresponding to one type of sampling method. If importance sampling is turned off, there is only one array in the list.
				# channels[i][b][s] = 4x4 matrix denoting a level-i channel.
	# For every top-level syndrome sample:
		# Sample 7^(l-1) level-1 syndromes and store thier associated effective channels in the array.
		# Store the level-l average logical channel conditioned on level-1 syndromes selected above.
	
	# PrintIntArray2D(qcode[0].projector, "action[%d]", qcode[0][0].nstabs, qcode[0][0].nstabs)

	srand48(time(0))
	cdef:
		int i, j, b, s, level, stat, batch, nbatches = sims[0][0].cores, nchans
		int *randsynds = <int *>malloc(sizeof(int) * 2)
		long double ******channels = <long double ******>malloc(sizeof(long double *****) * nbatches)
	# Allocate space for channels
	nchans = MemManageChannels(channels, nbatches, qcode, sims[0].nlevels, sims[0].importance, 0)
	# Compute level-0 metrics and level-1 effective channels and syndromes.
	# ComputeLevelZeroMetrics(sims[0], qcode[0][0].nlogs, consts)
	# for s in range(1 + sims[0].importance):
	# 	ComputeLevelOneChannels(sims[s], qcode[0], consts)
	
	# print("nbatches = %d" % (nbatches))

	cdef:
		int randsynd = 0
		unsigned int thread_id
		simul_t ***simbatch = <simul_t ***>malloc(sizeof(simul_t **) * nbatches)
	# Make copies of the simulation structure which can be run in parallel.
	for batch in range(nbatches):
		simbatch[batch] = <simul_t **>malloc(sizeof(simul_t *) * (1 + <int>(sims[0].importance == 2)))
		for s in range(1 + <int>(sims[0].importance == 2)):
			simbatch[batch][s] = <simul_t *>malloc(sizeof(simul_t))
			CopySimulation(simbatch[batch][s], sims[s], qcode[0])
			simbatch[batch][s][0].nstats = sims[s][0].nstats/nbatches
			# Compute level-1 effective channels and syndromes.
			ComputeLevelOneChannels(simbatch[batch][s], qcode[0], consts)
			FreeSimParamsQECC(simbatch[batch][s], qcode[0].N, qcode[0].nstabs, qcode[0].nlogs)

	# print("Finished level-1 computations with %d channels." % (nchans))

	if (sims[0][0].nlevels > 1):
		with nogil, parallel.parallel(num_threads=nbatches):
			for batch in parallel.prange(nbatches, schedule='static'):
				thread_id = parallel.threadid()
				for stat in range(simbatch[batch][0][0].nstats):
					# Fill the lowest level of the channels array with "nchans" samples of level-1 channels.
					# with gil:
						# print("Stat %d" % (stat))
					for b in range(nchans):
						if (sims[0][0].importance == 0):
							# Direct sampling
							randsynd = SampleCumulative(simbatch[batch][0][0].levelOneCumul, qcode[0][0].nstabs)
							for i in range(qcode[0][0].nlogs):
								for j in range(qcode[0][0].nlogs):
									channels[batch][0][b][0][i][j] = simbatch[batch][0][0].levelOneChannels[randsynd][i][j]
							channels[batch][0][b][0][qcode[0][0].nlogs][0] = 1.0
							channels[batch][0][b][0][qcode[0][0].nlogs][1] = simbatch[batch][0][0].levelOneSynds[randsynd]
							channels[batch][0][b][0][qcode[0][0].nlogs][2] = simbatch[batch][0][0].levelOneSynds[randsynd]
							# with gil:
							# 	PrintDoubleArray2D(channels[batch][0][b][0], "Level-1 Channel %d corresponding to syndrome %d, probability = %g." % (b, randsynd, simbatch[batch][0][0].levelOneSynds[randsynd]), qcode[0][0].nlogs, qcode[0][0].nlogs)
						elif (sims[0][0].importance == 1):
							# Draw a syndrome from the importance distribution specified by the power-law scaling.
							randsynd = SampleCumulative(simbatch[batch][0][0].levelOneImpCumul, qcode[0][0].nstabs)
							for i in range(qcode[0][0].nlogs):
								for j in range(qcode[0][0].nlogs):
									channels[batch][0][b][0][i][j] = simbatch[batch][0][0].levelOneChannels[randsynd][i][j]
							channels[batch][0][b][0][qcode[0].nlogs][0] = simbatch[batch][0][0].levelOneImpDist[randsynd]/simbatch[batch][0][0].levelOneSynds[randsynd]
							channels[batch][0][b][0][qcode[0].nlogs][1] = simbatch[batch][0][0].levelOneSynds[randsynd]
							channels[batch][0][b][0][qcode[0].nlogs][2] = simbatch[batch][0][0].levelOneSynds[randsynd]
						elif (sims[0][0].importance == 2):
							# Draw a syndrome from the nosier channel syndrome distribution.
							randsynd = SampleCumulative(simbatch[batch][1][0].levelOneCumul, qcode[0][0].nstabs)
							for i in range(qcode[0][0].nlogs):
								for j in range(qcode[0][0].nlogs):
									for s in range(2):
										channels[batch][0][b][s][i][j] = simbatch[batch][s][0].levelOneChannels[randsynd][i][j]
							channels[batch][0][b][0][qcode[0][0].nlogs][0] = simbatch[batch][0][0].levelOneImpDist[randsynd]/simbatch[batch][1][0].levelOneSynds[randsynd]
							channels[batch][0][b][0][qcode[0][0].nlogs][1] = simbatch[batch][0][0].levelOneSynds[randsynd]
							channels[batch][0][b][0][qcode[0][0].nlogs][2] = simbatch[batch][0][0].levelOneSynds[randsynd]

							channels[batch][0][b][1][qcode[0][0].nlogs][0] = 1.0
							channels[batch][0][b][1][qcode[0][0].nlogs][1] = 1.0
							channels[batch][0][b][1][qcode[0][0].nlogs][2] = simbatch[batch][1][0].levelOneSynds[randsynd]
						else:
							pass
					# with gil:
						# print("Loaded the level-1 channels in the tree.")
					# Compute average logical channels and average logical error rates.
					ComputeLogicalChannels(simbatch[batch], qcode, consts, channels[batch])
					# with gil:
						# print("Computed logical channels")

	# print("Merging all simulation data")

	# Merge the simulation results into one.
	for batch in range(nbatches):
		for s in range(1 + <int>(sims[0].importance == 2)):
			MergeSimulations(sims[s], simbatch[batch][s], qcode[0][0].nlogs)

	# print("Done merging")

	# Free the space allocated for the simulation batches.
	for batch in range(nbatches):
		for s in range(1 + <int>(sims[0].importance == 2)):
			FreeSimParams(simbatch[batch][s], qcode[0][0].nstabs, qcode[0][0].nlogs)
			free(<void *>simbatch[batch][s])
		free(<void *>simbatch[batch])
	free(<void *>simbatch)

	# Normalize the average metrics.
	for level in range(1, sims[0][0].nlevels):
		UpdateMetrics(level, 1.0, 1.0, 1, qcode[level], sims[0], consts)

	# Free memory for channels
	MemManageChannels(channels, nbatches, qcode, sims[0].nlevels, sims[0].importance, 1)
	free(<void *>channels)

	free(<void *>randsynds)
	return 0


cdef int UpdateMetrics(int level, long double bias, long double history, int isfinal, qecc_t *qcode, simul_t *sim, constants_t *consts) nogil:
	# Compute metrics for all the effective channels and update the average value of the metrics
	cdef:
		int r, c, m, s, mbin = 0
		## Metric values
		long double *metvals = <long double *>malloc(sim[0].nmetrics * sizeof(long double))
		long double *avgmetvals = <long double *>malloc(sim[0].nmetrics * sizeof(long double))
	for m in range(sim[0].nmetrics):
		metvals[m] = 0.0
		avgmetvals[m] = 0.0
	
	if (isfinal == 0):
		for s in range(qcode[0].nstabs):
			if (sim[0].syndprobs[s] > consts[0].atol):
				# Compute metrics
				ComputeMetrics(metvals, sim[0].nmetrics, sim[0].metricsToCompute, sim[0].effective[s], sim[0].chname)
				# with gil:
					# PrintDoubleArray2D(sim[0].effprocess[s], "s = %d" % (s), qcode[0].nlogs, qcode[0].nlogs)
					# PrintDoubleArray1D(metvals, "metrics", sim[0].nmetrics)
				for m in range(sim[0].nmetrics):
					avgmetvals[m] = avgmetvals[m] + metvals[m] * sim[0].syndprobs[s]
				# Compute average channel
				for r in range(qcode[0].nlogs):
					for c in range(qcode[0].nlogs):
						sim[0].logical[level + 1][r][c] = sim[0].logical[level + 1][r][c] + bias * sim[0].effprocess[s][r][c] * sim[0].syndprobs[s]
			else:
				sim[0].syndprobs[s] = 0.0

			# with gil:
			# 	PrintDoubleArray2D(sim[0].effprocess[s], "Effective process matrix for level %d, P(%s) = %g." % (level, s, sim[0].syndprobs[s]), qcode[0].nlogs, qcode[0].nlogs)
		
		## Syndrome-metric binning
		for m in range(sim[0].nmetrics):
			sim[0].metricValues[level + 1][m] = sim[0].metricValues[level + 1][m] + bias * avgmetvals[m]
		
		sbin = GetBinPosition(fabsl(history), sim[0].nbins, sim[0].maxbin)
		for m in range(sim[0].nmetrics):
			mbin = GetBinPosition(fabsl(avgmetvals[m]), sim[0].nbins, sim[0].maxbin)
			sim[0].bins[level + 1][m][sbin][mbin] = sim[0].bins[level + 1][m][sbin][mbin] + 1
		# Update the number of statistics done for the level and the corresponding sum of biases.
		sim[0].statsperlevel[level + 1] = sim[0].statsperlevel[level + 1] + 1
	
	else:
		# After all the simulations are done, the average metrics are to be computed by diving the metricValues by the total number of statistics done for that level.
		for m in range(sim[0].nmetrics):
			sim[0].metricValues[level + 1][m] = sim[0].metricValues[level + 1][m]/(<long double>sim[0].statsperlevel[level + 1])
			sim[0].variance[level + 1][m] = 1/(<long double>(sim[0].statsperlevel[level + 1] * (sim[0].statsperlevel[level + 1] - 1))) * sim[0].sumsq[level + 1][m] - (sim[0].statsperlevel[level + 1])/(<long double>(sim[0].statsperlevel[level + 1] - 1)) * sim[0].metricValues[level + 1][m] * sim[0].metricValues[level + 1][m]
		for r in range(qcode[0].nlogs):
			for c in range(qcode[0].nlogs):
				sim[0].logical[level + 1][r][c] = sim[0].logical[level + 1][r][c]/(<long double>sim[0].statsperlevel[level + 1])
				sim[0].variance[level + 1][sim[0].nmetrics + r * qcode[0].nlogs + c] = 1/(<long double>(sim[0].statsperlevel[level + 1] * (sim[0].statsperlevel[level + 1] - 1))) * sim[0].sumsq[level + 1][sim[0].nmetrics + r * qcode[0].nlogs + c] - (sim[0].statsperlevel[level + 1])/(<long double>(sim[0].statsperlevel[level + 1] - 1)) * sim[0].logical[level + 1][r][c] * sim[0].logical[level + 1][r][c]
		# with gil:
		# 	PrintDoubleArray1D(sim[0].metricValues[level + 1], "Level %d Metric values" % (level + 1), sim[0].nmetrics)
	## Free memory
	free(<void *>metvals)
	free(<void *>avgmetvals)
	return 0


cdef int GetBinPosition(long double number, int nbins, int maxbin) nogil:
	# place a number into one of the bins depending on the order of magnitude of the number.
	# if a number is of the order of magnitude of 10^-i, then return i.
	# to find the order of magnitude we will take the negative log and bin it from 0 to 20.
	cdef:
		int binindex = <int> ((-1) * log10(number)/(<float> maxbin) * (<float> nbins))
	if (binindex < 0):
		binindex = 0
	if (binindex > (nbins - 1)):
		binindex = (nbins - 1)
	return binindex
