#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False
# import time
cimport cython
from libc.stdio cimport printf
from libc.stdlib cimport malloc, realloc, free
from cython cimport parallel
from printfuns cimport PrintDoubleArray1D, PrintComplexArray2D, PrintDoubleArray2D, PrintIntArray2D, PrintIntArray1D
from constants cimport constants_t
from qecc cimport qecc_t, SingleShotErrorCorrection
from memory cimport simul_t, AllocSimParamsQECC, FreeSimParamsQECC, FreeSimParams, CountIndepLogicalChannels, MemManageInputChannels, MemManageChannels, CopySimulation, MergeSimulations
from sampling cimport PowerSearch, PowerBound, ConstructImportanceDistribution, ConstructCumulative, SampleCumulative
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

# cdef int ComputeLevelOneChannels(simul_t *sim, qecc_t *qcode, constants_t *consts, int copies):
cdef int ComputeLevelOneChannels(simul_t *sim, qecc_t *qcode, constants_t *consts):
	# Compute the effective channels and syndrome probabilities for all level-1 syndromes.
	# The pre-computation is to avoid re-computing level-1 syndromes for every new top-level syndrome.
	cdef:
		int s, q, i, j, isPauli = 0
		long double *searchin = <long double *> malloc(sizeof(long double) * 2)
	# Load the physical channels on to the simulation structure and perform qcode
	for q in range(qcode[0].N):
		for i in range(qcode[0].nlogs):
			for j in range(qcode[0].nlogs):
				sim[0].virtual[q][i][j] = sim[0].physical[i][j]
		if (isPauli > 0):
			isPauli = isPauli * IsDiagonal(sim[0].virtual[q], qcode[0].nlogs)
	# printf("SingleShotErrorCorrection with physical channels.\n")
	SingleShotErrorCorrection(0, isPauli, sim[0].frames[0], qcode, sim, consts)
	# printf("Update metrics with level-1 channels.\n")
	UpdateMetrics(0, 1.0, 1.0, 0, qcode, sim, consts)
	# printf("Storing the level-1 channels for future use.\n")
	for s in range(qcode[0].nstabs):
		sim[0].levelOneSynds[s] = sim[0].syndprobs[s]
		for i in range(qcode[0].nlogs):
			for j in range(qcode[0].nlogs):
				sim[0].levelOneChannels[s][i][j] = sim[0].effprocess[s][i][j]
	# printf("Constructing cumulative distibution\n")
	ConstructCumulative(sim[0].levelOneSynds, sim[0].levelOneCumul, qcode[0].nstabs)
	# PrintDoubleArray1D(sim[0].levelOneCumul, "Level-1 cumulative distribution", qcode[0].nstabs)
	# Compute the importance distribution for level-1 if necessary.
	if (sim[0].importance == 1):
		searchin[0] = 0
		searchin[1] = 1
		# expo = PowerBound(sim[0].syndprobs, qcode[0].nstabs, sim[0].nstats * copies)
		expo = PowerSearch(sim[0].syndprobs, qcode[0].nstabs, sim[0].outlierprobs, searchin)
		ConstructImportanceDistribution(sim[0].syndprobs, sim[0].levelOneImpDist, qcode[0].nstabs, expo)
		ConstructCumulative(sim[0].levelOneImpDist, sim[0].levelOneImpCumul, qcode[0].nstabs)
		# PrintDoubleArray1D(sim[0].syndprobs, "true", qcode[0].nstabs)
		# PrintDoubleArray1D(sim[0].levelOneImpDist, "importance", qcode[0].nstabs)
	# free memory
	free(<void *>searchin)
	# printf("Done ComputeLevelOneChannels.\n")
	return 0

###################
# C translation of the function: ComputeLevelOneChannels

void ComputeLevelOneChannels(simul_t *sim, qecc_t *qcode, constants_t *consts){
	// Compute the effective channels and syndrome probabilities for all level-1 syndromes.
	// The pre-computation is to avoid re-computing level-1 syndromes for every new top-level syndrome.
	int s, q, i, j, isPauli = 0;
	long double *searchin = <long double *> malloc(sizeof(long double) * 2);
	// Load the physical channels on to the simulation structure and perform qcode
	for (q = 0; q < qcode->N; q ++){
		for (i = 0; i < qcode->nlogs; i ++){
			for (j = 0; j < qcode->nlogs; j ++){
				(sim->virtual)[q][i][j] = (sim->physical)[i][j];
			}
		}
		if (isPauli > 0)
			isPauli = isPauli * IsDiagonal(sim->virtual[q], qcode->nlogs);
	}
	// printf("SingleShotErrorCorrection with physical channels.\n");
	SingleShotErrorCorrection(0, isPauli, sim->frames[0], qcode, sim, consts);
	// printf("Update metrics with level-1 channels.\n");
	UpdateMetrics(0, 1.0, 1.0, 0, qcode, sim, consts);
	// printf("Storing the level-1 channels for future use.\n");
	for (s = 0; s < qcode->nstabs; s ++){
		sim->levelOneSynds[s] = sim->syndprobs[s];
		for (i = 0; i < qcode->nlogs; i ++){
			for (j = 0; j < qcode->nlogs; j ++){
				(sim->levelOneChannels)[s][i][j] = (sim->effprocess)[s][i][j];
			}
		}
	}
	// printf("Constructing cumulative distibution\n")
	ConstructCumulative(sim->levelOneSynds, sim->levelOneCumul, qcode->nstabs);
	// PrintDoubleArray1D(sim->levelOneCumul, "Level-1 cumulative distribution", qcode->nstabs);
	// Compute the importance distribution for level-1 if necessary.
	if (sim->importance == 1){
		searchin[0] = 0;
		searchin[1] = 1;
		expo = PowerSearch(sim->syndprobs, qcode->nstabs, sim->outlierprobs, searchin);
		ConstructImportanceDistribution(sim->syndprobs, sim->levelOneImpDist, qcode->nstabs, expo);
		ConstructCumulative(sim->levelOneImpDist, sim->levelOneImpCumul, qcode->nstabs);
		// PrintDoubleArray1D(sim->syndprobs, "true", qcode->nstabs);
		// PrintDoubleArray1D(sim->levelOneImpDist, "importance", qcode->nstabs);
	}
	// free memory
	free(searchin);
	// printf("Done ComputeLevelOneChannels.\n");
}

###################


cdef int ComputeLogicalChannels(simul_t **sims, qecc_t **qcode, constants_t *consts, long double *****channels) nogil:
	# Compute a logical channel for the required concatenation level.
	# The logical channel at a concatenated level l depends on N channels from the previous concatenation level, and so on... until 7^l physical channels.
	# Here we will sample 7^(l-1) level-1 channels. Using blocks of 7 of them we will construct 7^(l-2) level-2 channels and so on... until 1 level-l channel.
	printf("Function: ComputeLogicalChannels\n")
	cdef:
		int i, j, level, b, q, s, randsynd = 0
		long double bias = 0.0, history = 0.0, expo = 0.0
		int *isPauli = <int *>malloc(sizeof(int) * 2)
		int *chans = <int *>malloc(sizeof(int) * sims[0][0].nlevels)
		long double *searchin = <long double *> malloc(sizeof(long double) * 2)
		long double *impdist = <long double*>malloc(sizeof(long double) * qcode[0][0].nstabs)
		long double *impcumul = <long double*>malloc(sizeof(long double) * qcode[0][0].nstabs)
		long double ****inputchannels = <long double ****>malloc(sizeof(long double ***))
	CountIndepLogicalChannels(chans, qcode, sims[0][0].nlevels)
	# At every level, select a set of 7 channels, consider them as physical channels and perform qcode to output a logical channel.
	# Place this logical channel in the channels array, at the succeeding level.
	# To start with, we will only initialize the last level with samples of the level-1 channels.
	
	# PrintIntArray1D(sims[0][0].runstats, "runstats", sims[0][0].runstats[0] + 1)

	for level in range(1, sims[0][0].nlevels):
		# printf("level %d\n", level)
		# Allocate memory for the simulation parameters which depend on the error correcting code
		for s in range(1 + <int>(sims[0][0].importance == 2)):
			AllocSimParamsQECC(sims[s], qcode[level][0].N, qcode[level][0].nstabs, qcode[level][0].nlogs)
		# Allocate memory for inputchannels
		inputchannels = <long double ****>realloc(<void *>inputchannels, sizeof(long double ***) * qcode[level][0].N)
		MemManageInputChannels(inputchannels, qcode[level][0].N, qcode[level][0].nlogs, sims[0][0].importance, 0)

		for b in range(chans[level]):
			bias = 1.0
			history = 1.0
			for q in range(qcode[level][0].N):
				# inputchannels[q] = {channels[level][7*b], ..., channels[level][7*(b+1)]}
				for s in range(1 + <int>(sims[0][0].importance == 2)):
					for i in range(qcode[0].nlogs):
						for j in range(qcode[0].nlogs):
							inputchannels[q][s][i][j] = channels[level - 1][qcode[level][0].N * b + q][s][i][j]
					inputchannels[q][s][qcode[0].nlogs][0] = channels[level - 1][qcode[level][0].N * b + q][s][qcode[level][0].nlogs][0]
					inputchannels[q][s][qcode[0].nlogs][1] = channels[level - 1][qcode[level][0].N * b + q][s][qcode[level][0].nlogs][1]
				bias = bias * inputchannels[q][0][qcode[level][0].nlogs][0]
				history = history * inputchannels[q][0][qcode[level][0].nlogs][1]

			# Load the input channels on to the simulation structures and perform qcode.
			for s in range(1 + <int>(sims[0][0].importance == 2)):
				isPauli[s] = 1
				for q in range(qcode[level][0].N):	
					for i in range(qcode[level][0].nlogs):
						for j in range(qcode[level][0].nlogs):
							sims[s][0].virtual[q][i][j] = inputchannels[q][s][i][j]
					if (isPauli[s] > 0):
						isPauli[s] = isPauli[s] * IsDiagonal(sims[s][0].virtual[q], qcode[level][0].nlogs)
				SingleShotErrorCorrection(level, isPauli[s], sims[s].frames[level], qcode[level], sims[s], consts)
			
			UpdateMetrics(level, bias, history, 0, qcode[level], sims[0], consts)

			if (level < (sims[0][0].nlevels - 1)):
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
					# The new distribution Q(s) is given by Eq. 6 of the article.
					# Sample a syndrome according to Q(s) and add a bias P(s)/Q(s).
					searchin[0] = 0
					searchin[1] = 1
					# expo = PowerBound(sims[0][0].syndprobs, qcode[level][0].nstabs, chans[level + 1] * sims[0][0].nstats)
					expo = PowerSearch(sims[0][0].syndprobs, qcode[level][0].nstabs, sims[0][0].outlierprobs, searchin)
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
		# printf("level = %d\n", level)
		# PrintIntArray1D(sims[0][0].runstats, "runstats", sims[0][0].runstats[0] + 1)
		
		# PrintDoubleArray1D(sims[0][0].metricValues[level + 1], "metrics at level %d" % (level + 1), sims[0][0].nmetrics)
		# Free memory for inputchannels
		MemManageInputChannels(inputchannels, qcode[level][0].N, qcode[level][0].nlogs, sims[0][0].importance, 1)
		# Free simulation parameters that depend on the qcode
		# printf("Calling FreeSimParamsQECC\n")
		for s in range(1 + <int>(sims[0][0].importance == 2)):
			FreeSimParamsQECC(sims[s], qcode[level][0].N, qcode[level][0].nstabs, qcode[level][0].nlogs)
	
	# PrintIntArray1D(sims[0][0].runstats, "runstats", sims[0][0].runstats[0] + 1)

	# Free memory
	free(<void *>searchin)
	free(<void *>isPauli)
	free(<void *>impdist)
	free(<void *>impcumul)
	# printf("Freeing chans\n")
	free(<void *>chans)
	# printf("Freeing inputchannels\n")
	for q in range(qcode[sims[0][0].nlevels - 1][0].N):
		free(<void *>inputchannels[q])
	free(<void *>inputchannels)
	printf("done ComputeLogicalChannels.\n")
	return 0

######################
# Converted to C

void ComputeLogicalChannels(simul_t **sims, qecc_t **qcode, constants_t *consts, long double *****channels){
	// Compute a logical channel for the required concatenation level.
	// The logical channel at a concatenated level l depends on N channels from the previous concatenation level, and so on... until 7^l physical channels.
	// Here we will sample 7^(l-1) level-1 channels. Using blocks of 7 of them we will construct 7^(l-2) level-2 channels and so on... until 1 level-l channel.
	printf("Function: ComputeLogicalChannels\n");
	int i, j, level, b, q, s, randsynd = 0;
	long double bias = 0.0, history = 0.0, expo = 0.0;
	int *isPauli = malloc(sizeof(int) * 2);
	int *chans = malloc(sizeof(int) * sims[0]->nlevels);
	long double *searchin = malloc(sizeof(long double) * 2);
	long double *impdist = malloc(sizeof(long double) * qcode[0]->nstabs);
	long double *impcumul = malloc(sizeof(long double) * qcode[0]->nstabs);
	long double ****inputchannels = malloc(sizeof(long double ***));

	CountIndepLogicalChannels(chans, qcode, sims[0]->nlevels);

	// At every level, select a set of n channels, consider them as physical channels and perform qcode to output a logical channel.
	// Place this logical channel in the channels array, at the succeeding level.
	// To start with, we will only initialize the last level with samples of the level-1 channels.
	// PrintIntArray1D(sims[0]->runstats, "runstats", sims[0]->runstats[0] + 1);
	
	for (level = 1; level < sims[0]->nlevels; level ++){
		// printf("level %d\n", level);
		// Allocate memory for the simulation parameters which depend on the error correcting code
		for (s = 0; s < (int)(sims[0]->importance == 2); s ++){
			AllocSimParamsQECC(sims[s], qcode[level]->N, qcode[level]->nstabs, qcode[level]->nlogs);
		}
		// Allocate memory for inputchannels
		inputchannels = (long double ****)realloc(inputchannels, sizeof(long double ***) * qcode[level]->N);
		MemManageInputChannels(inputchannels, qcode[level]->N, qcode[level]->nlogs, sims[0]->importance, 0);

		for (b = 0; b < chans[level]; b ++){
			bias = 1.0;
			history = 1.0;
			for (q = 0; q < qcode[level]->N; q ++){
				// inputchannels[q] = {channels[level][7*b], ..., channels[level][7*(b+1)]}
				for (s = 0; s < (int)(1 + (int)(sims[0]->importance == 2)); s ++){
					for (i = 0; i < qcode->nlogs; i ++){
						for (j = 0; j < qcode->nlogs; j ++){
							inputchannels[q][s][i][j] = channels[level - 1][qcode[level]->N * b + q][s][i][j];
						}
					}
					inputchannels[q][s][qcode->nlogs][0] = channels[level - 1][qcode[level]->N * b + q][s][qcode[level]->nlogs][0];
					inputchannels[q][s][qcode->nlogs][1] = channels[level - 1][qcode[level]->N * b + q][s][qcode[level]->nlogs][1];
				}
				bias = bias * inputchannels[q][0][qcode[level]->nlogs][0];
				history = history * inputchannels[q][0][qcode[level]->nlogs][1];
			}

			// Load the input channels on to the simulation structures and perform qcode.
			for (s = 0; s < (int)(1 + (int)(sims[0]->importance == 2)); s ++){
				isPauli[s] = 1;
				for (q = 0; q < qcode[level]->N; q ++){
					for (i = 0; i < qcode->nlogs; i ++){
						for (j = 0; j < qcode->nlogs; j ++){
							(sims[s]->virtual)[q][i][j] = inputchannels[q][s][i][j];
						}
					}
					if (isPauli[s] > 0)
						isPauli[s] = isPauli[s] * IsDiagonal((sims[s]->virtual)[q], qcode[level]->nlogs);
				}
				SingleShotErrorCorrection(level, isPauli[s], (sims[s]->frames)[level], qcode[level], sims[s], consts);
			}
			UpdateMetrics(level, bias, history, 0, qcode[level], sims[0], consts);

			if (level < (sims[0]->nlevels - 1)){
				if (sims[0]->importance == 0){
					randsynd = SampleCumulative(sims[0]->cumulative, qcode[level]->nstabs);
					for (i = 0; i < qcode[level]->nlogs; i ++){
						for (j = 0; j < qcode[level]->nlogs; j ++){
							channels[level][b][0][i][j] = (sims[0]->effprocess)[randsynd][i][j];
						}
					}
					channels[level][b][0][qcode[level][0].nlogs][0] = 1.0;
					channels[level][b][0][qcode[level][0].nlogs][1] = history * sims[0]->syndprobs[randsynd];
					channels[level][b][0][qcode[level][0].nlogs][2] = sims[0]->syndprobs[randsynd];
				}
				else if (sims[0]->importance == 1){
					// Compute a probability distribution where the probability of every syndrome is given by a power of the original syndrome distribution.
					// The new distribution Q(s) is given by Eq. 6 of the article.
					// Sample a syndrome according to Q(s) and add a bias P(s)/Q(s).
					searchin[0] = 0;
					searchin[1] = 1;
					expo = PowerSearch(sims[0]->syndprobs, qcode[level]->nstabs, sims[0]->outlierprobs, searchin);
					ConstructImportanceDistribution(sims[0]->syndprobs, impdist, qcode[level]->nstabs, expo);
					ConstructCumulative(impdist, impcumul, qcode[level]->nstabs);
					randsynd = SampleCumulative(impcumul, qcode[level]->nstabs);
					for (i = 0; i < qcode[level]->nlogs; i ++){
						for (j = 0; j < qcode[level]->nlogs; j ++){
							channels[level][b][1][i][j] = sims[0]->effprocess[randsynd][i][j];
						}
					}
					channels[level][b][1][qcode[level]->nlogs][0] = (sims[0]->syndprobs)[randsynd]/impdist[randsynd];
					channels[level][b][1][qcode[level]->nlogs][1] = history * (sims[0]->syndprobs)[randsynd];
					channels[level][b][1][qcode[level]->nlogs][2] = (sims[0]->syndprobs)[randsynd];
				}
				else if (sims[0]->importance == 2){
					// Draw two logical channels.
					// 1. Noisy channel simulation itself.
					randsynd = SampleCumulative(sims[1]->cumulative, qcode[level]->nstabs);
					for (i = 0; i < qcode[level]->nlogs; i ++){
						for (j = 0; j < qcode[level]->nlogs; j ++){
							channels[level][b][1][i][j] = sims[1]->effprocess[randsynd][i][j];
						}
					}
					channels[level][b][1][qcode[level]->nlogs][0] = 1.0;
					channels[level][b][1][qcode[level]->nlogs][1] = 1.0;
					channels[level][b][1][qcode[level]->nlogs][2] = (sims[1]->syndprobs)[randsynd];
					// 2. Drawing syndromes for the original channel according to the noisy channel syndrome distribution.
					for (i = 0; i < qcode[level]->nlogs; i ++){
						for (j = 0; j < qcode[level]->nlogs; j ++){
							channels[level][b][0][i][j] = (sims[0]->effprocess)[randsynd][i][j];
						}
					}
					channels[level][b][0][qcode[level]->nlogs][0] = bias * sims[0]->syndprobs[randsynd]/((sims[1]->syndprobs)[randsynd]);
					channels[level][b][0][qcode[level]->nlogs][1] = history * ((sims[0]->syndprobs)[randsynd]);
					channels[level][b][0][qcode[level]->nlogs][2] = ((sims[0]->syndprobs)[randsynd]);
				}
				else
					continue;
			}
		}
		// printf("level = %d\n", level);
		PrintIntArray1D(sims[0]->runstats, "runstats", sims[0]->runstats[0] + 1);
		// PrintDoubleArray1D(sims[0]->metricValues[level + 1], "metrics at level %d" % (level + 1), sims[0]->nmetrics);
		// Free memory for inputchannels
		MemManageInputChannels(inputchannels, qcode[level]->N, qcode[level]->nlogs, sims[0]->importance, 1);
		// Free simulation parameters that depend on the qcode
		// printf("Calling FreeSimParamsQECC\n")
		for (s = 0; s < (int) (1 + (int)(sims[0]->importance == 2)); s ++){
			FreeSimParamsQECC(sims[s], qcode[level]->N, qcode[level]->nstabs, qcode[level]->nlogs);
		}
	}
	// PrintIntArray1D(sims[0]->runstats, "runstats", sims[0]->runstats[0] + 1);
	// Free memory
	free(searchin);
	free(isPauli);
	free(impdist);
	free(impcumul);
	// printf("Freeing chans\n");
	free(chans);
	// printf("Freeing inputchannels\n")
	for (q = 0; q < qcode[sims[0]->nlevels - 1]->N; q ++){
		free(inputchannels[q]);
	}
	free(inputchannels);
	printf("done ComputeLogicalChannels.\n");
}

######################


cdef int PickFromInterval(int low, int high, int *arr, int *found, int alloc) nogil:
	# Find the elements in the given (sorted) array that are in a specified range.
	# The found elements must be recorded in the "found" array.
	# The first elements of the sorted and found arrays indicate their respective sizes.
	# printf("Find elements in the array\n")
	# PrintIntArray1D(arr, "array", arr[0] + 1)
	# printf("that are between %d and %d.\n", low, high)
	cdef:
		int i = 0, nelems = 0
	for i in range(arr[0]):
		if ((arr[i + 1] >= low) and (arr[i + 1] <= high)):
			nelems = nelems + 1
			if (alloc == 1):
				found[nelems] = arr[i + 1]
	if (alloc == 1):
		found[0] = nelems
		# PrintIntArray1D(found, "found", nelems + 1)
	return nelems

####################
# Converted to C
int PickFromInterval(int low, int high, int *arr, int *found, int alloc){
	// Find the elements in the given (sorted) array that are in a specified range.
	// The found elements must be recorded in the "found" array.
	// The first elements of the sorted and found arrays indicate their respective sizes.
	// printf("Find elements in the array\n");
	// PrintIntArray1D(arr, "array", arr[0] + 1);
	// printf("that are between %d and %d.\n", low, high);
	int i = 0, nelems = 0;
	for (i = 0; i < arr[0]; i ++){
		if ((arr[i + 1] >= low) && (arr[i + 1] <= high)){
			nelems = nelems + 1;
			if (alloc == 1){
				found[nelems] = arr[i + 1];
			}
		}
	}
	if (alloc == 1){
		found[0] = nelems;
		// PrintIntArray1D(found, "found", nelems + 1);
	}
	return nelems;
}
####################


cdef int IsElement(int *arr, int item) nogil:
	# Determine if an item is present in an array
	# printf("Testing if %d is present in the following array of size %d.\n", item, arr[0])
	# PrintIntArray1D(arr, "array", arr[0] + 1)
	cdef:
		int i
	for i in range(arr[0]):
		if (arr[i + 1] == item):
			return 1
	return 0

###################
# Converted to C
int IsElement(int *arr, int item){
	// Determine if an item is present in an array
	// printf("Testing if %d is present in the following array of size %d.\n", item, arr[0]);
	// PrintIntArray1D(arr, "array", arr[0] + 1);
	int i;
	for (i = 0; i < arr[0]; i ++){
		if (arr[i + 1] == item)
			return 1;
	}
	return 0;
}
###################


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

	printf("Function: Performance: %ld\n", (time(0)))
	
	srand48(time(0))
	cdef:
		unsigned int thread_id
		int m, i, j, b, s, l, level, stat, batch, nbatches = sims[0][0].cores, nchans = 0, randsynd = 0, size = 0
		int *inter
		long double *****channels
	# Allocate space for channels
	nchans = MemManageChannels(channels, qcode, sims[0][0].nlevels, sims[0][0].importance, 0)
	# Compute level-0 metrics and level-1 effective channels and syndromes.
	# ComputeLevelZeroMetrics(sims[0], qcode[0][0].nlogs, consts)
	for s in range(1 + sims[0][0].importance):
		# Compute level-1 effective channels and syndromes.
		ComputeLevelOneChannels(sims[s], qcode[0], consts)
	
	print("nbatches = %d, nlevels = %d, nchans = %d" % (nbatches, sims[0][0].nlevels, nchans))

	printf("Finished level-1 computations with %d channels.\n", (nchans))

	if (sims[0][0].nlevels > 1):
		for stat in range(sims[0][0].nstats):
			# Fill the lowest level of the channels array with "nchans" samples of level-1 channels.
			# printf("Stat %d\n", stat)
			print("Stat %d, nchans = %d" % (stat, nchans))
			for b in range(nchans):
				if (sims[0][0].importance == 0):
					# Direct sampling
					randsynd = SampleCumulative(sims[0][0].levelOneCumul, qcode[0][0].nstabs)
					for i in range(qcode[0][0].nlogs):
						for j in range(qcode[0][0].nlogs):
							channels[batch][0][b][0][i][j] = sims[0][0].levelOneChannels[randsynd][i][j]
					channels[0][b][0][qcode[0][0].nlogs][0] = 1.0
					channels[0][b][0][qcode[0][0].nlogs][1] = sims[0][0].levelOneSynds[randsynd]
					channels[0][b][0][qcode[0][0].nlogs][2] = sims[0][0].levelOneSynds[randsynd]
					# with gil:
					# 	PrintDoubleArray2D(channels[0][b][0], "Level-1 Channel %d corresponding to syndrome %d, probability = %g." % (b, randsynd, sims[0][0].levelOneSynds[randsynd]), qcode[0][0].nlogs, qcode[0][0].nlogs)
				elif (sims[0][0].importance == 1):
					# Draw a syndrome from the importance distribution specified by the power-law scaling.
					randsynd = SampleCumulative(sims[0][0].levelOneImpCumul, qcode[0][0].nstabs)
					for i in range(qcode[0][0].nlogs):
						for j in range(qcode[0][0].nlogs):
							channels[0][b][0][i][j] = sims[0][0].levelOneChannels[randsynd][i][j]
					channels[0][b][0][qcode[0][0].nlogs][0] = sims[0][0].levelOneSynds[randsynd]/sims[0][0].levelOneImpDist[randsynd]
					channels[0][b][0][qcode[0][0].nlogs][1] = sims[0][0].levelOneSynds[randsynd]
					channels[0][b][0][qcode[0][0].nlogs][2] = sims[0][0].levelOneSynds[randsynd]
					# PrintDoubleArray1D(sims[0][0].levelOneSynds, "true", qcode[0][0].nstabs)
					# PrintDoubleArray1D(sims[0][0].levelOneImpDist, "importance", qcode[0][0].nstabs)
					# with gil:
						# print("Random syndrome: %d, bias = %Lg/%Lg = %Lg" % (randsynd, sims[0][0].levelOneImpDist[randsynd], sims[0][0].levelOneSynds[randsynd], channels[0][b][0][qcode[0][0].nlogs][0]))
				elif (sims[0][0].importance == 2):
					# Draw a syndrome from the nosier channel syndrome distribution.
					randsynd = SampleCumulative(sims[1][0].levelOneCumul, qcode[0][0].nstabs)
					for i in range(qcode[0][0].nlogs):
						for j in range(qcode[0][0].nlogs):
							for s in range(2):
								channels[0][b][s][i][j] = sims[s][0].levelOneChannels[randsynd][i][j]
					channels[0][b][0][qcode[0][0].nlogs][0] = sims[0][0].levelOneImpDist[randsynd]/sims[1][0].levelOneSynds[randsynd]
					channels[0][b][0][qcode[0][0].nlogs][1] = sims[0][0].levelOneSynds[randsynd]
					channels[0][b][0][qcode[0][0].nlogs][2] = sims[0][0].levelOneSynds[randsynd]

					channels[0][b][1][qcode[0][0].nlogs][0] = 1.0
					channels[0][b][1][qcode[0][0].nlogs][1] = 1.0
					channels[0][b][1][qcode[0][0].nlogs][2] = sims[1][0].levelOneSynds[randsynd]
				else:
					pass
			# printf("Loaded the level-1 channels in the tree.\n")
			# Compute average logical channels and average logical error rates.
			ComputeLogicalChannels(sims, qcode, consts, channels)
			# printf("Computed logical channels.\n")
			for s in range(1 + <int>(sims[0][0].importance == 2)):
				# printf("sims[s][0].runstats[0] = %d\n", sims[s][0].runstats[0])
				# PrintIntArray1D(sims[s][0].runstats, "runstats", sims[s][0].runstats[0] + 1)
				if (IsElement(sims[s][0].runstats, stat + 1) == 1):
					for m in range(sims[0][0].nmetrics):
						sims[s][0].runavg[0][m][0] = sims[s][0].runavg[0][m][0] + 1
						sims[s][0].runavg[0][m][<int>(sims[s][0].runavg[0][m][0])] = sims[s][0].metricValues[sims[0][0].nlevels][m]
						sims[s][0].runavg[1][m][0] = sims[s][0].runavg[1][m][0] + 1
		
	# PrintDoubleArray2D(sims[0][0].runavg[0], "running average", sims[0][0].nmetrics, sims[0][0].runstats[0])
	# PrintDoubleArray2D(sims[0][0].runavg[1], "running variance", sims[0][0].nmetrics, sims[0][0].runstats[0])
	# printf("Done merging\n")
	
	# Normalize the average metrics.
	for level in range(1, sims[0][0].nlevels):
		UpdateMetrics(level, 1.0, 1.0, 1, qcode[level], sims[0], consts)

	printf("Updated metrics\n")

	# Compute the variance from the sum of squares
	# PrintDoubleArray2D(sims[0][0].sumsq, "sumsq", sims[0][0].nlevels, sims[0][0].nmetrics + qcode[0][0].nlogs * qcode[0][0].nlogs)
	for s in range(1 + <int>(sims[0][0].importance == 2)):
		for l in range(1, sims[s][0].nlevels):
			for m in range(sims[0][0].nmetrics + qcode[l][0].nlogs):
				sims[s][0].variance[l + 1][m] = 1/(<long double>(sims[s][0].statsperlevel[l + 1] - 1) * sims[s][0].statsperlevel[l + 1]) * sims[s][0].sumsq[l + 1][m]
				if (m < sims[s][0].nmetrics):
					sims[s][0].variance[l + 1][m] = sims[s][0].variance[l + 1][m] - 1/(<long double>(sims[s][0].statsperlevel[l + 1] - 1)) * sims[s][0].metricValues[l + 1][m] * sims[s][0].metricValues[l + 1][m]
				else:
					sims[s][0].variance[l + 1][m] = sims[s][0].variance[l + 1][m] - 1/(<long double>(sims[s][0].statsperlevel[l + 1] - 1)) * sims[s][0].logical[l + 1][m/qcode[l][0].nlogs][m % qcode[l][0].nlogs] * sims[s][0].logical[l + 1][m/qcode[l][0].nlogs][m % qcode[l][0].nlogs]
				sims[s].variance[l + 1][m] = sqrtl(sims[s].variance[l + 1][m])
	
	printf("Computed variance\n")

	# Free memory for channels
	MemManageChannels(channels, nbatches, qcode, sims[0][0].nlevels, sims[0][0].importance, 1)
	free(<void *>channels)

	printf("Freed channels\n")

	printf("Done Performance.\n")

	return 0

########################
# Converted to C

void Performance(qecc_t **qcode, simul_t **sims, constants_t *consts){
	// Compute logical error rates for a concatenation level.
	// Allocate memory required for channels
	// Allocate the space required for all the channels
	// channels = list of l arrays.
		// channels[i] = list of list of 7^(l-i-1) arrays.
			// channels[i][b] = list of 3 arrays -- each corresponding to one type of sampling method. If importance sampling is turned off, there is only one array in the list.
				// channels[i][b][s] = 4x4 matrix denoting a level-i channel.
	// For every top-level syndrome sample:
		// Sample 7^(l-1) level-1 syndromes and store thier associated effective channels in the array.
		// Store the level-l average logical channel conditioned on level-1 syndromes selected above.
	// PrintIntArray2D(qcode[0]->projector, "action[%d]", qcode[0]->nstabs, qcode[0]->nstabs);
	// printf("Function: Performance: %ld\n", (time(0)))
	srand48(time(0));
	int m, i, j, b, s, l, level, stat, nchans = 0, randsynd = 0, size = 0;
	int *inter;
	long double *****channels;
	// Allocate space for channels
	nchans = MemManageChannels(channels, qcode, sims[0]->nlevels, sims[0]->importance, 0);
	printf("nlevels = %d, nchans = %d\n", sims[0]->nlevels, nchans);
	// Compute level-0 metrics and level-1 effective channels and syndromes.
	// ComputeLevelZeroMetrics(sims[0], qcode[0]->nlogs, consts)
	for (s = 0; s < (1 + sims[0]->importance); s ++){
		// Compute level-1 effective channels and syndromes.
		ComputeLevelOneChannels(sims[s], qcode[0], consts);
	}
	printf("Finished level-1 computations with %d channels.\n", (nchans))
	
	if (sims[0]->nlevels > 1){
		for (stat = 0; stat < sims[0]->nstats; stat ++){
			// Fill the lowest level of the channels array with "nchans" samples of level-1 channels.
			// print("Stat %d, nchans = %d" % (stat, nchans))
			for (b = 0; b < nchans; b ++){
				if (sims[0]->importance == 0){
					// Direct sampling
					randsynd = SampleCumulative(sims[0]->levelOneCumul, qcode[0]->nstabs);
					for (i = 0; i < qcode[0]->nlogs; i ++){
						for (j = 0; j < qcode[0]->nlogs; j ++){
							channels[batch][0][b][0][i][j] = (sims[0]->levelOneChannels)[randsynd][i][j];
						}
					}
					channels[0][b][0][qcode[0][0].nlogs][0] = 1.0;
					channels[0][b][0][qcode[0][0].nlogs][1] = sims[0]->levelOneSynds[randsynd];
					channels[0][b][0][qcode[0][0].nlogs][2] = sims[0]->levelOneSynds[randsynd];
					// PrintDoubleArray2D(channels[0][b][0], "Level-1 Channel %d corresponding to syndrome %d, probability = %g." % (b, randsynd, sims[0]->levelOneSynds[randsynd]), qcode[0]->nlogs, qcode[0]->nlogs);
				}
				else if (sims[0]->importance == 0){
					// Draw a syndrome from the importance distribution specified by the power-law scaling.
					randsynd = SampleCumulative(sims[0]->levelOneImpCumul, qcode[0]->nstabs);
					for (i = 0; i < qcode[0]->nlogs; i ++){
						for (j = 0; j < qcode[0]->nlogs; j ++){
							channels[0][b][0][i][j] = sims[0]->levelOneChannels[randsynd][i][j];
						}
					}
					channels[0][b][0][qcode[0]->nlogs][0] = sims[0]->levelOneSynds[randsynd]/sims[0]->levelOneImpDist[randsynd];
					channels[0][b][0][qcode[0]->nlogs][1] = sims[0]->levelOneSynds[randsynd];
					channels[0][b][0][qcode[0]->nlogs][2] = sims[0]->levelOneSynds[randsynd];
					PrintDoubleArray1D(sims[0]->levelOneSynds, "true", qcode[0]->nstabs);
					PrintDoubleArray1D(sims[0]->levelOneImpDist, "importance", qcode[0]->nstabs);
					printf("Random syndrome: %d, bias = %Lg/%Lg = %Lg", randsynd, sims[0]->levelOneImpDist[randsynd], sims[0]->levelOneSynds[randsynd], channels[0][b][0][qcode[0]->nlogs][0]);
				}
				else if (sims[0]->importance == 2){
					// Draw a syndrome from the nosier channel syndrome distribution.
					randsynd = SampleCumulative(sims[1]->levelOneCumul, qcode[0]->nstabs);
					for (i = 0; i < qcode[0]->nlogs; i ++){
						for (j = 0; j < qcode[0]->nlogs; j ++){
							for s in range(2){
								channels[0][b][s][i][j] = sims[s]->levelOneChannels[randsynd][i][j];
							}
						}
					}
					channels[0][b][0][qcode[0]->nlogs][0] = sims[0]->levelOneImpDist[randsynd]/sims[1]->levelOneSynds[randsynd];
					channels[0][b][0][qcode[0]->nlogs][1] = sims[0]->levelOneSynds[randsynd];
					channels[0][b][0][qcode[0]->nlogs][2] = sims[0]->levelOneSynds[randsynd];

					channels[0][b][1][qcode[0]->nlogs][0] = 1.0;
					channels[0][b][1][qcode[0]->nlogs][1] = 1.0;
					channels[0][b][1][qcode[0]->nlogs][2] = sims[1]->levelOneSynds[randsynd];
				}
				else
					continue;
			}
			// printf("Loaded the level-1 channels in the tree.\n")
			// Compute average logical channels and average logical error rates.
			ComputeLogicalChannels(sims, qcode, consts, channels);
			// printf("Computed logical channels.\n")
			for (s = 0; s < 1 + (int)(sims[0]->importance == 2); s ++){
				// printf("sims[s]->runstats[0] = %d\n", sims[s]->runstats[0]);
				// PrintIntArray1D(sims[s]->runstats, "runstats", sims[s]->runstats[0] + 1);
				if (IsElement(sims[s]->runstats, stat + 1) == 1){
					for (m = 0; m < sims[0]->nmetrics; m ++){
						sims[s]->runavg[0][m][0] = sims[s]->runavg[0][m][0] + 1;
						sims[s]->runavg[0][m][<int>(sims[s]->runavg[0][m][0])] = sims[s]->metricValues[sims[0]->nlevels][m];
						sims[s]->runavg[1][m][0] = sims[s]->runavg[1][m][0] + 1;
					}
				}
			}
		}
	}
	// PrintDoubleArray2D(sims[0]->runavg[0], "running average", sims[0]->nmetrics, sims[0]->runstats[0]);
	// PrintDoubleArray2D(sims[0]->runavg[1], "running variance", sims[0]->nmetrics, sims[0]->runstats[0]);
	// printf("Done merging\n");
	
	// Normalize the average metrics.
	for (level = 1; level < sims[0]->nlevels; level ++)
		UpdateMetrics(level, 1.0, 1.0, 1, qcode[level], sims[0], consts);

	printf("Updated metrics\n");

	// Compute the variance from the sum of squares.
	// PrintDoubleArray2D(sims[0]->sumsq, "sumsq", sims[0]->nlevels, sims[0]->nmetrics + qcode[0]->nlogs * qcode[0]->nlogs);
	for (s = 0; s < 1 + (int)(sims[0]->importance == 2); s ++){
		for (level = 1; level < sims[0]->nlevels; level ++){
			for (m = 0; m < sims[0]->nmetrics + qcode[level]->nlogs * qcode[level]->nlogs; m ++){
				(sims[s]->variance)[level + 1][m] = 1/((long double)((sims[s]->statsperlevel)[level + 1] - 1) * (sims[s]->statsperlevel)[level + 1]) * sims[s]->sumsq[level + 1][m];
				if (m < sims[s]->nmetrics){
					(sims[s]->variance)[level + 1][m] = (sims[s]->variance)[level + 1][m] - 1/((long double)((sims[s]->statsperlevel)[level + 1] - 1)) * (sims[s]->metricValues)[level + 1][m] * (sims[s]->metricValues)[level + 1][m];
				}
				else{
					(sims[s]->variance)[level + 1][m] = (sims[s]->variance)[level + 1][m] - 1/((long double)((sims[s]->statsperlevel)[level + 1] - 1)) * (sims[s]->logical)[level + 1][m/(qcode[level]->nlogs)][m % (qcode[level]->nlogs)] * (sims[s]->logical)[level + 1][m/(qcode[level]->nlogs)][m % (qcode[level]->nlogs)];
				}
				(sims[s]->variance)[level + 1][m] = sqrtl((sims[s]->variance)[level + 1][m]);
			}
		}
	}
	printf("Computed variance\n");

	// Free memory for channels.
	MemManageChannels(channels, nbatches, qcode, sims[0]->nlevels, sims[0]->importance, 1);
	free(channels);

	printf("Freed channels\n");
	printf("Done Performance.\n");
}
########################


cdef int UpdateMetrics(int level, long double bias, long double history, int isfinal, qecc_t *qcode, simul_t *sim, constants_t *consts) nogil:
	# Compute metrics for all the effective channels and update the average value of the metrics
	cdef:
		int r, c, m, s, mbin = 0
		## Metric values
		long double *metvals = <long double *>malloc(sim[0].nmetrics * sizeof(long double))
		long double *avg = <long double *>malloc((sim[0].nmetrics + qcode[0].nlogs * qcode[0].nlogs) * sizeof(long double))
	for m in range(sim[0].nmetrics + qcode[0].nlogs * qcode[0].nlogs):
		if (m < sim[0].nmetrics):
			metvals[m] = 0.0
		avg[m] = 0.0
	
	if (isfinal == 0):
		for s in range(qcode[0].nstabs):
			if (sim[0].syndprobs[s] > consts[0].atol):
				# Compute metrics
				ComputeMetrics(metvals, sim[0].nmetrics, sim[0].metricsToCompute, sim[0].effective[s], sim[0].chname)
				# PrintDoubleArray2D(sim[0].effprocess[s], "effprocess", qcode[0].nlogs, qcode[0].nlogs)
				# PrintDoubleArray1D(metvals, "metrics", sim[0].nmetrics)
				for m in range(sim[0].nmetrics):
					avg[m] = avg[m] + metvals[m] * sim[0].syndprobs[s]
				# Compute average channel
				for r in range(qcode[0].nlogs):
					for c in range(qcode[0].nlogs):
						avg[sim[0].nmetrics + r * qcode[0].nlogs + c] = avg[sim[0].nmetrics + r * qcode[0].nlogs + c] + sim[0].effprocess[s][r][c] * sim[0].syndprobs[s]
			else:
				sim[0].syndprobs[s] = 0.0

			# with gil:
			# 	PrintDoubleArray2D(sim[0].effprocess[s], "Effective process matrix for level %d, P(%s) = %g." % (level, s, sim[0].syndprobs[s]), qcode[0].nlogs, qcode[0].nlogs)
		
		# printf("Function: UpdateMetrics, history = %Lf, bias = %Lf, isfinal = %d\n", history, bias, isfinal)

		# Average of metrics
		for m in range(sim[0].nmetrics):
			sim[0].metricValues[level + 1][m] = sim[0].metricValues[level + 1][m] + bias * avg[m]
			sim[0].sumsq[level + 1][m] = sim[0].sumsq[level + 1][m] + avg[m] * avg[m] * bias * bias
		for r in range(qcode[0].nlogs):
			for c in range(qcode[0].nlogs):
				sim[0].logical[level + 1][r][c] = sim[0].logical[level + 1][r][c] + bias * avg[sim[0].nmetrics + r * qcode[0].nlogs + c]
				sim[0].sumsq[level + 1][sim[0].nmetrics + r * qcode[0].nlogs + c] = sim[0].sumsq[level + 1][sim[0].nmetrics + r * qcode[0].nlogs + c] + (avg[sim[0].nmetrics + r * qcode[0].nlogs + c] * avg[sim[0].nmetrics + r * qcode[0].nlogs + c])

		# PrintDoubleArray1D(sim[0].metricValues[level + 1], "sim[0].metricValues[level + 1]", sim[0].nmetrics)
		# PrintDoubleArray1D(sim[0].sumsq[level + 1], "sim[0].sumsq[level + 1]", sim[0].nmetrics + qcode[0].nlogs * qcode[0].nlogs)
		
		## Syndrome-metric binning
		sbin = GetBinPosition(fabsl(history), sim[0].nbins, sim[0].maxbin)
		for m in range(sim[0].nmetrics):
			mbin = GetBinPosition(fabsl(avg[m]), sim[0].nbins, sim[0].maxbin)
			sim[0].bins[level + 1][m][sbin][mbin] = sim[0].bins[level + 1][m][sbin][mbin] + 1
		# Update the number of statistics done for the level
		sim[0].statsperlevel[level + 1] = sim[0].statsperlevel[level + 1] + 1
	
	else:
		# After all the simulations are done, the average metrics are to be computed by diving the metricValues by the total number of statistics done for that level.
		for m in range(sim[0].nmetrics):
			sim[0].metricValues[level + 1][m] = sim[0].metricValues[level + 1][m]/(<long double>sim[0].statsperlevel[level + 1])
			# sim[0].variance[level + 1][m] = 1/(<long double>(sim[0].statsperlevel[level + 1] * (sim[0].statsperlevel[level + 1] - 1))) * sim[0].sumsq[level + 1][m] - 1/(<long double>(sim[0].statsperlevel[level + 1] - 1)) * sim[0].metricValues[level + 1][m] * sim[0].metricValues[level + 1][m]
		for r in range(qcode[0].nlogs):
			for c in range(qcode[0].nlogs):
				sim[0].logical[level + 1][r][c] = sim[0].logical[level + 1][r][c]/(<long double>sim[0].statsperlevel[level + 1])
				# sim[0].variance[level + 1][sim[0].nmetrics + r * qcode[0].nlogs + c] = 1/(<long double>(sim[0].statsperlevel[level + 1] * (sim[0].statsperlevel[level + 1] - 1))) * sim[0].sumsq[level + 1][sim[0].nmetrics + r * qcode[0].nlogs + c] - 1/(<long double>(sim[0].statsperlevel[level + 1] - 1)) * sim[0].logical[level + 1][r][c] * sim[0].logical[level + 1][r][c]
		# PrintDoubleArray1D(sim[0].sumsq[level + 1], "normalized sim[0].sumsq[level + 1]", sim[0].nmetrics + qcode[0].nlogs * qcode[0].nlogs)
		# with gil:
		# 	PrintDoubleArray1D(sim[0].metricValues[level + 1], "Level %d Metric values" % (level + 1), sim[0].nmetrics)
	## Free memory
	free(<void *>metvals)
	free(<void *>avg)
	return 0

########################
# Converted to C

void UpdateMetrics(int level, long double bias, long double history, int isfinal, qecc_t *qcode, simul_t *sim, constants_t *consts){
	// Compute metrics for all the effective channels and update the average value of the metrics.
	int r, c, m, s, mbin = 0;
	// Metric values.
	long double *metvals = <long double *>malloc(sim->nmetrics * sizeof(long double));
	long double *avg = <long double *>malloc((sim->nmetrics + qcode->nlogs * qcode->nlogs) * sizeof(long double));
	for (m = 0; m < sim->nmetrics + qcode->nlogs * qcode->nlogs; m ++){
		if (m < sim->nmetrics)
			metvals[m] = 0;
		avg[m] = 0;
	}

	if (isfinal == 0){
		for (s = 0; s < qcode->nstabs; s ++){
			if ((sim->syndprobs)[s] > consts->atol){
				// Compute metrics.
				ComputeMetrics(metvals, sim->nmetrics, sim->metricsToCompute, sim->effective[s], sim->chname);
				// PrintDoubleArray2D(sim->effprocess[s], "effprocess", qcode->nlogs, qcode->nlogs);
				// PrintDoubleArray1D(metvals, "metrics", sim->nmetrics);
				for (m = 0; m < sim->nmetrics; m ++)
					avg[m] = avg[m] + metvals[m] * (sim->syndprobs)[s];
				// Compute average channel.
				for (r = 0; r < qcode->nlogs; r ++){
					for (c = 0; c < qcode->nlogs; c ++){
						avg[sim->nmetrics + r * qcode->nlogs + c] = avg[sim->nmetrics + r * qcode->nlogs + c] + (sim->effprocess)[s][r][c] * (sim->syndprobs)[s];
					}
				}
			}
			else
				(sim->syndprobs)[s] = 0.0;

			// PrintDoubleArray2D((sim->effprocess)[s], "Effective process matrix for level %d, P(%s) = %g." % (level, s, (sim->syndprobs)[s]), qcode->nlogs, qcode->nlogs);
		}

		// printf("Function: UpdateMetrics, history = %Lf, bias = %Lf, isfinal = %d\n", history, bias, isfinal)
		// Average of metrics.
		for (m = 0; m < sim->nmetrics; m ++){
			(sim->metricValues)[level + 1][m] = (sim->metricValues)[level + 1][m] + bias * avg[m];
			(sim->sumsq)[level + 1][m] = (sim->sumsq)[level + 1][m] + avg[m] * avg[m] * bias * bias;
		}
		for (r = 0; r < qcode->nlogs; r ++){
			for (c = 0; c < qcode->nlogs; c ++){
				(sim->logical)[level + 1][r][c] = (sim->logical)[level + 1][r][c] + bias * avg[sim->nmetrics + r * qcode->nlogs + c]
				(sim->sumsq)[level + 1][sim->nmetrics + r * qcode->nlogs + c] = (sim->sumsq)[level + 1][sim->nmetrics + r * qcode->nlogs + c] + (avg[sim->nmetrics + r * qcode->nlogs + c] * avg[sim->nmetrics + r * qcode->nlogs + c]);
			}
		}

		// PrintDoubleArray1D((sim->metricValues)[level + 1], "(sim->metricValues)[level + 1]", sim->nmetrics);
		// PrintDoubleArray1D((sim->sumsq)[level + 1], "(sim->sumsq)[level + 1]", sim->nmetrics + qcode->nlogs * qcode->nlogs);

		// Syndrome-metric binning.
		sbin = GetBinPosition(fabsl(history), sim->nbins, sim->maxbin);
		for (m = 0; m < sim->nmetrics; m ++){
			mbin = GetBinPosition(fabsl(avg[m]), sim->nbins, sim->maxbin);
			(sim->bins)[level + 1][m][sbin][mbin] = (sim->bins)[level + 1][m][sbin][mbin] + 1;
		}
		// Update the number of statistics done for the level.
		(sim->statsperlevel)[level + 1] = (sim->statsperlevel)[level + 1] + 1;
	}
	else{
		// After all the simulations are done, the average metrics are to be computed by diving the metricValues by the total number of statistics done for that level.
		for (m = 0; m < sim->nmetrics; m ++){
			(sim->metricValues)[level + 1][m] = (sim->metricValues)[level + 1][m]/((long double)((sim->statsperlevel)[level + 1]));
			(sim->variance)[level + 1][m] = 1/(<long double>((sim->statsperlevel)[level + 1] * ((sim->statsperlevel)[level + 1] - 1))) * sim->sumsq[level + 1][m] - 1/(<long double>((sim->statsperlevel)[level + 1] - 1)) * (sim->metricValues)[level + 1][m] * (sim->metricValues)[level + 1][m];
		}
		for (r = 0; r < qcode->nlogs; r ++){
			for (c = 0; c < qcode->nlogs; c ++){
				(sim->logical)[level + 1][r][c] = (sim->logical)[level + 1][r][c]/((long double)((sim->statsperlevel)[level + 1]));
			}
		}
		// PrintDoubleArray1D((sim->sumsq)[level + 1], "normalized (sim->sumsq)[level + 1]", sim->nmetrics + qcode->nlogs * qcode->nlogs);
		// printf("Level %d Metric values\n", level);
		// PrintDoubleArray1D((sim->metricValues)[level + 1], "Metric values", sim->nmetrics);
	}
	// Free memory.
	free(metvals);
	free(avg);
}
########################


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


########################
# Converted to C

int GetBinPosition(long double number, int nbins, int maxbin){
	// Place a number into one of the bins depending on the order of magnitude of the number.
	// If a number is of the order of magnitude of 10^-i, then return i.
	// To find the order of magnitude we will take the negative log and bin it from 0 to 20.
	int binindex = (int)((-1) * log10(number)/((float) maxbin) * ((float) nbins));
	if (binindex < 0)
		binindex = 0;
	if (binindex > (nbins - 1))
		binindex = (nbins - 1);
	return binindex;
}
########################