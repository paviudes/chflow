#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False
from libc.stdio cimport printf, fprintf, stderr, fopen, fclose, FILE
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
	
	# printf("Function: AllocSimParamsQECC, nqecc = %d, nstabs = %d, nlogs = %d\n", nqecc, nstabs, nlogs)

	cdef int c, l, q, r, s
	## Logical channels at intermediate concatenation levels
	simul[0].virtual = <long double ***>malloc(nqecc * sizeof(long double **))
	for q in range(nqecc):
		simul[0].virtual[q] = <long double **>malloc(nlogs * sizeof(long double *))
		for r in range(nlogs):
			simul[0].virtual[q][r] = <long double *>malloc(nlogs * sizeof(long double))
			for c in range(nlogs):
				simul[0].virtual[q][r][c] = 0.0
	# printf("_/ virtual\n")
	## Syndrome sampling
	simul[0].syndprobs = <long double *>malloc(nstabs * sizeof(long double))
	simul[0].cumulative = <long double *>malloc(nstabs * sizeof(long double))
	for s in range(nstabs):
		simul[0].syndprobs[s] = 0.0
		simul[0].cumulative[s] = 0.0
	# printf("_/ syndprobs, cumulative\n")
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
	# printf("_/ process\n")
	simul[0].corrections = <int *>malloc(nstabs * sizeof(int))
	for s in range(nstabs):
		simul[0].corrections[s] = 0
	# printf("_/ corrections\n")
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
	# printf("_/ effective, effprocess\n")
	return 0

######################
# Convereted to C

void AllocSimParamsQECC(simul_t *simul, int nqecc, int nstabs, int nlogs){
	// Allocate memory for parameters of the simulation structure that depend upon the QECC used.
	// This memeory will be reallocated everytime there is a new QECC, i.e, at a new concatenation level.
	// These parameters are
	// virtual, logical, syndprobs, cumulative, levelZeroSynds, levelZeroCumulative, levelZeroImpDist, levelZeroImpCumul, process, corrections, effprocess, effective, levelZeroChannels
	printf("Function: AllocSimParamsQECC, nqecc = %d, nstabs = %d, nlogs = %d\n", nqecc, nstabs, nlogs);
	int c, l, q, r, s;
	simul->virtual = malloc(nqecc * sizeof(long double **));
	for (q = 0; q < nqecc; q ++){
		(simul->virtual)[q] = malloc(nlogs * sizeof(long double *));
		for (r = 0; r < nlogs; r ++){
			(simul->virtual)[q][r] = malloc(nlogs * sizeof(long double));
			for (c = 0; c < nlogs; c ++){
				(simul->virtual)[q][r][c] = 0.0;
			}
		}
	}
	// printf("_/ virtual\n");
	// Syndrome sampling
	(simul->syndprobs) = malloc(nstabs * sizeof(long double));
	(simul->cumulative) = malloc(nstabs * sizeof(long double));
	for (s = 0 s < nstabs; s ++){
		(simul->syndprobs)[s] = 0.0;
		(simul->cumulative)[s] = 0.0;
	}
	// printf("_/ syndprobs, cumulative\n");
	// Quantum error correction
	simul->process = malloc(nlogs * sizeof(long double ***));
	for (r = 0; r < nlogs; r ++){
		(simul->process)[r] = malloc(nlogs * sizeof(long double **));
		for (c = 0; c < nlogs; c ++){
			(simul->process)[r][c] = malloc(nstabs * sizeof(long double *));
			for (i = 0; i < nstabs; i ++){
				(simul->process)[r][c][i] = malloc(nstabs * sizeof(long double));
				for (s = 0; s < nstabs; s ++){
					(simul->process)[r][c][i][s] = 0.0;
				}
			}
		}
	}
	// printf("_/ process\n");
	(simul->corrections) = malloc(nstabs * sizeof(int));
	for (s = 0; s < nstabs; s ++)
		(simul->corrections)[s] = 0;
	// printf("_/ corrections\n");
	(simul->effective) = malloc(nstabs * sizeof(complex128_t **));
	(simul->effprocess) = malloc(nstabs * sizeof(long double **));
	for (s = 0; s < nstabs; s ++){
		(simul->effective)[s] = malloc(nlogs * sizeof(complex128_t *));
		(simul->effprocess)[s] = malloc(nlogs * sizeof(long double *));
		for (r = 0; r < nlogs; r ++){
			(simul->effective)[s][r] = malloc(nlogs * sizeof(complex128_t));
			(simul->effprocess)[s][r] = malloc(nlogs * sizeof(long double));
			for (c = 0; c < nlogs; c ++){
				(simul->effective)[s][r][c] = 0.0 + 0.0 * I;
				(simul->effprocess)[s][r][c] = 0.0;
			}
		}
	}
	// printf("_/ effective, effprocess\n");
}
######################

cdef int AllocSimParams(simul_t *simul, int nstabs, int nlogs):
	# initialize the elements that pertain to the montecarlo simulation of channels

	# printf("Function: AllocSimParams, nstabs = %d, nlogs = %d\n", nstabs, nlogs)

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
	simul[0].outlierprobs[0] = 1
	simul[0].outlierprobs[1] = 1
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
	simul[0].sumsq = <long double **>malloc((simul[0].nlevels + 1) * sizeof(long double *))
	for l in range(simul[0].nlevels + 1):
		simul[0].sumsq[l] = <long double *>malloc((simul[0].nmetrics + nlogs * nlogs) * sizeof(long double))
		for m in range(simul[0].nmetrics + nlogs * nlogs):
			simul[0].sumsq[l][m] = 0.0
	# fprintf(stderr, "sumsq\n")
	# PrintDoubleArray2D(simul[0].sumsq, "sumsq", simul[0].nlevels + 1, simul[0].nmetrics + nlogs * nlogs)
	simul[0].variance = <long double **>malloc((simul[0].nlevels + 1) * sizeof(long double *))
	for l in range(simul[0].nlevels + 1):
		simul[0].variance[l] = <long double *>malloc((simul[0].nmetrics + nlogs * nlogs) * sizeof(long double))
		for m in range(simul[0].nmetrics + nlogs * nlogs):
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


###########################
# Converted to C.

void AllocSimParams(simul_t *simul, int nstabs, int nlogs){
	// Initialize the elements that pertain to the montecarlo simulation of channels.
	// printf("Function: AllocSimParams, nstabs = %d, nlogs = %d\n", nstabs, nlogs);
	int s, l, r, c, m;
	// Physical channels.
	simul->physical = malloc(nlogs * sizeof(long double *));
	for (r = 0; r < nlogs; r ++){
		(simul->physical)[r] = malloc(nlogs * sizeof(long double));
		for (c = 0; c < nlogs; c ++){
			(simul->physical)[r][c] = 0.0;
		}
	}
	simul->chname = malloc(100 * sizeof(char));
	// Metrics to be computed at every level.
	simul->metricsToCompute = malloc(simul->nmetrics * sizeof(char *));
	for (m = 0; m < simul->nmetrics; m ++){
		(simul->metricsToCompute)[m] = malloc(100 * sizeof(char));
	}
	simul->metricValues = malloc((simul->nlevels + 1) * sizeof(long double *));
	for (l = 0; l < simul->nlevels + 1; l ++){
		(simul->metricValues)[l] = malloc(simul->nmetrics * sizeof(long double));
		for (m = 0; m < simul->nmetrics; m ++){
			(simul->metricValues)[l][m] = 0.0;
		}
	}
	// Average logical channel at top level.
	simul->logical = malloc((simul->nlevels + 1) * sizeof(long double **));
	for (l = 0; l < simul->nlevels + 1; l ++){
		(simul->logical)[l] = malloc(nlogs * sizeof(long double *));
		for (r = 0; r < nlogs; r ++){
			(simul->logical)[l][r] = malloc(nlogs * sizeof(long double));
			for (c = 0; c < nlogs; c ++){
				(simul->logical)[l][r][c] = 0.0;
			}
		}
	}
	// Syndrome sampling.
	simul->levelOneSynds = malloc(nstabs * sizeof(long double));
	simul->levelOneImpDist = malloc(nstabs * sizeof(long double));
	simul->levelOneCumul = malloc(nstabs * sizeof(long double));
	simul->levelOneImpCumul = malloc(nstabs * sizeof(long double));
	for (s = 0; s < nstabs; s ++){
		(simul->levelOneSynds)[s] = 0.0;
		(simul->levelOneImpDist)[s] = 0.0;
		(simul->levelOneCumul)[s] = 0.0;
		(simul->levelOneImpCumul)[s] = 0.0;
	}
	simul->statsperlevel = malloc((simul->nlevels + 1) * sizeof(long));
	for (l = 0; l < simul->nlevels + 1; l ++){
		(simul->statsperleve)l[l] = 0;
	}
	// Upper and lower limits for the probability of the outlier syndromes.
	simul->outlierprobs = malloc(sizeof(long double) * 2);
	(simul->outlierprobs)[0] = 1;
	(simul->outlierprobs)[1] = 1;
	// Syndrome-metric bins.
	simul->bins = malloc((simul->nlevels + 1) * sizeof(int ***));
	for (l = 0; l < simul->nlevels + 1; l ++){
		(simul->bins)[l] = malloc((simul->nmetrics) * sizeof(int **));
		for (m = 0; m < simul->nmetrics; m ++){
			(simul->bins)[l][m] = malloc(simul->nbins * sizeof(int *));
			for (r = 0; r < simul->nbins; r ++){
				(simul->bins)[l][m][r] = malloc(simul->nbins * sizeof(int));
				for (c = 0; c < simul->nbins; c ++){
					(simul->bins)[l][m][r][c] = 0;
				}
			}
		}
	}
	// Variance measures.
	simul->sumsq = malloc((simul->nlevels + 1) * sizeof(long double *));
	for (l = 0; l < simul->nlevels + 1; l ++){
		(simul->sumsq)[l] = malloc((simul->nmetrics + nlogs * nlogs) * sizeof(long double));
		for (m = 0; m < simul->nmetrics + nlogs * nlogs; m ++){
			(simul->sumsq)[l][m] = 0.0;
		}
	}
	// fprintf(stderr, "_/ sumsq, variance.\n");
	// PrintDoubleArray2D(simul[0].sumsq, "sumsq", simul[0].nlevels + 1, simul[0].nmetrics + nlogs * nlogs).
	simul->variance = <long double **>malloc((simul->nlevels + 1) * sizeof(long double *));
	for (l = 0; l < simul->nlevels + 1; l ++){
		(simul->variance)[l] = <long double *>malloc((simul->nmetrics + nlogs * nlogs) * sizeof(long double));
		for (m = 0; m < simul->nmetrics + nlogs * nlogs; m ++){
			(simul->variance)[l][m] = 0.0;
		}
	}
	// Quantum error correction.
	simul->levelOneChannels = malloc(nstabs * sizeof(long double **));
	for (s = 0; s < nstabs; s ++){
		(simul->levelOneChannels)[s] = malloc(nlogs * sizeof(long double *));
		for (r = 0; r < nlogs; r ++){
			(simul->levelOneChannels)[s][r] = malloc(nlogs * sizeof(long double));
			for (c = 0; r < nlogs; c ++){
				(simul->levelOneChannels)[s][r][c] = 0.0;
			}
		}
	}
	simul->frames = malloc(simul->nlevels * sizeof(int));
	for (l = 0; l < simul->nlevels; l ++){
		(simul->frames)[l] = 0;
	}
	// fprintf(stderr, "_/ levelOneChannels, frames.\n");
}
###########################


cdef int FreeSimParamsQECC(simul_t *simul, int nqecc, int nstabs, int nlogs) nogil:
	# free the memory allocated to simulation parameters that depend on the QECC
	# These parameters are
	# virtual, logical, syndprobs, cumulative, levelZeroSynds, levelZeroCumulative, levelZeroImpDist, levelZeroImpCumul, process, corrections, effprocess, effective, levelZeroChannels
	# printf("FreeSimParamsQECC: nqecc = %d, nstabs = %d, nlogs = %d\n", nqecc, nstabs, nlogs)
	cdef int c, l, q, r, s
	## Logical channels at intermediate levels
	for q in range(nqecc):
		for r in range(nlogs):
			# printf("q = %d, r = %d\n", q, r)
			free(<void *>simul[0].virtual[q][r])
		free(<void *>simul[0].virtual[q])
	free(<void *>simul[0].virtual)	
	# printf("_/ virtual\n")
	## Quantum error correction
	for r in range(nlogs):
		for c in range(nlogs):
			for s in range(nstabs):
				free(<void *>simul[0].process[r][c][s])
			free(<void *>simul[0].process[r][c])
		free(<void *>simul[0].process[r])
	free(<void *>simul[0].process)
	free(<void *>simul[0].corrections)
	# printf("_/ process, corrections\n")
	for s in range(nstabs):
		for r in range(nlogs):
			free(<void *>simul[0].effective[s][r])
			free(<void *>simul[0].effprocess[s][r])
		free(<void *>simul[0].effective[s])
		free(<void *>simul[0].effprocess[s])
	free(<void *>simul[0].effective)
	free(<void *>simul[0].effprocess)
	# printf("_/ effective, effprocess\n")
	## Syndrome sampling
	free(<void *>simul[0].syndprobs)
	free(<void *>simul[0].cumulative)
	# printf("_/ syndprobs, cumulative\n")
	# printf("Done\n")
	return 0

###################
# Converted to C.

void FreeSimParamsQECC(simul_t *simul, int nqecc, int nstabs, int nlogs){
	// Free the memory allocated to simulation parameters that depend on the QECC.
	// These parameters are
	// virtual, logical, syndprobs, cumulative, levelZeroSynds, levelZeroCumulative, levelZeroImpDist, levelZeroImpCumul, process, corrections, effprocess, effective, levelZeroChannels.
	// printf("FreeSimParamsQECC: nqecc = %d, nstabs = %d, nlogs = %d\n", nqecc, nstabs, nlogs);
	int c, l, q, r, s;
	// Logical channels at intermediate levels.
	for (q = 0; q < nqecc; q ++){
		for (r = 0; r < nlogs; r ++){
			free((simul->virtual)[q][r]);
		}
		free((simul->virtual)[q]);
	}
	free(simul->virtual);
	// printf("_/ virtual\n");
	// Quantum error correction.
	for (r = 0; r < nlogs; r ++){
		for (c = 0; c < nlogs; c ++){
			for (s = 0; s < nstabs; s ++){
				free((simul->process)[r][c][s]);
			}
			free((simul->process)[r][c]);
		}
		free((simul->process)[r]);
	}
	free(simul->process);
	free(simul->corrections);
	// printf("_/ process, corrections\n");
	for (s = 0; s < nstabs; s ++){
		for (r = 0; r < nlogs; r ++){
			free((simul->effective)[s][r]);
			free((simul->effprocess)[s][r]);
		}
		free(simul->effective[s]);
		free(simul->effprocess[s]);
	}
	free(simul->effective);
	free(simul->effprocess);
	// printf("_/ effective, effprocess\n");
	// Syndrome sampling.
	free(simul->syndprobs);
	free(simul->cumulative);
	// printf("_/ syndprobs, cumulative\n");
	// printf("Done\n")
}
###################


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
		free(<void *>simul[0].sumsq[i])
	free(<void *>simul[0].sumsq)
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

###################
# Converted to C.

void FreeSimParams(simul_t *simul, int nstabs, int nlogs){
	// Free memory allocated to the simulation structure.
	int l, i, s, r, c, g, m;
	// Physical channels.
	for (r = 0; r < nlogs; r ++){
		free((simul->physical)[r]);
	}
	free(simul->physical);
	free(simul->chname);
	// Metrics to be computed at logical levels.
	free(simul->metricsToCompute);
	for (l = 0; l < 1 + simul->nlevels; l ++){
		free((simul->metricValues)[l]);
	}
	free(simul->metricValues);
	// Average logical channel at the top level.
	for (l = 0; l < 1 + simul->nlevels; l ++){
		for (r = 0; r < nlogs; r ++){
			free((simul->logical)[l][r]);
		}
		free((simul->logical)[l]);
	}
	free(simul->logical);
	// Syndrome sampling.
	free(simul->statsperlevel);
	free(simul->outlierprobs);
	free(simul->levelOneSynds);
	free(simul->levelOneCumul);
	free(simul->levelOneImpDist);
	free(simul->levelOneImpCumul);
	// Variance measure.
	for (l = 0; l < 1 + simul->nlevels; l ++){
		free((simul->sumsq)[l]);
	}
	free(simul->sumsq);
	for (l = 0; l < 1 + simul->nlevels; l ++){
		free((simul->variance)[l]);
	}
	free(simul->variance);
	// Syndrome metric bins.
	for (l = 0; l < 1 + simul->nlevels; l ++){
		for (m = 0; m < 1 + simul->nmetrics; m ++){
			for (r = 0; r < simul->nbins; r ++){
				free((simul->bins)[l][m][r]);
			}
			free((simul->bins)[l][m]);
		}
		free((simul->bins)[l]);
	}
	free(simul->bins);
	// Quantum error correction.
	for (s = 0; s < nstabs; s ++){
		for (r = 0; r < simul->nlogs; r ++){
			free(simul[0].levelOneChannels[s][r]);
		}
		free(simul[0].levelOneChannels[s]);
	}
	free(simul[0].levelOneChannels);
	free(simul[0].frames);
}
###################


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
		# 3. average metric values and sum of squares
		for m in range(parent[0].nmetrics):
			parent[0].metricValues[l + 1][m] = (parent[0].metricValues[l + 1][m]) * (<int>(l > 0)) + child[0].metricValues[l + 1][m]
			parent[0].sumsq[l + 1][m] = (parent[0].sumsq[l + 1][m]) * (<int>(l > 0)) + child[0].sumsq[l + 1][m]
		# 4. Binning information
		for m in range(parent[0].nmetrics):
			for i in range(parent[0].nbins):
				for j in range(parent[0].nbins):
					parent[0].bins[l + 1][m][i][j] = (parent[0].bins[l + 1][m][i][j]) * (<int>(l > 0)) + child[0].bins[l + 1][m][i][j]
	return 0



cdef int CountIndepLogicalChannels(int *chans, qecc_t **qecc, int nlevels) nogil:
	# Determine the number of independent logical channels at every level, that determine the logical channels of higher levels.
	# printf("Function: CountIndepLogicalChannels\n")
	cdef:
		int i, l, nchans
	for l in range(nlevels):
		chans[l] = 1
		for i in range(l, nlevels):
			chans[l] = chans[l] * qecc[i][0].N
	nchans = chans[0]
	# PrintIntArray1D(chans, "chans", nlevels)
	# printf("Done\n")
	return nchans

##############
# Converted to C

int CountIndepLogicalChannels(int *chans, qecc_t **qecc, int nlevels){
	// Determine the number of independent logical channels at every level, that determine the logical channels of higher levels.
	// printf("Function: CountIndepLogicalChannels\n")
	int i, l, nchans;
	for (l = 0; l < nlevels; l ++){
		chans[l] = 1;
		for (i = l; i < nlevels; i ++){
			chans[l] = chans[l] * qecc[i]->N;
		}
	}
	nchans = chans[0];
	// PrintIntArray1D(chans, "chans", nlevels);
	// printf("Done\n");
	return nchans;
}
##############


cdef int MemManageChannels(long double ******channels, int nbatches, qecc_t **qecc, int nlevels, int importance, int tofree):
	# Allocate and Free memory for the tree of lower-level channels which determine a logical channel.
	cdef:
		int b, i, l, s, sb, nchans
		int *chans = <int *>malloc(sizeof(int) * nlevels)
	# printf("Function: MemManageChannels, tofree = %d\n", tofree)
	nchans = CountIndepLogicalChannels(chans, qecc, nlevels)
	if (tofree == 0):
		# Allocate memory
		channels = <long double *****>malloc(sizeof(long double ****) * nlevels)
		for l in range(nlevels):
			channels[l] = <long double ****>malloc(sizeof(long double ***) * chans[l])
			for sb in range(chans[l]):
				channels[l][sb] = <long double ***>malloc(sizeof(long double **) * (1 + <int>(importance == 2)))
				for s in range(1 + <int>(importance == 2)):
					channels[l][sb][s] = <long double **>malloc(sizeof(long double *) * (1 + qecc[l][0].nlogs))
					for i in range(qecc[l][0].nlogs):
						channels[l][sb][s][i] = <long double *>malloc(sizeof(long double) * qecc[l][0].nlogs)
					channels[l][sb][s][qecc[l][0].nlogs] = <long double *>malloc(sizeof(long double) * 3)
	else:
		# free memory
		# printf("Going to free: nchans = %d\n", nchans)
		for l in range(nlevels):
			for sb in range(<int> chans[l]):
				for s in range(1 + <int>(importance == 2)):
					for i in range(qecc[l][0].nlogs):
						free(<void *>channels[l][sb][s][i])
						# printf("i = %d\n", i)
					free(<void *>channels[l][sb][s])
					# printf("s = %d\n", s)
				free(<void *>channels[l][sb])
				# printf("sb = %d\n", sb)
			free(<void *>channels[l])
			# printf("l = %d\n", l)
		# printf("All freed\n")
	free(<void *>chans)
	return nchans

######################
# Converted in C.

int MemManageChannels(long double ******channels, int nbatches, qecc_t **qecc, int nlevels, int importance, int tofree){
	// Allocate and Free memory for the tree of lower-level channels which determine a logical channel.
	int b, i, l, s, sb, nchans;
	int *chans = malloc(sizeof(int) * nlevels);
	// printf("Function: MemManageChannels, tofree = %d\n", tofree);
	nchans = CountIndepLogicalChannels(chans, qecc, nlevels);
	if (tofree == 0){
		// Allocate memory.
		channels = malloc(sizeof(long double ****) * nlevels);
		for (l = 0; l < nlevels; l ++){
			channels[l] = malloc(sizeof(long double ***) * chans[l]);
			for (sb = 0; sb < chans[l]; sb ++){
				channels[l][sb] = malloc(sizeof(long double **) * (1 + <int>(importance == 2)));
				for (s = 0; s < 1 + (int)(importance == 2); s ++){
					channels[l][sb][s] = malloc(sizeof(long double *) * (1 + qecc[l]->nlogs));
					for (i = 0; i < qecc[l]->nlogs; i ++){
						channels[l][sb][s][i] = malloc(sizeof(long double) * qecc[l]->nlogs);
					}
					channels[l][sb][s][qecc[l]->nlogs] = malloc(sizeof(long double) * 3);
				}
			}
		}
	}
	else{
		// Free memory.
		// printf("Going to free: nchans = %d\n", nchans);
		for (l = 0; l < nlevels; l ++){
			for (sb = 0; sb < chans[l]; sb ++){
				for (s = 0; s < 1 + (int)(importance == 2); s ++){
					for (i = 0; i < qecc[l]->nlogs; i ++){
						free(channels[l][sb][s][i]);
						// printf("i = %d\n", i);
					}
					free(channels[l][sb][s]);
					// printf("s = %d\n", s);
				}
				free(channels[l][sb]);
				// printf("sb = %d\n", sb);
			}
			free(channels[l]);
			// printf("l = %d\n", l);
		}
		// printf("All freed\n");
	}
	free(chans);
	return nchans;
}
######################


cdef int MemManageInputChannels(long double ****inputchannels, int nqecc, int nlogs, int importance, int tofree) nogil:
	# Allocate and free memory for the input channels structure in ComputeLogicalChannels(...)
	# printf("Function: MemManageInputChannels, tofree = %d\n", tofree)
	cdef:
		int i, j, q, s
	if (tofree == 0):
		# Initialize the space required for the input channels
		# inputchannels = <long double ****>malloc(sizeof(long double ***) * nqecc)
		for q in range(nqecc):
			inputchannels[q] = <long double ***>malloc(sizeof(long double **) * (1 + <int>(importance == 2)))
			for s in range(1 + <int>(importance == 2)):
				inputchannels[q][s] = <long double **>malloc(sizeof(long double *) * (nlogs + 1))
				for i in range(nlogs):
					inputchannels[q][s][i] = <long double *>malloc(sizeof(long double) * nlogs)
				inputchannels[q][s][nlogs] = <long double *>malloc(sizeof(long double) * 2)
	else:
		for q in range(nqecc):
			for s in range(1 + <int>(importance == 2)):
				for i in range(1 + nlogs):
					free(<void *>inputchannels[q][s][i])
					# printf("q = %d, s = %d, i = %d\n", q, s, i)
				free(<void *>inputchannels[q][s])
				# printf("q = %d, s = %d\n", q, s)
			# free(<void *>inputchannels[q])
			# printf("q = %d\n", q)
		# free(<void *>inputchannels)
		# printf("freed\n")
	# printf("Done\n")
	return 0


###################
# Converted to C.

void MemManageInputChannels(long double ****inputchannels, int nqecc, int nlogs, int importance, int tofree){
	// Allocate and free memory for the input channels structure in ComputeLogicalChannels(...).
	// printf("Function: MemManageInputChannels, tofree = %d\n", tofree)
	int i, j, q, s;
	if (tofree == 0){
		// Initialize the space required for the input channels.
		// inputchannels = <long double ****>malloc(sizeof(long double ***) * nqecc)
		for (q = 0; q < nqecc; q ++){
			inputchannels[q] = malloc(sizeof(long double **) * (1 + <int>(importance == 2)));
			for (s = 0; s < 1 + (int)(importance == 2); s ++){
				inputchannels[q][s] = malloc(sizeof(long double *) * (nlogs + 1));
				for (i = 0; i < nlogs; i ++){
					inputchannels[q][s][i] = malloc(sizeof(long double) * nlogs);
				}
				inputchannels[q][s][nlogs] = malloc(sizeof(long double) * 2);
			}
		}
	}
	else{
		for (q = 0; q < nqecc; q ++){
			for (s = 0; s < 1 + (int)(importance == 2); s ++){
				for (i = 0; i < nlogs; i ++){
					free(inputchannels[q][s][i]);
					// printf("q = %d, s = %d, i = %d\n", q, s, i);
				}
				free(inputchannels[q][s]);
				// printf("q = %d, s = %d\n", q, s);
			}
			free(inputchannels[q]);
			// printf("q = %d\n", q);
		}
		// free(inputchannels);
		// printf("freed\n");
	}
	// printf("Done\n");
}
###################	