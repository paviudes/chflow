#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include "mt19937/mt19937ar.h"
#include "printfuns.h"
#include "constants.h"
#include "qecc.h"
#include "memory.h"
#include "sampling.h"
#include "checks.h"
#include "logmetrics.h"

int IsElement(long *arr, int size, long item){
	// Determine if an item is present in an array
	// printf("Testing if %d is present in the following array of size %d.\n", item, arr[0]);
	// PrintIntArray1D(arr, "array", arr[0] + 1);
	int i;
	for (i = 0; i < size; i ++)
		if (arr[i] == item)
			return 1;
	return 0;
}

int GetBinPosition(double number, int nbins, int maxbin){
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
			if (alloc == 1)
				found[nelems] = arr[i + 1];
		}
	}
	if (alloc == 1)
		found[0] = nelems;
	return nelems;
}

void UpdateMetrics(int level, double bias, double history, int isfinal, struct qecc_t *qcode, struct simul_t *sim, struct constants_t *consts){
	// Compute metrics for all the effective channels and update the average value of the metrics.
	// Metric values.
	double *metvals = malloc(sim->nmetrics * sizeof(double));
	double *avg = malloc((sim->nmetrics + qcode->nlogs * qcode->nlogs) * sizeof(double));
	int m;
	for (m = 0; m < sim->nmetrics + qcode->nlogs * qcode->nlogs; m ++)
		avg[m] = 0;

	int s, i, j;
	if (isfinal == 0){
		for (s = 0; s < qcode->nstabs; s ++){
			if ((sim->syndprobs)[s] > consts->atol){
				// Compute metrics.
				ComputeMetrics(metvals, sim->nmetrics, sim->metricsToCompute, sim->effective[s], sim->chname, consts);
				for (m = 0; m < sim->nmetrics; m ++)
					avg[m] += metvals[m] * (sim->syndprobs)[s];
				// Compute average channel.
				for (i = 0; i < qcode->nlogs; i ++)
					for (j = 0; j < qcode->nlogs; j ++)
						avg[sim->nmetrics + i * qcode->nlogs + j] += (sim->effprocess)[s][i][j] * (sim->syndprobs)[s];
			}
		}

		// PrintDoubleArray1D(avg, "Avg Metric values and channels", sim->nmetrics + qcode->nlogs * qcode->nlogs);
		// Average of metrics.
		for (m = 0; m < sim->nmetrics; m ++){
			(sim->metricValues)[level + 1][m] += bias * avg[m];
			(sim->sumsq)[level + 1][m] += pow(bias * avg[m], 2);
		}
		for (i = 0; i < qcode->nlogs; i ++){
			for (j = 0; j < qcode->nlogs; j ++){
				(sim->logical)[level + 1][i][j] += bias * avg[sim->nmetrics + i * qcode->nlogs + j];
				(sim->sumsq)[level + 1][sim->nmetrics + i * qcode->nlogs + j] += pow(bias * avg[sim->nmetrics + i * qcode->nlogs + j], 2);
			}
		}

		// Syndrome-metric binning.
		int mbin, sbin = GetBinPosition(fabsl(history), sim->nbins, sim->maxbin);
		for (m = 0; m < sim->nmetrics; m ++){
			mbin = GetBinPosition(fabsl(avg[m]), sim->nbins, sim->maxbin);
			(sim->bins)[level + 1][m][sbin][mbin] ++;
		}
		// Update the number of statistics done for the level.
		(sim->statsperlevel)[level + 1] ++;
	}
	else{
		// printf("(sim->statsperlevel)[%d] = %ld.\n", level + 1, (sim->statsperlevel)[level + 1]);
		// PrintDoubleArray1D(sim->metricValues[level + 1], "Metrics", sim->nmetrics);
		// After all the simulations are done, the average metrics are to be computed by diving the metricValues by the total number of statistics done for that level.
		for (m = 0; m < sim->nmetrics; m ++){
			(sim->metricValues)[level + 1][m] /= ((double)((sim->statsperlevel)[level + 1]));
			(sim->variance)[level + 1][m] = 1/((double)((sim->statsperlevel)[level + 1] * ((sim->statsperlevel)[level + 1] - 1))) * sim->sumsq[level + 1][m] - pow((sim->metricValues)[level + 1][m], 2);
		}
		for (i = 0; i < qcode->nlogs; i ++){
			for (j = 0; j < qcode->nlogs; j ++){
				(sim->logical)[level + 1][i][j] /= ((double)((sim->statsperlevel)[level + 1]));
				(sim->variance)[level + 1][sim->nmetrics + i * qcode->nlogs + j] = 1/((double)((sim->statsperlevel)[level + 1] * ((sim->statsperlevel)[level + 1] - 1))) * sim->sumsq[level + 1][sim->nmetrics + i * qcode->nlogs + j] - pow((sim->logical)[level + 1][i][j], 2);
			}
		}
	}
	// Free memory.
	free(metvals);
	free(avg);

	// printf("Updated metrics.\n");
}

void ProcessToChoi(double **process, int nlogs, double complex **choi, double complex ***pauli){
	// Convert from the process matrix to the Choi matrix.
	// J = 1/K * sum_P (E(P) o P^T)
	// where K is the dimension of the Hilbert space, P runs over Pauli operators and E is the error channel.
	// E(P) = 1/K * sum_Q G_(P,Q) Q
	// where G(P, Q) = Tr(E(P).Q) is the process matrix and Q runs over Pauli operators.
	// hence we have: J = 1/K sum_(P, Q) G_(P,Q) Q o P^T
	int i, j, p, q;
	for (i = 0; i < nlogs; i ++){
		for (j = 0; j < nlogs; j ++){
			// J = J + G[i,j] * pauli[j] o pauli[i]^T
			choi[i][j] = 0;
			for (p = 0; p < nlogs; p ++)
				for (q = 0; q < nlogs; q ++)
					choi[i][j] += (double complex) (0.25 * process[p][q]) * pauli[q][i/2][j/2] * pauli[p][j % 2][i % 2];
		}
	}
	// PrintComplexArray2D(choi, "Choi", nlogs, nlogs);
}

void ComputeLevelZeroMetrics(struct simul_t *sim, int nlogs, struct constants_t *consts){
	// Compute the level-0 (physical) metrics.
	double complex **choi = malloc(sizeof(double complex *) * nlogs);
	int i;
	for (i = 0; i < nlogs; i ++)
		choi[i] = malloc(sizeof(double complex) * nlogs);
	ProcessToChoi((sim->logical)[0], nlogs, choi, consts->pauli);
	ComputeMetrics((sim->metricValues)[0], sim->nmetrics, sim->metricsToCompute, choi, sim->chname, consts);
	// PrintComplexArray2D(choi, "Physical channel", nlogs, nlogs);
	// if (IsState(choi) == 1)
	// 	printf("_/ is a quantum state.\n");
	// else
	// 	printf("X not a quantum state.\n");
	for (i = 0; i < nlogs; i ++)
		free(choi[i]);
	free(choi);
	// PrintDoubleArray1D((sim->metricValues)[0], "Level-0 metrics", sim->nmetrics);
}

void ComputeLevelOneChannels(struct simul_t *sim, struct qecc_t *qcode, struct constants_t *consts){
	// Compute the effective channels and syndrome probabilities for all level-1 syndromes.
	// The pre-computation is to avoid re-computing level-1 syndromes for every new top-level syndrome.
	// Load the physical channels on to the simulation structure and perform qcode
	// printf("Function: ComputeLevelOneChannels\n");
	int q, i, j, isPauli = 1;
	AllocSimParamsQECC(sim, qcode->N, qcode->K);
	for (q = 0; q < qcode->N; q ++){
		for (i = 0; i < qcode->nlogs; i ++)
			for (j = 0; j < qcode->nlogs; j ++)
				(sim->virtchan)[q][i][j] = (sim->physical)[i][j];
		if (isPauli > 0)
			isPauli = isPauli * IsDiagonal((sim->virtchan)[q], qcode->nlogs);
		// PrintDoubleArray2D((sim->virtchan)[q], "Level 0 channel", qcode->nlogs, qcode->nlogs);
	}
	// printf("Loaded virtual channels, isPauli = %d.\n", isPauli);
	SingleShotErrorCorrection(isPauli, (sim->frames)[0], qcode, sim, consts);
	// printf("Completed SingleShotErrorCorrection.\n");
	UpdateMetrics(0, 1, 1, 0, qcode, sim, consts);
	int s;
	for (s = 0; s < qcode->nstabs; s ++){
		(sim->levelOneSynds)[s] = (sim->syndprobs)[s];
		for (i = 0; i < qcode->nlogs; i ++)
			for (j = 0; j < qcode->nlogs; j ++)
				(sim->levelOneChannels)[s][i][j] = (sim->effprocess)[s][i][j];
		// printf("s = %d:\n", s);
		// PrintDoubleArray2D((sim->levelOneChannels)[s], "Level 1 channel", qcode->nlogs, qcode->nlogs);
	}
	ConstructCumulative(sim->levelOneSynds, sim->levelOneCumul, qcode->nstabs);
	// Compute the importance distribution for level-1 if necessary.
	double *searchin = malloc(sizeof(double) * 2);
	if (sim->importance == 1){
		searchin[0] = 0;
		searchin[1] = 1;
		double expo = PowerSearch(sim->syndprobs, qcode->nstabs, sim->outlierprobs, searchin);
		ConstructImportanceDistribution(sim->syndprobs, sim->levelOneImpDist, qcode->nstabs, expo);
		ConstructCumulative(sim->levelOneImpDist, sim->levelOneImpCumul, qcode->nstabs);
	}
	free(searchin);
	// Running averages have no meaning since we compute the exact average for level 1. However, we'll set all the running averages to the exact one in this case.
	// int m;
	// for (m = 0; m < sim->nmetrics; m ++){
	// 	(sim->runavg)[m][0] = (double) (sim->nbreaks);
	// 	for (i = 0; i < sim->nbreaks; i ++){
	// 		(sim->runstats)[i] = 1;
	// 		(sim->runavg)[m][i + 1] = (sim->metricValues)[1][m];
	// 	}
	// }
	FreeSimParamsQECC(sim, qcode->N, qcode->K);
}

void Coarsegrain(int level, struct simul_t **sims, double *****channels, int nchans, int nlogs){
	// We would like to average over some of the channels in the given level.
	// This averaging provides a coarse-grained information on the different channels in the level.
	// The decbins contains a number for every channel in the level, such that the number for those channels which must be averaged are identical.
	// printf("Function: Coarsegrain(%d, sims, channels, %d, %d)\n", level, nchans, nlogs);
	int b, i, j, s, c;
	int *binsizes;
	double ***avgchans;
	for (s = 0; s < 1 + (int)(sims[0]->importance == 2); s ++){
		// printf("sims[%d]->hybrid = %d, sims[%d]->importance = %d\n", s, sims[s]->hybrid, s, sims[s]->importance);
		if (sims[s]->hybrid > 0){
			// printf("Coarse graining process with %d bins.\n", (sims[s]->ndecbins)[level]);
			// PrintIntArray1D((sims[s]->decbins)[level], "decoder bins", nchans);
			binsizes = malloc(sizeof(int) * (sims[s]->ndecbins)[level]);
			avgchans = malloc(sizeof(double **) * (sims[s]->ndecbins)[level]);
			for (b = 0; b < (sims[s]->ndecbins)[level]; b ++){
				binsizes[b] = 0;
				avgchans[b] = malloc(sizeof(double *) * nlogs);
				for (i = 0; i < nlogs; i ++){
					avgchans[b][i] = malloc(sizeof(double) * nlogs);
					for(j = 0; j < nlogs; j ++)
						avgchans[b][i][j] = 0;
				}
			}
			for (c  = 0; c < nchans; c ++){
				for (i = 0; i < nlogs; i ++)
					for (j = 0; j < nlogs; j ++)
						avgchans[(sims[s]->decbins)[level][c]][i][j] += channels[level][c][s][i][j];
				binsizes[(sims[s]->decbins)[level][c]] ++;
			}
			for (c = 0; c < nchans; c ++)
				for (i = 0; i < nlogs; i ++)
					for (j = 0; j < nlogs; j ++)
						channels[level][c][s][i][j] = avgchans[(sims[s]->decbins)[level][c]][i][j]/((double) binsizes[(sims[s]->decbins)[level][c]]);
			// Free memory
			free(binsizes);
			for (b = 0; b < (sims[0]->ndecbins)[level]; b ++){
				for (i = 0; i < nlogs; i ++)
					free(avgchans[b][i]);
				free(avgchans[b]);
			}
			free(avgchans);
		}
	}
	// printf("Completed coarse-graining.\n");
}


void ComputeLogicalChannels(struct simul_t **sims, struct qecc_t **qcode, struct constants_t *consts, double *****channels){
	// Compute a logical channel for the required concatenation level.
	// The logical channel at a concatenated level l depends on N channels from the previous concatenation level, and so on... until 7^l physical channels.
	// printf("Function: ComputeLogicalChannels\n");
	int *nphys = malloc(sizeof(int) * sims[0]->nlevels);
	int l;
	for (l = 0; l < sims[0]->nlevels; l ++)
		nphys[l] = qcode[l]->N;
	// PrintIntArray1D(nphys, "nphys", sims[0]->nlevels);
	int *chans = malloc(sizeof(int) * (sims[0]->nlevels + 1));
	CountIndepLogicalChannels(chans, nphys, sims[0]->nlevels);
	// PrintIntArray1D(chans, "chans", sims[0]->nlevels + 1);

	// At every level, select a set of n channels, consider them as physical channels and perform qcode to output a logical channel.
	// Place this logical channel in the channels array, at the succeeding level.
	// To start with, we will only initialize the last level with samples of the level-1 channels.
	int s, b, i, j, q, randsynd;
	double bias, history, expo;
	double *searchin = malloc(sizeof(double) * 2);
	double *impdist = malloc(sizeof(double) * qcode[0]->nstabs);
	double *impcumul = malloc(sizeof(double) * qcode[0]->nstabs);
	double ****inputchannels = malloc(sizeof(double ***));
	int *isPauli = malloc(sizeof(int) * (int)(1 + (int)(sims[0]->importance == 2)));

	// printf("Computing logical channels for %d levels.\n", sims[0]->nlevels);

	for (l = 1; l < sims[0]->nlevels; l ++){
		// Allocate memory for the simulation parameters which depend on the error correcting code
		for (s = 0; s < 1 + (int)(sims[0]->importance == 2); s ++)
			AllocSimParamsQECC(sims[s], qcode[l]->N, qcode[l]->K);

		// Allocate memory for inputchannels
		inputchannels = (double ****)realloc(inputchannels, sizeof(double ***) * qcode[l]->N);
		MemManageInputChannels(inputchannels, qcode[l]->N, qcode[l]->nlogs, sims[0]->importance, 0);

		for (b = 0; b < chans[l + 1]; b ++){
			// printf("batch = %d of %d\n", b, chans[l]);
			/*
			bias = 1;
			history = 1;
			for (q = 0; q < qcode[l]->N; q ++){
				// inputchannels[q] = {channels[l][n*b], ..., channels[l][n*(b+1)]}
				for (s = 0; s < 1 + (int)(sims[0]->importance == 2); s ++){
					for (i = 0; i < qcode[l]->nlogs; i ++)
						for (j = 0; j < qcode[l]->nlogs; j ++)
							inputchannels[q][s][i][j] = channels[l - 1][qcode[l]->N * b + q][s][i][j];
					inputchannels[q][s][qcode[l]->nlogs][0] = channels[l - 1][qcode[l]->N * b + q][s][qcode[l]->nlogs][0];
					inputchannels[q][s][qcode[l]->nlogs][1] = channels[l - 1][qcode[l]->N * b + q][s][qcode[l]->nlogs][1];
				}
				bias *= inputchannels[q][0][qcode[l]->nlogs][0];
				history *= inputchannels[q][0][qcode[l]->nlogs][1];
			}
			*/
			// Load the input channels on to the simulation structures and perform qcode.
			// Perform coarsegraining of logical channels
			Coarsegrain(l, sims, channels, chans[l], qcode[l]->nlogs);
			for (s = 0; s < 1 + (int)(sims[0]->importance == 2); s ++){
				bias = 1;
				history = 1;
				isPauli[s] = 1;
				for (q = 0; q < qcode[l]->N; q ++){
					for (i = 0; i < qcode[l]->nlogs; i ++)
						for (j = 0; j < qcode[l]->nlogs; j ++)
							(sims[s]->virtchan)[q][i][j] = channels[l - 1][qcode[l]->N * b + q][s][i][j];
					if (isPauli[s] > 0)
						isPauli[s] = isPauli[s] * IsDiagonal((sims[s]->virtchan)[q], qcode[l]->nlogs);
					// printf("qubit %d:\n", q);
					// PrintDoubleArray2D((sims[s]->virtchan)[q], "virtual channel", qcode[l]->nlogs, qcode[l]->nlogs);
					bias *= channels[l - 1][qcode[l]->N * b + q][s][qcode[l]->nlogs][0];
					history *= channels[l - 1][qcode[l]->N * b + q][s][qcode[l]->nlogs][1];
				}
				// printf("Going to perform SingleShotErrorCorrection on s = %d, isPauli = %d and frame = %d.\n", s, isPauli[s], (sims[s]->frames)[l]);
				SingleShotErrorCorrection(isPauli[s], (sims[s]->frames)[l], qcode[l], sims[s], consts);
			}
			// printf("bias = %g, history = %g.\n", bias, history);
			UpdateMetrics(l, bias, history, 0, qcode[l], sims[0], consts);

			if (l < (sims[0]->nlevels - 1)){
				if (sims[0]->importance == 0){
					randsynd = SampleCumulative(sims[0]->cumulative, qcode[l]->nstabs);
					// printf("Random syndrome = %d\n", randsynd);
					for (i = 0; i < qcode[l]->nlogs; i ++)
						for (j = 0; j < qcode[l]->nlogs; j ++)
							channels[l][b][0][i][j] = (sims[0]->effprocess)[randsynd][i][j];
					channels[l][b][0][qcode[l]->nlogs][0] = 1;
					channels[l][b][0][qcode[l]->nlogs][1] = history * sims[0]->syndprobs[randsynd];
					channels[l][b][0][qcode[l]->nlogs][2] = sims[0]->syndprobs[randsynd];
				}
				else if (sims[0]->importance == 1){
					// Compute a probability distribution where the probability of every syndrome is given by a power of the original syndrome distribution.
					// The new distribution Q(s) is given by Eq. 6 of the article.
					// Sample a syndrome according to Q(s) and add a bias P(s)/Q(s).
					searchin[0] = 0;
					searchin[1] = 1;
					expo = PowerSearch(sims[0]->syndprobs, qcode[l]->nstabs, sims[0]->outlierprobs, searchin);
					ConstructImportanceDistribution(sims[0]->syndprobs, impdist, qcode[l]->nstabs, expo);
					ConstructCumulative(impdist, impcumul, qcode[l]->nstabs);
					randsynd = SampleCumulative(impcumul, qcode[l]->nstabs);
					// printf("randsynd = %d\n", randsynd);
					for (i = 0; i < qcode[l]->nlogs; i ++)
						for (j = 0; j < qcode[l]->nlogs; j ++)
							channels[l][b][0][i][j] = (sims[0]->effprocess)[randsynd][i][j];
					// printf("Populated channels.\n");
					channels[l][b][0][qcode[l]->nlogs][0] = (sims[0]->syndprobs)[randsynd]/impdist[randsynd];
					channels[l][b][0][qcode[l]->nlogs][1] = history * (sims[0]->syndprobs)[randsynd];
					channels[l][b][0][qcode[l]->nlogs][2] = (sims[0]->syndprobs)[randsynd];
				}
				else if (sims[0]->importance == 2){
					// Draw two logical channels.
					// 1. Noisy (reference) channel simulation itself.
					randsynd = SampleCumulative(sims[1]->cumulative, qcode[l]->nstabs);
					for (i = 0; i < qcode[l]->nlogs; i ++)
						for (j = 0; j < qcode[l]->nlogs; j ++)
							channels[l][b][1][i][j] = sims[1]->effprocess[randsynd][i][j];
					channels[l][b][1][qcode[l]->nlogs][0] = 1;
					channels[l][b][1][qcode[l]->nlogs][1] = 1;
					channels[l][b][1][qcode[l]->nlogs][2] = (sims[1]->syndprobs)[randsynd];
					// 2. Drawing syndromes for the original channel according to the noisy channel syndrome distribution.
					for (i = 0; i < qcode[l]->nlogs; i ++)
						for (j = 0; j < qcode[l]->nlogs; j ++)
							channels[l][b][0][i][j] = (sims[0]->effprocess)[randsynd][i][j];
					channels[l][b][0][qcode[l]->nlogs][0] = bias * sims[0]->syndprobs[randsynd]/((sims[1]->syndprobs)[randsynd]);
					channels[l][b][0][qcode[l]->nlogs][1] = history * ((sims[0]->syndprobs)[randsynd]);
					channels[l][b][0][qcode[l]->nlogs][2] = ((sims[0]->syndprobs)[randsynd]);
				}
				else
					continue;
			}
		}
		// Free memory for inputchannels
		MemManageInputChannels(inputchannels, qcode[l]->N, qcode[l]->nlogs, sims[0]->importance, 1);
		// Free simulation parameters that depend on the qcode
		for (s = 0; s < 1 + (int)(sims[0]->importance == 2); s ++)
			FreeSimParamsQECC(sims[s], qcode[l]->N, qcode[l]->K);
	}
	// printf("Freeing all memory.\n");
	// Free memory
	free(searchin);
	free(isPauli);
	free(impdist);
	free(impcumul);
	free(chans);
	free(nphys);
	// printf("Freeing inputchannels.\n");
	// for (q = 0; q < qcode[sims[0]->nlevels - 1]->N; q ++)
	// 	free(inputchannels[q]);
	free(inputchannels);
}

void Performance(struct qecc_t **qcode, struct simul_t **sims, struct constants_t *consts){
	// Compute logical error rates for a concatenation level.
	init_genrand(time(NULL)); // See the random number generator
	/*
		channels = list of l arrays.
			channels[i] = list of list of 7^(l-i-1) arrays.
				channels[i][b] = list of 3 arrays -- each corresponding to one type of sampling method. If importance sampling is turned off, there is only one array in the list.
					channels[i][b][s] = 4x4 matrix denoting a level-i channel.
	*/
	int *nphys = malloc(sims[0]->nlevels * sizeof(int));
	int *nencs = malloc(sims[0]->nlevels * sizeof(int));
	int l;
	for (l = 0; l < sims[0]->nlevels; l ++){
		nphys[l] = qcode[l]->N;
		nencs[l] = qcode[l]->K;
	}
	// printf("Allocate memory to channels.\n");
	double *****channels = malloc(sizeof(double ****) * (sims[0]->nlevels));;
	int nchans = MemManageChannels(channels, nphys, nencs, sims[0]->nlevels, sims[0]->importance, 0);
	// Compute level-0 metrics and level-1 effective channels and syndromes.
	ComputeLevelZeroMetrics(sims[0], qcode[0]->nlogs, consts);
	// Compute level-1 effective channels and syndromes.
	int s;
	for (s = 0; s < 1 + (int) (sims[0]->importance == 2); s ++)
		ComputeLevelOneChannels(sims[s], qcode[0], consts);
	// printf("Finished level-1 computations with %d channels.\n", (nchans));
	// PrintDoubleArray2D((sims[0]->logical)[1], "Logical channel", qcode[0]->nlogs, qcode[0]->nlogs);

	int c, i, j, m, randsynd;
	long t;
	if (sims[0]->nlevels > 1){
		for (t = 0; t < sims[0]->nstats; t ++){
			// Fill the lowest level of the channels array with "nchans" samples of level-1 channels.
			printf("Stat %ld, nchans = %d.\n", t, nchans);
			for (c = 0; c < nchans; c ++){
				if (sims[0]->importance == 0){
					// Direct sampling
					randsynd = SampleCumulative(sims[0]->levelOneCumul, qcode[0]->nstabs);
					// printf("Random syndrome = %d, with probability: %g.\n", randsynd, (sims[0]->levelOneSynds)[randsynd]);
					for (i = 0; i < qcode[0]->nlogs; i ++)
						for (j = 0; j < qcode[0]->nlogs; j ++)
							channels[0][c][0][i][j] = (sims[0]->levelOneChannels)[randsynd][i][j];
					channels[0][c][0][qcode[0]->nlogs][0] = 1.0;
					channels[0][c][0][qcode[0]->nlogs][1] = (sims[0]->levelOneSynds)[randsynd];
					channels[0][c][0][qcode[0]->nlogs][2] = (sims[0]->levelOneSynds)[randsynd];
				}
				else if (sims[0]->importance == 1){
					// Draw a syndrome from the importance distribution specified by the power-law scaling.
					randsynd = SampleCumulative(sims[0]->levelOneImpCumul, qcode[0]->nstabs);
					for (i = 0; i < qcode[0]->nlogs; i ++)
						for (j = 0; j < qcode[0]->nlogs; j ++)
							channels[0][c][0][i][j] = (sims[0]->levelOneChannels)[randsynd][i][j];
					channels[0][c][0][qcode[0]->nlogs][0] = (sims[0]->levelOneSynds)[randsynd]/(sims[0]->levelOneImpDist)[randsynd];
					channels[0][c][0][qcode[0]->nlogs][1] = (sims[0]->levelOneSynds)[randsynd];
					channels[0][c][0][qcode[0]->nlogs][2] = (sims[0]->levelOneSynds)[randsynd];
				}
				else if (sims[0]->importance == 2){
					// Draw a syndrome from the nosier channel syndrome distribution.
					randsynd = SampleCumulative(sims[1]->levelOneCumul, qcode[0]->nstabs);
					for (i = 0; i < qcode[0]->nlogs; i ++)
						for (j = 0; j < qcode[0]->nlogs; j ++)
							for (s = 0; s < 1 + (int)(sims[0]->importance == 2); s ++)
								channels[0][c][s][i][j] = (sims[s]->levelOneChannels)[randsynd][i][j];
					channels[0][c][0][qcode[0]->nlogs][0] = (sims[0]->levelOneImpDist)[randsynd]/(sims[1]->levelOneSynds)[randsynd];
					channels[0][c][0][qcode[0]->nlogs][1] = (sims[0]->levelOneSynds)[randsynd];
					channels[0][c][0][qcode[0]->nlogs][2] = (sims[0]->levelOneSynds)[randsynd];

					channels[0][c][1][qcode[0]->nlogs][0] = 1;
					channels[0][c][1][qcode[0]->nlogs][1] = 1;
					channels[0][c][1][qcode[0]->nlogs][2] = (sims[1]->levelOneSynds)[randsynd];
				}
				else
					continue;
			}
			
			// Compute average logical channels and average logical error rates.
			ComputeLogicalChannels(sims, qcode, consts, channels);
			for (s = 0; s < 1 + (int)(sims[0]->importance == 2); s ++){
				if (IsElement(sims[s]->runstats, sims[s]->nbreaks, t + 1) == 1){
					for (m = 0; m < sims[0]->nmetrics; m ++){
						(sims[s]->runavg)[m][0] += 1;
						(sims[s]->runavg)[m][(int)(sims[s]->runavg)[m][0]] = (sims[s]->metricValues)[sims[s]->nlevels][m]/((double) (t + 1));
					}
				}
			}
		}
	}
	
	// Normalize the average metrics.
	for (l = 1; l < sims[0]->nlevels; l ++)
		UpdateMetrics(l, 1.0, 1.0, 1, qcode[l], sims[0], consts);

	// printf("Updated metrics\n");
	// PrintDoubleArray2D((sims[0]->logical)[1], "logical level-1 channel", qcode[0]->nlogs, qcode[0]->nlogs);
	// PrintDoubleArray1D((sims[0]->metricValues)[1], "logical level-1 metrics", sims[0]->nmetrics);
	// PrintDoubleArray2D((sims[0]->logical)[1], "Logical channel", qcode[0]->nlogs, qcode[0]->nlogs);
	
	// Free memory for channels.
	MemManageChannels(channels, nphys, nencs, sims[0]->nlevels, sims[0]->importance, 1);
	free(channels);
	free(nphys);
	free(nencs);

	// printf("Freed channels\n");
	// printf("Done Performance.\n");
}
