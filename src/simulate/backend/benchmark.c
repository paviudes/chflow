#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include "printfuns.h"
#include "constants.h"
#include "memory.h"
#include "qecc.h"
#include "effective.h"
#include "benchmark.h"
// #include "logmetrics.h" // only for testing


void InitBenchOut(struct BenchOut *pbout, int nlevels, int nmetrics, int nlogs, int nbins, int nbreaks){
	// Initialize the memory allocated the to the benchmark output.
	pbout->logchans = malloc(sizeof(double) * (nlevels + 1) * nlogs * nlogs);
	pbout->chanvar = malloc(sizeof(double) * (nlevels + 1) * nlogs * nlogs);
	pbout->logerrs = malloc(sizeof(double) * (nlevels + 1) * nmetrics);
	pbout->logvars = malloc(sizeof(double) * (nlevels + 1) * nmetrics);
	pbout->bins = malloc(sizeof(int) * nmetrics * (nlevels + 1) * nbins * nbins);
	pbout->running = malloc(sizeof(double) * nmetrics * nbreaks);
}

void FreeBenchOut(struct BenchOut *pbout){
	// free the memory allocated to the Benchmark output.
	free(pbout->logchans);
	free(pbout->chanvar);
	free(pbout->logerrs);
	free(pbout->logvars);
	free(pbout->bins);
	free(pbout->running);
}

struct BenchOut Benchmark(int nlevels, int *nkd, int *SS, int *normalizer, double *normphases_real, double *normphases_imag, const char *chname, double *physical, int nmetrics, char **metrics, int decoder, int frame, int nbreaks, long *stats, int nbins, int maxbin, int importance, double *refchan){
	/*
	Benchmark an error correcting scheme.
	Inputs:
		(a). Specifications of the error correcting code
			1. number of concatenation layers: int levels
			2. N,K,D for each code
				int **nkd : 2D array of integers
					where the i-th row gives the N,K,D values of the i-th code.
			3. Stabilizer syndrome signs for each code
				int ***SS: 3D array of integers
					where the i-th array gives the stabilizer syndrome signs for the i-th code.
			4. Logical action each code
				int ****normalizer: 4D array of integers
					where the i-th array gives the logical action for the i-th code.
			5. Logical action phases for each code.
				int ***normphases: 3D array where the i-th array gives the phases for the i-th code.
		(b). Error channel
			1. Channel name: char *chname
			2. Physical noise map (as a process matrix)
				double **physical: 2D double array
		(c). Metrics
			1. Number of metrics to be computed: int nmetrics
			2. Names of the metrics to be computed: char *metrics.
		(d). Specifications of the simulation
			1. Type of syndrome sampling
				int importance: string specifying the sampling algorithm.
			2. Number of syndromes to be sampled at the top level, with breakpoints.
				long *stats: array of longs, where the i-th element gives the i-th breakpoint for the running average.
			3. Number of syndrome metric bins: int nbins
			4. Maximum order of magnitude for a bin: int maxbin
			5. Quantum error correction frame: int frame
			6. Decoding technique, hard or soft: int decoder.
	*/
	struct constants_t *consts = malloc(sizeof(struct constants_t));
	InitConstants(consts);
	// Initialize the error correcting code structure.
	// printf("Quantum error correcting code: %d levels\n", nlevels);
	struct qecc_t **qcode = malloc(sizeof(struct qecc_t*) * nlevels);
	int l, s, g, i, q, stabcount = 0, normcount = 0, norm_phcount = 0;
	for (l = 0; l < nlevels; l ++){
		// printf("l = %d\n", l);
		qcode[l] = malloc(sizeof(struct qecc_t));
		qcode[l]->N = nkd[3 * l];
		qcode[l]->K = nkd[3 * l + 1];
		qcode[l]->D = nkd[3 * l + 2];
		InitQECC(qcode[l]);
		for (s = 0; s < qcode[l]->nstabs; s ++)
			for (g = 0; g < qcode[l]->nstabs; g ++)
				(qcode[l]->projector)[s][g] = SS[stabcount + s * qcode[l]->nstabs + g];
		stabcount += qcode[l]->nstabs * qcode[l]->nstabs;

		// printf("stabcount = %d\n", stabcount);

		for (i = 0; i < qcode[l]->nlogs; i ++)
			for (s = 0; s < qcode[l]->nstabs; s ++)
				for (q = 0; q < qcode[l]->N; q ++)
					(qcode[l]->action)[i][s][q] = normalizer[normcount + i * qcode[l]->nstabs * qcode[l]->N + s * qcode[l]->N + q];
		normcount += qcode[l]->nlogs * qcode[l]->nstabs * qcode[l]->N;

		// printf("normcount = %d\n", normcount);

		for (i = 0; i < qcode[l]->nlogs; i ++)
			for (s = 0; s < qcode[l]->nstabs; s ++)
				(qcode[l]->phases)[i][s] = normphases_real[norm_phcount + i * qcode[l]->nstabs + s] + I * normphases_imag[norm_phcount + i * qcode[l]->nstabs + s];
		norm_phcount += qcode[l]->nlogs * qcode[l]->nstabs;

		// printf("norm_phcount = %d\n", norm_phcount);
	}

	// printf("QECC assigned.\n");

	// Parameters that are specific to the Montecarlo simulations to estimate the logical error rate.
	struct simul_t **sims = malloc(sizeof(struct simul_t*) * (1 + (int)(importance == 2)));
	int m, j;
	for (s = 0; s < 1 + (int)(importance == 2); s ++){
		// printf("s = %d\n", s);
		sims[s] = malloc(sizeof(struct simul_t));
		sims[s]->nlevels = nlevels;
		sims[s]->nmetrics = nmetrics;
		sims[s]->importance = importance;
		sims[s]->decoder = decoder;
		sims[s]->nbins = nbins;
		sims[s]->maxbin = maxbin;
		sims[s]->nbreaks = nbreaks;
		sims[s]->nstats = stats[nbreaks - 1];

		// printf("Allocating simulation parameters for\ns = %d, nlevels = %d, nmetrics = %d, importance = %d, decoder = %d, nbins = %d, maxbin = %d, nstats = %ld, nbreaks = %d.\n", s, sims[s]->nlevels, sims[s]->nmetrics, sims[s]->importance, sims[s]->decoder, sims[s]->nbins, sims[s]->maxbin, sims[s]->nstats, sims[s]->nbreaks);

		AllocSimParams(sims[s], qcode[0]->N, qcode[0]->K);

		// Logical frame for Quantum error correction
		if (frame == 0)
			for (l = 0 ; l < nlevels; l ++)
				(sims[s]->frames)[l] = 4;
		else if (frame == 1)
			for (l = 0 ; l < nlevels; l ++)
				(sims[s]->frames)[l] = consts->nclifford;
		else{
			for (l = 0 ; l < nlevels - 1; l ++)
				(sims[s]->frames)[l] = 4;
			(sims[s]->frames)[nlevels - 1] = consts->nclifford;
		}

		// Error model and metrics
		sprintf(sims[s]->chname, "%s", chname);
		for (i = 0; i < qcode[0]->nlogs; i ++){
			for (j = 0; j < qcode[0]->nlogs; j ++){
				if (s == 0)
					(sims[s]->physical)[i][j] = physical[i * qcode[s]->nlogs + j];
				else
					(sims[s]->physical)[i][j] = refchan[i * qcode[s]->nlogs + j];
				(sims[s]->logical)[0][i][j] = (sims[s]->physical)[i][j];
			}
		}
		for (m = 0; m < nmetrics; m ++)
			sprintf((sims[s]->metricsToCompute)[m], "%s", metrics[m]);

		// Running average
		for (i = 0; i < nbreaks; i ++)
			(sims[s]->runstats)[i] = stats[i];

		// PrintDoubleArray1D(physical, "Physical Channel", qcode[0]->nlogs * qcode[0]->nlogs);
		// printf("Allocations complete for s = %d.\n", s);

	}

	// printf("Going to start Performance.\n");

	// ###################################

	Performance(qcode, sims, consts);

	// ###################################

	// printf("Loading outputs on to BenchOut.\n");

	int nlogs = qcode[0]->nlogs;
	struct BenchOut bout;
	InitBenchOut(&bout, sims[0]->nlevels, sims[0]->nmetrics, nlogs, sims[0]->nbins, (int) stats[0]);

	// Benchmark output
	// 1. Average effective channels for each concatenation level.
	// 2. Variance in the average effective channel for each concatenation level.
	// 3. For each metric
		// (a). Logical metric values for each concatenation level.
		// (b). Variance of logical error metric values for each concatenation level.
		// (c). Counts binned according to the syndrome probability and conditioned logical error metric.
	// 4. Running average of average metric values for topmost level.

	// printf("Assigning output values.\n");

	for (l = 0; l < nlevels + 1; l ++){
		// printf("l = %d\n", l);
		// PrintDoubleArray2D((sims[0]->logical)[l], "logical channel", nlogs, nlogs);
		for (i = 0; i < nlogs; i ++){
			for (j = 0; j < nlogs; j ++){
				(bout.logchans)[l * (int) pow((double) nlogs, 2) + i * nlogs + j] = (sims[0]->logical)[l][i][j];
				(bout.chanvar)[l * (int) pow((double) nlogs, 2) + i * nlogs + j] = (sims[0]->variance)[l][sims[0]->nmetrics + i * nlogs + j];
			}
		}
	}
	// PrintDoubleArray1D((bout.logchans), "Logical channels", (nlevels + 1) * nlogs * nlogs);

	for (m = 0; m < nmetrics; m ++){
		for (l = 0; l < nlevels + 1; l ++){
			(bout.logerrs)[m * (nlevels + 1) + l] = (sims[0]->metricValues)[l][m];
			(bout.logvars)[m * (nlevels + 1) + l] = (sims[0]->variance)[l][m];
			for (i = 0; i < nbins; i ++)
				for (j = 0; j < nbins; j ++)
					(bout.bins)[m * (nlevels + 1) * (int) pow((double) nbins, 2) + l * (int) pow((double) nbins, 2) + i * nbins + j] = (sims[0]->bins)[l][m][i][j];
		}
		for (i = 0; i < nbreaks; i ++)
			(bout.running)[m * nbreaks + i] = (sims[0]->runavg)[m][i + 1];
	}
	// PrintDoubleArray1D((bout.logerrs), "Metric values", nmetrics * (nlevels + 1));
	// PrintDoubleArray1D((bout.running), "Running average", nmetrics * nbreaks);

	// ######################################
	// printf("Freeing memory.\n");

	// Free memory
	// printf("Freeing %d simulation structures.\n", 1 + (int)(importance == 2));
	for (s = 0; s < 1 + (int)(importance == 2); s ++){
		FreeSimParams(sims[s], qcode[0]->N, qcode[0]->K);
		free(sims[s]);
	}
	free(sims);
	// printf("Freeing %d qcode structures.\n", nlevels);
	for (l = 0; l < nlevels; l ++){
		FreeQECC(qcode[l]);
		free(qcode[l]);
	}
	free(qcode);
	FreeConstants(consts);

	// printf("Benchmark complete!\n");
	return bout;
}
