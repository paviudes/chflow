#ifndef MEMORY_H
#define MEMORY_H

#include <complex.h>

struct simul_t{
	// Quantum error correction.
	char *chname;
	int decoder;
	double **physical;
	double ***virtchan;
	double ***logical;
	double ****process;
	double *syndprobs;
	double *cumulative;
	double *levelOneSynds;
	double *levelOneCumul;
	int *corrections;
	double complex ***effective;
	double ***effprocess;
	double ***levelOneChannels;
	int *frames;
	// Metrics.
	int nmetrics;
	char **metricsToCompute;
	double **metricValues;
	// Syndrome sampling.
	long nstats;
	int nlevels;
	int maxbin;
	int importance;
	double *levelOneImpDist;
	double *levelOneImpCumul;
	double outlierprobs[2];
	double **sampling;
	long *statsperlevel;
	int nbins;
	int ****bins;
	double **sumsq;
	double **variance;
	int nbreaks;
	long *runstats;
	double **runavg;
	double threshold;
};

// Initialize the elements that pertain to the montecarlo simulation of channels.
extern void AllocSimParams(struct simul_t *simul, int nphys, int nenc);

// Free memory allocated to the elements of the simulation structure that do not depend on the QECC.
extern void FreeSimParams(struct simul_t *simul, int nphys, int nenc);

/*
	Allocate memory for parameters of the simulation structure that depend upon the QECC used.
	This memeory will be reallocated everytime there is a new QECC, i.e, at a new concatenation level.
	These parameters are
	virtual, logical, syndprobs, cumulative, levelZeroSynds, levelZeroCumulative, levelZeroImpDist, levelZeroImpCumul, process, corrections, effprocess, effective, levelZeroChannels
*/
extern void AllocSimParamsQECC(struct simul_t *simul, int nphys, int nenc);

// Free the memory allocated to simulation parameters that depend on the QECC.
extern void FreeSimParamsQECC(struct simul_t *simul, int nphys, int nenc);

// Determine the number of independent logical channels at every level, that determine the logical channels of higher levels.
extern int CountIndepLogicalChannels(int *chans, int *nphys, int nlevels);

// Allocate and Free memory for the tree of lower-level channels which determine a logical channel.
extern int MemManageChannels(double *****channels, int *nphys, int *nencs, int nlevels, int importance, int tofree);

// Allocate and free memory for the input channels structure in ComputeLogicalChannels(...).
extern void MemManageInputChannels(double ****inputchannels, int nphys, int nlogs, int importance, int tofree);

#endif /* MEMORY_H */