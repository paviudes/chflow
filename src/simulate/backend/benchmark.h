#ifndef BENCHMARK_H
#define BENCHMARK_H

struct BenchOut{
	/*
	Output parameters of a benchmarking process.
	1. Average effective channels for each concatenation level.
		logchans = 3D array of shape: nlevels x 4^K x 4^K
	2. Variance in the average effective channel for each concatenation level.
		chanvar = 3D array of shape: nlevels x 4^K x 4^K
	3. For each metric
		(a). Logical metric values for each concatenation level.
			logerrs = 2D array of shape:nmetrics x nlevels
		(b). Variance of logical error metric values for each concatenation level.
			logvars = 2D array of shape:nmetrics x nlevels
		(c). Counts binned according to the syndrome probability and conditioned logical error metric.
			bins = 4D array of shape:nmetrics x nlevels x nbins x nbins
	4. Running average of average metric values for topmost level.
			running = 1D array of shape: nbreaks

	All the arrays are vectorized.
	*/
	double *logchans;
	double *chanvar;
	double *logerrs;
	double *logvars;
	int *bins;
	double *running;
};

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
		(e). Decoding
			1. Decoding technique, soft or hybrid: int hybrid
			2. Channels that must be averaged at intermediate levels: int **decoderbins
			3. Number of distinct (bins) of channels at intermediate levels: int *ndecoderbins

	All the multi-dimensional arrays are (row) vectorized.
*/
extern struct BenchOut Benchmark(int nlevels, int *nkd, int *SS, int *normalizer, double *normphases_real, double *normphases_imag, const char *chname, double *physical, int nmetrics, char **metrics, int hybrid, int *decoderbins, int *ndecoderbins, int frame, int nbreaks, long *stats, int nbins, int maxbin, int importance, double *refchan);

#endif /* BENCHMARK_H */
