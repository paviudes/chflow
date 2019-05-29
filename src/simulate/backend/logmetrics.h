#ifndef LOGMETRICS_H
#define LOGMETRICS_H

#include <complex.h>

/*
	Compute all the metrics for a given channel in the Choi matrix form.
	Inputs:
		double *metvals = double array of shape (nmetrics), which will contain the metric values after function execution.
		int nmetrics = number of metric values to be computed.
		char **metricsToCompute = array of strings containing the names of the metrics to be computed.
		double complex **choi = array of shape (4 x 4) containing the Choi matrix of the channel whose metrics need to be computed.
		char *chname = name of the channel.
		struct constants_t *consts = Pointer to the constants structure, to access Pauli matrices.
*/
extern void ComputeMetrics(double *metvals, int nmetrics, char **metricsToCompute, double complex **choi, char *chname, struct constants_t *consts);

#endif /* LOGMETRICS_H */