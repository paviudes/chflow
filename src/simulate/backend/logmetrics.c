#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include "linalg.h"
#include "constants.h"
#include "logmetrics.h"

double Entropy(double complex **choi){
	// Compute the Von-Neumann entropy of the Choi matrix.
	// If the channel is unitary, its entropy will be zero, as the choi matrix will correspond to a pure state.
	// If the channel causes decoherence, its choi matrix will be a mixed state with non-zero entropy.
	double complex *eigvals = malloc(sizeof(double complex) * 4);
	Diagonalize(choi, 4, eigvals, 0, NULL);
	int i;
	double entropy = 0;
	for (i = 0; i < 4; i ++)
		entropy -= creal(eigvals[i]) * log(creal(eigvals[i]));
	// Free memory
	free(eigvals);
	return entropy;
}

double TraceDistance(double complex **choi){
	// Compute the trace distance between the Choi matrix of a channel and that of an identity channel: ||E - id||_1.
	// The ||.||_1 of a matrix is just the sum of the absolute values of the eigenvalues.
	double complex **chandiff = malloc(sizeof(double complex) * 4);
	int i, j;
	for (i = 0; i < 4; i ++){
		chandiff[i] = malloc(sizeof(double complex) * 4);
		for (j = 0; j < 4; j ++){
			chandiff[i][j] = choi[i][j];
		}
	}
	// Note that the Choi matrix for the identity channel is a (4 x 4) whose four corners are 0.5.
	chandiff[0][0] -= 0.5; chandiff[0][3] -= 0.5;
	chandiff[3][0] -= 0.5; chandiff[3][3] -= 0.5;
	// Compute the sigular values of E - id.
	double complex *singvals = malloc(sizeof(double complex) * 4);
	Diagonalize(chandiff, 4, singvals, 0, NULL);
	double trn = 0;
	for (i = 0; i < 4; i ++)
		trn += cabs(singvals[i]);
	// Free memory
	for (i = 0; i < 4; i ++)
		free(chandiff[i]);
	free(chandiff);
	free(singvals);
	return trn;
}

double Infidelity(double complex **choi){
	// Compute the Infidelity between the input Choi matrix and the Choi matrix corresponding to the identity state.
	// Choi matrix for the identity is: 0.5 * [[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]].
	// Returns 1-fidelity.
	double infidelity = 1 - 0.5 * (creal(choi[0][0] + choi[3][0] + choi[0][3] + choi[3][3]));
	return infidelity;
}

double ProcessFidelity(double complex **choi){
	// Compute the average infidelity, given by: 1/6 * (4 - Tr(N)) where N is the process matrix describing a noise channel.
	// The above expression for infidelity can be simplified to 2/3 * entanglement infidelity.
	return (2/((double)3) * Infidelity(choi));
}

double FrobeniousNorm(double complex **choi){
	// Compute the Frobenious norm of the difference between the input Choi matrix and the Choi matrix corresponding to the Identity channel.
	// Frobenious of A is defined as: sqrtl(Trace(A^\dagger . A)).
	// https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm .
	int i, j;
	double frobenious = 0;
	for (i = 0 ; i < 4; i ++)
		for (j = 0 ; j < 4; j ++)
			frobenious = frobenious + cabs(choi[i][j]) * cabs(choi[i][j]);
	frobenious = frobenious + 1 - (creal(choi[0][0] + choi[3][0] + choi[0][3] + choi[3][3]));
	frobenious = sqrt(fabs(frobenious));
	return frobenious;
}


void ChoiToChi(double complex **choi, double complex **chi, struct constants_t *consts){
	// Convert from the Choi to the Chi representation.
	// Use the idea in ConvertRepresentations.
	// printf("function: ChoiToChi\n");
	int i, j;
	for (i = 0; i < 16; i ++)
		for (j = 0; j < 16; j ++)
			chi[j/4][j%4] += choi[i/4][i%4] * (consts->choitochi)[i][j];
}

double NonPauliness(double complex **choi, struct constants_t *consts){
	// Quantify the behaviour of a quantum channel by its difference from a Pauli channel.
	// Convert the input Choi matrix to it's Chi-representation.
	// Compute the ration between the  sum of offdiagonal entries to the sum of disgonal entries.
	// While computing the sums, consider the absolution values of the entries.
	// printf("function: NonPauliness\n");
	int i, j;
	double complex **chi = malloc(sizeof(double complex *) * 4);
	for (i = 0 ; i < 4; i ++){
		chi[i] = malloc(sizeof(double complex) * 4);
		for (j = 0 ; j < 4; j ++)
			chi[i][j] = 0.0;
	}
	ChoiToChi(choi, chi, consts);
	double nonpauli = 0.0, atol = 10E-20;
	for (i = 0 ; i < 4; i ++){
		for (j = 0 ; j < 4; j ++){
			if (i != j){
				if (cabs(chi[i][i]) * cabs(chi[j][j]) >= atol){
					nonpauli = nonpauli + cabs(chi[i][j]) * cabs(chi[i][j])/(cabs(chi[i][i]) * cabs(chi[j][j]));
				}
			}
		}
	}
	// Free memory
	for (i = 0 ; i < 4; i ++)
		free(chi[i]);
	free(chi);
	return nonpauli;
}

void ComputeMetrics(double *metvals, int nmetrics, char **metricsToCompute, double complex **choi, char *chname, struct constants_t *consts){
	// Compute all the metrics for a given channel, in the Choi matrix form.
	// printf("Metrics for channel %s.\n", chname);
	int m;
	for (m = 0; m < nmetrics; m ++){
		if (strncmp(metricsToCompute[m], "frb", 3) == 0){
			metvals[m] = FrobeniousNorm(choi);
		}
		else if (strncmp(metricsToCompute[m], "infid", 5) == 0){
			metvals[m] = Infidelity(choi);
		}
		else if (strncmp(metricsToCompute[m], "processfidelity", 15) == 0){
			metvals[m] = ProcessFidelity(choi);
		}
		else if (strncmp(metricsToCompute[m], "np1", 3) == 0){
			metvals[m] = NonPauliness(choi, consts);
		}
		else if (strncmp(metricsToCompute[m], "entropy", 7) == 0){
			metvals[m] = Entropy(choi);
		}
		else if (strncmp(metricsToCompute[m], "trn", 3) == 0){
			metvals[m] = TraceDistance(choi);
		}
		else{
			// Metrics that are not optimized in C cannot be evaluated at the Logical levels. For these, the metric value is simply zero.
			printf("Metric %s for channel %s is not optimized in C.\n", metricsToCompute[m], chname);
			metvals[m] = 0;
		}
	}
}