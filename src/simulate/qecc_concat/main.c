#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include "mt19937/mt19937ar.h" // Random number generator
#include "printfuns.h"
#include "linalg.h"
#include "constants.h"
#include "memory.h"
#include "qecc.h"
#include "sampling.h"
#include "checks.h"
#include "logmetrics.h"

int main(int argc, char **argv)
{
	/* 
	This function is simply to test all the C functions in the converted/ folder.
	*/
	if (argc < 2){
		printf("Usage: ./bmark <file name> <function name>\n");
		return 0;
	}
	int i, j, k;
	double complex **mat = malloc(sizeof(double complex) * 4);
	for (i = 0; i < 4; i ++)
		mat[i] = malloc(sizeof(double complex) * 4);
	/* Creating a random number generator.
		See https://stackoverflow.com/questions/822323/how-to-generate-a-random-int-in-c on why not to use the in-built rand() function.
		Instead, we use the Mersenne Twister random number generator explained in http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html .
		The source code can be found in mt19937/ folder.
	*/
	init_genrand(time(NULL));
	// The file and the function to be tested are obtained from shell.
	char *file = malloc(sizeof(char) * 100);
	sprintf(file, "%s", argv[1]);
	char *func = malloc(sizeof(char) * 100);
	sprintf(func, "%s", argv[2]);

	if (strncmp(file, "sampling", 8) == 0){
		// Testing the functions in samplig.c.
		printf("File: samplig.c.\n");
		if (strncmp(func, "ConstructCumulative", 19) == 0){
			// Testing ConstructCumulative.
			printf("Testing ConstructCumulative.\n");
			int size = 10;
			// Creating a test cumulative distribution.
			double *dist = malloc(sizeof(double) * size);
			int i;
			double sum = 0;
			for (i = 0; i < size; i ++){
				dist[i] = genrand_real3();
				sum += dist[i];
			}
			for (i = 0; i < size; i ++)
				dist[i] = dist[i]/sum;
			PrintDoubleArray1D(dist, "True Distribution", size);
			double *cumul = malloc(sizeof(double) * size);
			ConstructCumulative(dist, cumul, size);
			PrintDoubleArray1D(cumul, "Cumulative distribution", size);
			// free memory
			free(dist);
			free(cumul);
		}
		if (strncmp(func, "PowerSearch", 11) == 0){
			// Testing PowerSearch.
			printf("Function: PowerSearch.\n");
			int size = 10;
			// Creating a test cumulative distribution.
			double *dist = malloc(sizeof(double) * size);
			int i;
			dist[0] = genrand_real3();
			double sum = dist[0];
			for (i = 1; i < size; i ++){
				dist[i] = 0.01 * genrand_real3();
				sum += dist[i];
			}
			for (i = 0; i < size; i ++)
				dist[i] = dist[i]/sum;
			double window[2] = {0.45, 0.5};
			double searchin[2] = {0, 1};
			PrintDoubleArray1D(dist, "Probability distribution: P", size);
			printf("Searching for k in [%g, %g] such that (P^k)[0] > %g.\n", searchin[0], searchin[1], window[0]);
			double exponent = PowerSearch(dist, size, window, searchin);
			printf("k = %g.\n", exponent);
			double *powerdist = malloc(sizeof(double) * size);
			ConstructImportanceDistribution(dist, powerdist, size, exponent);
			PrintDoubleArray1D(powerdist, "New distribution", size);
			// free memory
			free(dist);
			free(powerdist);
		}
	}

	if (strncmp(file, "constants", 9) == 0){
		// Testing the functions in constants.c.
		if (strncmp(func, "InitConstants", 13) == 0){
			// Testing the function InitConstants. This doesn't output anything.
			struct constants_t *consts = malloc(sizeof(struct constants_t));
			InitConstants(consts);
			PrintComplexArray2D(consts->choitochi, "Choi to Chi transformation matrix", 16, 16);
			FreeConstants(consts);
			free(consts);
		}
	}

	if (strncmp(file, "checks", 5) == 0){
		// Testing functions in checks.c.
		double complex **mat = malloc(sizeof(double complex) * 4);
		for (i = 0; i < 4; i ++)
			for (j = 0; j < 4; j ++)
				mat[i][j] = genrand_real3() + genrand_real3() * I;
		if (strncmp(argv[1], "IsPositive", 10) == 0){
			printf("Function: IsPositive.\n");
			PrintComplexArray2D(mat, "Matrix", 4, 4);
			int ispos = IsPositive(mat);
			if (ispos == 1)
				printf("is positive.\n");
			else{
				printf("is not positive, but\n");
				double complex **matP = malloc(sizeof(double complex) * 4);
				for (i = 0; i < 4; i ++){
					matP[i] = malloc(sizeof(double complex) * 4);
					for (j = 0; j < 4; j ++){
						matP[i][j] = 0;
						for (k = 0; k < 4; k ++)
							matP[i][j] += mat[i][k] * conj(mat[j][k]);
					}
				}
				PrintComplexArray2D(matP, "M . M^\\dag", 4, 4);
				ispos = IsPositive(matP);
				if (ispos == 1)
					printf("is positive.\n");
				else
					printf("Error in function.\n");
				// Free memory
				for (i = 0; i < 4; i ++)
					free(matP[i]);
				free(matP);
			}
		}
		if (strncmp(argv[1], "IsHermitian", 11) == 0){
			printf("Function: IsHermitian.\n");
			PrintComplexArray2D(mat, "Matrix M", 4, 4);
			int isherm = IsHermitian(mat);
			if (isherm == 1)
				printf("is Hermitian.\n");
			else{
				printf("is not Hermitian, but\n");
				double complex **matH = malloc(sizeof(double complex) * 4);
				for (i = 0; i < 4; i ++){
					matH[i] = malloc(sizeof(double complex) * 4);
					for (j = 0; j < 4; j ++)
						matH[i][j] = mat[i][j] + conj(mat[j][i]);
				}
				PrintComplexArray2D(matH, "M + M^\\dag", 4, 4);
				isherm = IsHermitian(matH);
				if (isherm == 1)
					printf("is Hermitian.\n");
				else
					printf("Error in function.\n");
				// Free memory
				for (i = 0; i < 4; i ++)
					free(matH[i]);
				free(matH);
			}
		}
		if (strncmp(argv[1], "IsTraceOne", 10) == 0){
			printf("Function: IsTraceOne.\n");
			PrintComplexArray2D(mat, "Matrix", 4, 4);
			int istrone = IsTraceOne(mat);
			if (istrone == 1)
				printf("has unit trace.\n");
			else{
				printf("does not have unit trace but\n");
				double trace = 0;
				for (i = 0; i < 4; i ++)
					trace = trace + creal(mat[i][i]);
				double complex **matN = malloc(sizeof(double complex) * 4);
				for (i = 0; i < 4; i ++){
					matN[i] = malloc(sizeof(double complex) * 4);
					for (j = 0; j < 4; j ++)
						matN[i][j] = mat[i][j]/trace;
					matN[i][i] = creal(matN[i][i]);
				}
				PrintComplexArray2D(matN, "M/trace", 4, 4);
				istrone = IsTraceOne(matN);
				if (istrone == 1)
					printf("has unit trace.\n");
				else
					printf("Error in function.\n");
				// Free memory
				for (i = 0; i < 4; i ++)
					free(matN[i]);
				free(matN);
			}
		}
		if (strncmp(argv[1], "IsState", 7) == 0){
			printf("Function: IsState.\n");
			PrintComplexArray2D(mat, "Matrix", 4, 4);
			int isstate = IsState(mat);
			if (isstate == 0)
				printf("is not a state.\n");
			else
				printf("is a state.\n");
		}
		if (strncmp(argv[1], "IsPDF", 5) == 0){
			double sum = 0;
			double *dist = malloc(sizeof(double) * 100);
			for (i = 0; i < 100; i ++){
				dist[i] = genrand_real3();
				sum += dist[i];
			}
			PrintDoubleArray1D(dist, "Un-normalized distribution", 100);
			int ispdf = IsPDF(dist, 100);
			printf("has ispdf = %d.\n", ispdf);
			for (i = 0; i < 100; i ++)
				dist[i] = dist[i]/sum;
			PrintDoubleArray1D(dist, "And after normalization", 100);
			ispdf = IsPDF(dist, 100);
			printf("it has ispdf = %d.\n", ispdf);
			// Free memory
			free(dist);
		}
	}

	if (strncmp(file, "logmetrics", 10) == 0){
		// Testing functions in logmetrics.c.
		if (strncmp(func, "ComputeMetrics", 14) == 0){
			printf("Function: ComputeMetrics.\n");
			// Name of the channel
			char *chname = malloc(sizeof(char) * 100);
			sprintf(chname, "Random Channel");
			
			// Assign names of metrics whose values must be computed.
			int nmetrics = 6;
			char **metrics = malloc(sizeof(char *) * nmetrics);
			for (i = 0; i < nmetrics; i ++)
				metrics[i] = malloc(sizeof(char) * 100);
			sprintf(metrics[0], "frb");
			sprintf(metrics[1], "infid");
			sprintf(metrics[2], "processfidelity");
			sprintf(metrics[3], "trn");
			sprintf(metrics[4], "entropy");
			sprintf(metrics[5], "np1");
			
			// Create an error channel -- any complex matrix which is positive definte matrix and has unit trace, can be the input channel's Choi matrix.
			// Create a random matrix M and declare (M + M^dag)/(2 * trace(M)) to be the input Choi matrix.
			for (i = 0; i < 4; i ++)
				for (j = 0; j < 4; j ++)
					mat[i][j] = genrand_real3() + genrand_real3() * I;
			double complex **choi = malloc(sizeof(double complex*) * 4);
			double trchoi = 0;
			for (i = 0; i < 4; i ++){
				choi[i] = malloc(sizeof(double complex) * 4);
				for (j = 0; j < 4; j ++){
					choi[i][j] = 0;
					for (k = 0; k < 4; k ++)
						choi[i][j] += mat[i][k] * conj(mat[j][k]);
				}
				trchoi += choi[i][i];
			}
			for (i = 0; i < 4; i ++)
				for (j = 0; j < 4; j ++)
					choi[i][j] = choi[i][j]/trchoi;
			PrintComplexArray2D(choi, "Noise channel", 4, 4);

			// Initialize the constants
			struct constants_t *consts_logmetrics = malloc(sizeof(struct constants_t));
			InitConstants(consts_logmetrics);
			
			// Initialize an array to hold the output metric values
			double *metvals = malloc(sizeof(double) * nmetrics);
			
			// Call function to compute the metric values.
			ComputeMetrics(metvals, nmetrics, metrics, choi, chname, consts_logmetrics);
			
			// Print metric values
			for (i = 0; i < nmetrics; i ++)
				printf("%s = %g\n", metrics[i], metvals[i]);
			
			// Free memory
			free(chname);
			for (i = 0; i < nmetrics; i ++)
				free(metrics[i]);
			free(metrics);
			free(metvals);
			for (i = 0; i < 4; i ++)
				free(choi[i]);
			free(choi);
			FreeConstants(consts_logmetrics);
			free(consts_logmetrics);
		}
	}

	if (strncmp(file, "memory", 6) == 0){
		// Testing functions in memory.c
		// Create a pointer to simulation structure.
		int nphys = 7, nenc = 1;
		struct simul_t *sim = malloc(sizeof(struct simul_t));
		if (strncmp(file, "AllocSimParams", 14) == 0){
			printf("Function: AllocSimParams.\n");
			AllocSimParams(sim, nphys, nenc);
		}
		if (strncmp(file, "AllocSimParamsQECC", 18) == 0){
			printf("Function: AllocSimParamsQECC.\n");
			AllocSimParamsQECC(sim, nphys, nenc);
		}
		if (strncmp(file, "FreeSimParams", 13) == 0){
			printf("Function: FreeSimParams.\n");
			FreeSimParams(sim, nphys, nenc);
		}
		if (strncmp(file, "FreeSimParamsQECC", 17) == 0){
			printf("Function: FreeSimParamsQECC.\n");
			FreeSimParamsQECC(sim, nphys, nenc);
		}
	}

	// Free memory
	for (i = 0; i < 4; i ++)
		free(mat[i]);
	free(mat);
	free(file);
	free(func);
	return 0;
}