#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "arrayfuns.h"

typedef double complex complex128_t;

void Initialize4D(complex128_t ****arr, const int shape[4]){
	// Populate a 4D array of a given shape
	// The array entries are numbers between 1 and 100.
	int i, j, k, l;
	// arr
	for (i = 0; i < shape[0]; i ++){
		arr[i] = malloc(sizeof(complex128_t **) * shape[1]);
		for (j = 0; j < shape[1]; j ++){
			arr[i][j] = malloc(sizeof(complex128_t *) * shape[2]);
			for (k = 0; k < shape[2]; k ++){
				arr[i][j][k] = malloc(sizeof(complex128_t) * shape[3]);
				for (l = 0; l < shape[3]; l ++)
					arr[i][j][k][l] = pow(-1, rand() % 100) + I * pow(-1, rand() % 100);
			}
		}
	}
}

void Initialize2D(complex128_t **arr, const int shape[4]){
	// Populate a 2D array of a given shape
	// The array entries are numbers between 1 and 100.
	int i, j;
	for (i = 0; i < shape[0]; i ++){
		arr[i] = malloc(sizeof(complex128_t) * shape[1]);
		for (j = 0; j < shape[1]; j ++)
			arr[i][j] = 0;
	}
}

void Free4D(complex128_t ****arr, const int shape[4]){
	// Free memory allocated to a 4D array
	int i, j, k;
	for (i = 0; i < shape[0]; i ++){
		for (j = 0; j < shape[1]; j ++){
			for (k = 0; k < shape[2]; k ++)
				free(arr[i][j][k]);
			free(arr[i][j]);
		}
		free(arr[i]);
	}
}

void Free2D(complex128_t **arr, const int shape[4]){
	// Free memory allocated to a 2D array
	int i, j, k;
	for (i = 0; i < shape[0]; i ++)
		free(arr[i]);
}

void Print4D(complex128_t ****arr, const int shape[4]){
	// Print the contents of a 4D array
	int i, j, k, l;
	for (i = 0; i < shape[0]; i ++)
		for (j = 0; j < shape[1]; j ++)
			for (k = 0; k < shape[2]; k ++)
				for (l = 0; l < shape[3]; l ++)
					printf("arr[%d][%d][%d][%d] = %g + i %g\n", i, j, k, l, creal(arr[i][j][k][l]), cimag(arr[i][j][k][l]));
	printf("xxxxxx\n");
}

void Print2D(complex128_t **arr, const int shape[4]){
	// Print the contents of a 4D array
	int i, j, k, l;
	for (i = 0; i < shape[0]; i ++)
		for (j = 0; j < shape[1]; j ++)
			printf("C[%d][%d] = %g + i %g\n", i, j, creal(arr[i][j]), cimag(arr[i][j]));
	printf("xxxxxx\n");
}
