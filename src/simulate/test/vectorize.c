#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include "arrayfuns.h"
// To compile: gcc -fopt-info-vec-missed=missed.txt vectorize.c -o vectorize.o

typedef double complex complex128_t;

complex128_t Sum(complex128_t ****arr, const int shape[4]){
	// Compute the sum of all elements in a 4D array
	int i, j, k, l, s;
	const int size = shape[0] * shape[1] * shape[2] * shape[3];
	complex128_t sum = 0;
	for (s = 0; s < size; s ++){
		l = s % shape[3];
		k = ((int) (s/shape[3])) % shape[2];
		j = ((int) (s/(shape[2] * shape[3]))) % shape[1];
		i = ((int) (s/(shape[1] * shape[2] * shape[3]))) % shape[0];
		sum += arr[i][j][k][l];
	}
	return sum;
}

void Contract(complex128_t ****arr, const int shape[4], complex128_t **contracted){
	// Contract the last two indices of a 4D array
	int i, j, k, l, s;
	const int size = shape[0] * shape[1] * shape[2] * shape[3];
	for (s = 0; s < size; s ++){
		l = s % shape[3];
		k = ((int) (s/shape[3])) % shape[2];
		j = ((int) (s/(shape[2] * shape[3]))) % shape[1];
		i = ((int) (s/(shape[1] * shape[2] * shape[3]))) % shape[0];
		contracted[i][j] += arr[i][j][k][l];
	}
}

void Offset(complex128_t ****arr, const int shape[4]){
	// If the array indices are in ascending order, add an offset.
	// Else, do nothing.
	// The offset is some random number between 1 and 100
	int i, j, k, l, s;
	const int size = shape[0] * shape[1] * shape[2] * shape[3];
	for (s = 0; s < size; s ++){
		l = s % shape[3];
		k = ((int) (s/shape[3])) % shape[2];
		j = ((int) (s/(shape[2] * shape[3]))) % shape[1];
		i = ((int) (s/(shape[1] * shape[2] * shape[3]))) % shape[0];
		arr[i][j][k][l] += pow(-1, rand() % 100) + I * pow(-1, rand() % 100);
	}
}

int main(int argc, char const *argv[]){
	// Initialize the random number generator
	srand(time(NULL));
	
	// Populate a 4D array with entries between 0 and 100
	const int shape[4] = {4, 4, 4, 4};
	complex128_t ****arr = malloc(sizeof(complex128_t ***) * shape[0]);;
	Initialize4D(arr, shape);
	// print the array
	Print4D(arr, shape);
	
	// Ofset elements
	// Offset(arr, shape);

	// Compute the sum of elements in the array
	complex128_t sum = Sum(arr, shape);
	printf("Sum of all elements = %g + i %g.\n", creal(sum), cimag(sum));

	// Contract the last two indices
	complex128_t **contracted = malloc(sizeof(complex128_t *) * shape[0]);;
	Initialize2D(contracted, shape);
	Contract(arr, shape, contracted);
	Print2D(contracted, shape);
	
	// Free the array
	Free4D(arr, shape);
	free(arr);
	Free2D(contracted, shape);
	free(contracted);
	return 0;
}