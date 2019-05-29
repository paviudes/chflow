#include <stdio.h>
#include <complex.h>
#include "printfuns.h"

void PrintComplexArray1D(complex double *array, char *name, int nrows){
	// Print a complex 2D array.
	int r;
	printf("-----\n");
	printf("Array name: %s\n", name);
	for (r = 0; r < nrows; r ++)
		printf("    %g + i %g", creal(array[r]), cimag(array[r]));
	printf("\n");
	printf("-----\n");
}

void PrintComplexArray2D(complex double **array, char *name, int nrows, int ncols){
	// Print a complex 2D array.
	int r, c;
	printf("-----\n");
	printf("Array name: %s\n", name);
	for (r = 0; r < nrows; r ++){
		for (c = 0; c < ncols; c ++)
			printf("    %g + i %g", creal(array[r][c]), cimag(array[r][c]));
		printf("\n");
	}
	printf("-----\n");
}

void PrintDoubleArray2D(double **array, char *name, int nrows, int ncols){
	// Print a double 2D array.
	int r, c;
	printf("-----\n");
	printf("Array name: %s\n", name);
	for (r = 0; r < nrows; r ++){
		for (c = 0; c < ncols; c ++)
			printf("    %g", (array[r][c]));
		printf("\n");
	}
	printf("-----\n");
}

void PrintDoubleArray1D(double *array, char *name, int nrows){
	// Print a double 1D array.
	int r;
	printf("-----\n");
	printf("Array name: %s\n", name);
	for (r = 0; r < nrows; r ++)
		printf("    %g", (array[r]));
	printf("\n");
	printf("-----\n");
}

void PrintIntArray2D(int **array, char *name, int nrows, int ncols){
	// Print a int 2D array.
	int r, c;
	printf("-----\n");
	printf("Array name: %s\n", name);
	for (r = 0; r < nrows; r ++){
		for (c = 0; c < ncols; c ++)
			printf("    %d", (array[r][c]));
		printf("\n");
	}
	printf("-----\n");
}

void PrintIntArray1D(int *array, char *name, int nrows){
	// Print a int 1D array.
	int r;
	printf("-----\n");
	printf("Array name: %s\n", name);
	for (r = 0; r < nrows; r ++)
		printf("    %d", (array[r]));
	printf("\n");
	printf("-----\n");
}

void PrintLongArray2D(long **array, char *name, int nrows, int ncols){
	// Print a long 2D array.
	int r, c;
	printf("-----\n");
	printf("Array name: %s\n", name);
	for (r = 0; r < nrows; r ++){
		for (c = 0; c < ncols; c ++)
			printf("    %ld", (array[r][c]));
		printf("\n");
	}
	printf("-----\n");
}

void PrintLongArray1D(long *array, char *name, int nrows){
	// Print a long 1D array.
	int r;
	printf("-----\n");
	printf("Array name: %s\n", name);
	for (r = 0; r < nrows; r ++)
		printf("    %ld", (array[r]));
	printf("\n");
	printf("-----\n");
}