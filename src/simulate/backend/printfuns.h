#ifndef PRINTFUNS_H
#define PRINTFUNS_H

#include <complex.h>

// Print a complex 2D array.
extern void PrintComplexArray1D(double complex *array, char *name, int nrows);

// Print a complex 2D array.
extern void PrintComplexArray2D(double complex **array, char *name, int nrows, int ncols);

// Print a double 2D array.
extern void PrintDoubleArray2D(double **array, char *name, int nrows, int ncols);

// Print a double 1D array.
extern void PrintDoubleArray1D(double *array, char *name, int nrows);

// Print a int 2D array.
extern void PrintIntArray2D(int **array, char *name, int nrows, int ncols);

// Print a int 1D array.
extern void PrintIntArray1D(int *array, char *name, int nrows);

// Print a long 2D array.
extern void PrintLongArray2D(long **array, char *name, int nrows, int ncols);

// Print a long 1D array.
extern void PrintLongArray1D(long *array, char *name, int nrows);

#endif /* PRINTFUNS_H */