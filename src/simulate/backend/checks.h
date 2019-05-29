#ifndef CHECKS_H
#define CHECKS_H

#include <complex.h>

// Check is a matrix is a diagonal matrix or not.
extern int IsDiagonal(double **matrix, int size);

// Check if a given list of numbers is a normalized PDF.
extern int IsPDF(double *dist, int size);

// Check if an input complex 4x4 matrix is completely positive.
extern int IsPositive(double complex **choi);

// Check if the trace of a complex 4x4 matrix is 1.
extern int IsTraceOne(double complex **choi);

// Check is a complex 4x4 matrix is Hermitian.
int IsHermitian(double complex **choi);

// Check is a 4 x 4 matrix is a valid density matrix.
extern int IsState(double complex **choi);

#endif /* CHECKS_H */