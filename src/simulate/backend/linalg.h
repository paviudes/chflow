#ifndef LINALG_H
#define LINALG_H

#include <complex.h>

/*
	Multiply two complex matrices A and B to produce a matrix C := A . B.
	We will use the BLAS routine zgemm to multiply matrices. See https://software.intel.com/en-us/node/520775.
	Inputs:
		double complex **A: complex matrix of shape (i x k)
		double complex **B: complex matrix of shape (k x j)
		double complex **C: complex matrix of shape (i x j)
		int i: number of rows in A
		int k: number of columns in A (also, number of rows in B)
		int j: number of columns in B
		In the function definition, we use
			A --> matA
			B --> matB
			C --> prod
			i --> rowsA
			k --> colsA ,which is also rowsB
			j --> colsB
*/
extern void Dot(double complex **matA, double complex **matB, double complex **prod, int rowsA, int colsA, int rowsB, int colsB);

/*
	Compute the eigenvalues and the right-eigenvectors of a complex square matrix.
	We will use the LAPACK routine zgeev to compute the eigenvalues. The LAPACK function is defined in https://software.intel.com/en-us/node/521147.
	Inputs:
		double complex **mat: square matrix of shape (nrows x nrows).
		int nrows: number of rows in the square matrix (also the number of columns)
		double complex *eigvals: complex array which will contain the eigenvalues of the input matrix "mat".
		int iseigvecs: binary value which is 0 when eigenvectors need not be stored and 1 otherwise.
		double complex **eigvecs: complex matrix of shape (nrows x nrows). The i-th column of this matrix contains the i-th eigen vector associated to the eigenvalue in "eigvals[i]".
*/
extern void Diagonalize(double complex **mat, int nrows, double complex *eigvals, int iseigvecs, double complex **eigvecs);

#endif /* LINALG_H */