#ifndef LINALG_H
#define LINALG_H

#include <complex.h>

/*
	Given a matrix A and a vector v, compute the sum of entries in (A.v).
*/
extern double SumDot(double **matA, double *vecB, int rowsA, int colsA, int rowsB);

/*
	This is a wrapper for SumDot where the vector has integers.
*/
extern double SumDotInt(double **matA, int *vecB, int rowsA, int colsA, int rowsB);

/*
	Compute the trace of a matrix.
	We will use a vectorized for loop.
*/
extern double Trace(double **mat, int nrows);

/*
	Given a matrix A and a vector v, compute the dot product: diag(A).v where diag(A) referes to the 1D vector containing the diagonal of A.
	The vector is provided as a 1D array (row vector), but we intend to use it as a column vector and multiply it to the right of the given matrix.
*/
extern double DiagGDotV(double **matA, double *vecB, int rowsA, int colsA, int rowsB);

/*
	Multiply a double matrix M with a vector v, as C = M.v.
	The vector is provided as a 1D array (row vector), but we intend to use it as a column vector and multiply it to the right of the given matrix.
	The product is also a 1D array, assumed to be a column vector.
	We will use the BLAS routine dgemm to multiply matrices. See https://software.intel.com/en-us/mkl-tutorial-c-multiplying-matrices-using-dgemm .
	Inputs:
		double **M: double matrix of shape (i x k)
		double *v: double array of length k
		double *prod: double array of length i
		int i: number of rows in A
		int j: number of columns in A
		int k: number of elements in B
		In the function definition, we use
			M --> matA
			v --> matB
			C --> prod
			i --> rowsA
			k --> colsA , which is also rowsB
			j --> colsB
*/
extern void GDotV(double **matA, double *vecB, double *prod, int rowsA, int colsA, int rowsB);

/*
	This is a wrapper for DiagGDotV where the vector has integers.
*/
extern double DiagGDotIntV(double **matA, int *vecB, int rowsA, int colsA, int rowsB);

/*
	Multiply two double matrices A and B to produce a matrix C := A . B.
	We will use the BLAS routine zgemm to multiply matrices. See https://software.intel.com/en-us/node/520775.
	Inputs:
		double **A: double matrix of shape (i x k)
		double **B: double matrix of shape (k x j)
		double **C: double matrix of shape (i x j)
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
extern void GDot(double **matA, double **matB, double **prod, int rowsA, int colsA, int rowsB, int colsB);



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
extern void ZDot(double complex **matA, double complex **matB, double complex **prod, int rowsA, int colsA, int rowsB, int colsB);

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