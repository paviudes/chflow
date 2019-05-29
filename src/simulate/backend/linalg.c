#include <stdlib.h>
#include <stdio.h>
#include <time.h> // Only for testing purposes
#include <string.h> // Only for testing purposes
#include <math.h>
#include <complex.h>
#include "mkl_lapacke.h"
#include "mkl_cblas.h"
#include "mt19937/mt19937ar.h"
#include "printfuns.h" // Only for testing purposes
#include "linalg.h"

/*
	To compile this file, use the compiler options and link commands in file:///opt/intel/documentation_2019/en/mkl/common/mkl_link_line_advisor.htm .
	Do the following commands to compile this file.
		source /opt/intel/compilers_and_libraries_2019/mac/bin/compilervars.sh intel64
		gcc mt19937/mt19937ar.c printfuns.c -m64 -I${MKLROOT}/include -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_rt -lpthread -lm -ldl linalg.c -o linalg.o
*/

double SumDotInt(double **matA, int *vecB, int rowsA, int colsA, int rowsB){
	// This is a wrapper for SumDot where the vector has integers.
	double *dvecB = malloc(sizeof(double) * rowsB);
	int i;
	for (i = 0; i < rowsB; i ++)
		dvecB[i] = (double) vecB[i];
	double sum = SumDot(matA, dvecB, rowsA, colsA, rowsB);
	free(dvecB);
	return sum;
}

double SumDot(double **matA, double *vecB, int rowsA, int colsA, int rowsB){
	// Given a matrix A and a vector v, compute the sum of entries in (A.v).
	// The number of columns in A must be equal to the number of rows in B.
	const double atol = 10E-12;
	int i;
	// Allocate memory for the product matrix
	double *prod = malloc(sizeof(double *) * rowsA);

	// Compute the product
	GDotV(matA, vecB, prod, rowsA, colsA, rowsB);

	// Compute the sum of entries
	double sum = 0;
	for (i = 0; i < rowsA; i ++)
		sum += prod[i];

	// Free memory of the product matrix.
	free(prod);

	if(fabs(sum) < atol)
		sum = 0;
	return sum;
}

double Trace(double **mat, int nrows){
	// Compute the trace of a matrix.
	// We will use a vectorized for loop.
	int i;
	double trace = 0;
	for (i = 0; i < nrows; i ++)
		trace += mat[i][i];
	return trace;
}

double DiagGDotIntV(double **matA, int *vecB, int rowsA, int colsA, int sizeB){
	// This is a wrapper for DiagGDotV where the vector has integers.
	double *dvecB = malloc(sizeof(double) * sizeB);
	int i;
	for (i = 0; i < sizeB; i ++)
		dvecB[i] = (double) vecB[i];
	double diagdot = DiagGDotV(matA, dvecB, rowsA, colsA, sizeB);
	free(dvecB);
	return diagdot;
}

double DiagGDotV(double **matA, double *vecB, int rowsA, int colsA, int sizeB){
	// Given a matrix A and a vector v, compute the dot product: diag(A).v where diag(A) referes to the 1D vector containing the diagonal of A.
	// The vector is provided as a 1D array (row vector), but we intend to use it as a column vector and multiply it to the right of the given matrix.
	// For high-performance, we will use the cblas_ddot function of the BLAS library.
	// See https://software.intel.com/en-us/mkl-developer-reference-c-cblas-dot.
	const double atol = 10E-12;
	double prod = 0;
	if ((rowsA != colsA) && (colsA != sizeB))
		printf("Cannot multiply a matrix of shape (%d x %d) to a vector of shape (%d x 1).\n", rowsA, colsA, sizeB);
	else{
		MKL_INT n = rowsA;
		const MKL_INT incx = 1, incy = 1;
		double x[n], y[n];
		int i;
		for (i = 0; i < rowsA; i ++)
			x[i] = matA[i][i];
		for (i = 0; i < sizeB; i ++)
			y[i] = vecB[i];
		prod = cblas_ddot(n, x, incx, y, incy);
		if (fabs(prod) < atol)
			prod = 0;
	}
	return prod;
}


void GDotV(double **matA, double *vecB, double *prod, int rowsA, int colsA, int sizeB){
	/*
		Multiply a double matrix M with a vector v, as M.v.
		The vector is provided as a 1D array (row vector), but we intend to use it as a column vector and multiply it to the right of the given matrix.
		The product is also a 1D array, assumed to be a column vector.
		For high-performance, we will use the zgemm function of the BLAS library.
		See https://software.intel.com/en-us/mkl-tutorial-c-multiplying-matrices-using-dgemm .
		The dgemm function is defined with the following parameters.
		extern cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						   int m, // number of rows of A
						   int n, // number of columns of B
						   int k, // number of columns of A = number of rows of B
						   double alpha, // scalar factor to be multiplied to the product.
						   double *A, // input matrix A
						   int k, // leading dimension of A
						   double *B, // input matrix B
						   int n, // leading dimension of B
						   double beta, // relative shift from the product, by a factor of C.
						   double *C, // product of A and B
						   int n //leading dimension of C.
						  );
	*/
	if (colsA != sizeB)
		printf("Cannot multiply a matrix of shape (%d x %d) to a vector of shape (%d x 1).\n", rowsA, colsA, sizeB);
	else{
		MKL_INT m = rowsA, n = 1, k = colsA;
		double A[rowsA * colsA], B[sizeB], C[rowsA], alpha, beta;
		int i;
		for (i = 0; i < rowsA * colsA; i ++)
			A[i] = matA[i/colsA][i % colsA];

		for (i = 0; i < sizeB; i ++)
			B[i] = vecB[i];
		
		alpha = 1;
		beta = 0;
		// Call the BLAS function.
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
		// Load the product
		for (i = 0; i < rowsA; i ++)
			prod[i] = C[i];
	}
}

void GDot(double **matA, double **matB, double **prod, int rowsA, int colsA, int rowsB, int colsB){
	/*
		Multiply two double matrices.
		For high-performance, we will use the zgemm function of the BLAS library.
		See https://software.intel.com/en-us/mkl-tutorial-c-multiplying-matrices-using-dgemm .
		The dgemm function is defined with the following parameters.
		extern cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						   int m, // number of rows of A
						   int n, // number of columns of B
						   int k, // number of columns of A = number of rows of B
						   double alpha, // scalar factor to be multiplied to the product.
						   double *A, // input matrix A
						   int k, // leading dimension of A
						   double *B, // input matrix B
						   int n, // leading dimension of B
						   double beta, // relative shift from the product, by a factor of C.
						   double *C, // product of A and B
						   int n //leading dimension of C.
						  );
	*/
	if (colsA != rowsB)
		printf("Cannot multiply matrices of shape (%d x %d) and (%d x %d).\n", rowsA, colsA, rowsB, colsB);
	else{
		MKL_INT m = rowsA, n = colsB, k = colsA;
		double A[rowsA * colsA], B[rowsB * colsB], C[rowsA * colsB], alpha, beta;
		int i;
		for (i = 0; i < rowsA * colsA; i ++)
			A[i] = matA[i/colsA][i % colsA];

		for (i = 0; i < rowsB * colsB; i ++)
			B[i] = matB[i/colsB][i % colsB];
		
		alpha = 1;
		beta = 0;
		// Call the BLAS function.
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
		// Load the product
		for (i = 0; i < rowsA * colsB; i ++)
			prod[i/colsB][i % colsB] = C[i];
	}
}

void ZDot(double complex **matA, double complex **matB, double complex **prod, int rowsA, int colsA, int rowsB, int colsB){
	/*
		Multiply two complex matrices.
		For high-performance, we will use the zgemm function of the BLAS library.
		See https://software.intel.com/en-us/node/520775 .
		The zgemm function is defined with the following parameters.
		extern cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						   int m, // number of rows of A
						   int n, // number of columns of B
						   int k, // number of columns of A = number of rows of B
						   double alpha, // scalar factor to be multiplied to the product.
						   double complex *A, // input matrix A
						   int k, // leading dimension of A
						   double complex *B, // input matrix B
						   int n, // leading dimension of B
						   double beta, // relative shift from the product, by a factor of C.
						   double complex *C, // product of A and B
						   int n //leading dimension of C.
						  );
	*/
	if (colsA != rowsB)
		printf("Cannot multiply matrices of shape (%d x %d) and (%d x %d).\n", rowsA, colsA, rowsB, colsB);
	else{
		MKL_INT m = rowsA, n = colsB, k = colsA;
		MKL_Complex16 A[rowsA * colsA], B[rowsB * colsB], C[rowsA * colsB], alpha, beta;
		int i, j;
		for (i = 0; i < rowsA; i ++){
			for (j = 0; j < colsA; j ++){
				A[i * colsA + j].real = creal(matA[i][j]);
				A[i * colsA + j].imag = cimag(matA[i][j]);
			}
		}
		for (i = 0; i < rowsB; i ++){
			for (j = 0; j < colsB; j ++){
				B[i * colsB + j].real = creal(matB[i][j]);
				B[i * colsB + j].imag = cimag(matB[i][j]);
			}
		}
		alpha.real = 1;
		alpha.imag = 0;
		beta.real = 0;
		beta.imag = 0;
		// Call the BLAS function.
		cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, A, k, B, n, &beta, C, n);
		// Load the product
		for (i = 0; i < rowsA; i ++)
			for (j = 0; j < colsB; j ++)
				prod[i][j] = C[i * colsB + j].real + C[i * colsB + j].imag * I;
	}
}


void Diagonalize(double complex **mat, int nrows, double complex *eigvals, int iseigvecs, double complex **eigvecs){
	/*
		Compute the eigenvalues and the right-eigenvectors of a complex square matrix.
		We will use the LAPACK routine zgeev to compute the eigenvalues. The LAPACK function is defined as follows.
		extern void zgeev(char* jobvl, // Should left eigenvectors be computed? 'V' for yes and 'N' for no.
						  char* jobvr, // Should right eigenvectors be computed? 'V' for yes and 'N' for no.
						  int* n, // The order of the matrix.
						  dcomplex* a, // complex array containing the n x n complex matrix in vectorized form.
						  int* lda, // The leading dimension of the matrix.
						  dcomplex* w, // Array of the computed eigenvalues. Complex array of size n.
						  dcomplex* vl, // The left eigenvectors of A, stored as columns of this n x n matrix. The order of the eigenvectors is the same as the order of the eigenvalues.
						  int* lvdl, // The leading dimension of the array vl.
						  dcomplex* vr, // The right eigenvectors of A, stored as columns of this n x n matrix. The order of the eigenvectors is the same as the order of the eigenvalues.
						  int* lvdr, // The leading dimension of the array vr.
						  dcomplex* work, // A scratch workspace for the algorithm.
						  int* lwork, // Size of the array: work. If it is -1, then running the algorithm doesn't compute the eigenvalues but assigns the optimal size of the work array to lwork.
						  double* rwork, // double precision array of size 2 * n
						  int* info); // result. 0 if successful, -i if i-th argument has illegal value and +i if 1 to i eigen values of w have not converged.
		
		There is a C wrapper to the LAPACK routine:
		https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_zgeev_row.c.htm
	*/

	MKL_INT n = nrows, lda = nrows, lvdl = nrows, lvdr = nrows, info;
	MKL_Complex16 w[nrows], vl[nrows * nrows], vr[nrows * nrows];
	MKL_Complex16 a[nrows * nrows];
	int i, j;
	for (i = 0; i < nrows; i ++){
		for (j = 0; j < nrows; j ++){
			a[i * nrows + j].real = creal(mat[i][j]);
			a[i * nrows + j].imag = cimag(mat[i][j]);
		}
	}
	info = LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'V', 'V', n, a, lda, w, vl, lvdl, vr, lvdr);
	if (info > 0)
		printf("Eigenvalues %d to %d did not converge properly.\n", info, nrows);
	else if (info < 0)
		printf("Error in the the %d-th input parameter.\n", -1 * info);
	else{
		for (i = 0; i < nrows; i ++){
			eigvals[i] = w[i].real + w[i].imag * I;
			if (iseigvecs == 1){
				for (j = 0; j < nrows; j ++)
					eigvecs[i][j] = vr[i * nrows + j].real + vr[i * nrows + j].imag * I;
			}
		}
	}
}

/*
int main(int argc, char const *argv[]){
	// Seed random number generator
	init_genrand(time(NULL));
	int i, j;
	int rowsA = 4, colsA = 4, rowsB = 4, colsB = 4;
	if (strncmp(argv[1], "ZDot", 4) == 0){
		printf("Function: ZDot.\n");
		// Assign random elements to matrices A and B.
		double complex **matA = malloc(sizeof(double complex *) * rowsA);
		for (i = 0; i < rowsA; i ++){
			matA[i] = malloc(sizeof(double complex) * colsA);
			for (j = 0; j < colsA; j ++)
				matA[i][j] = genrand_real3() + genrand_real3() * I;
		}
		PrintComplexArray2D(matA, "A", rowsA, colsA);
		double complex **matB = malloc(sizeof(double complex *) * rowsB);
		for (i = 0; i < rowsB; i ++){
			matB[i] = malloc(sizeof(double complex) * colsB);
			for (j = 0; j < colsB; j ++)
				matB[i][j] = genrand_real3() + genrand_real3() * I;
		}
		PrintComplexArray2D(matB, "B", rowsB, colsB);
		// Initialize the product
		double complex **matC = malloc(sizeof(double complex *) * rowsA);
		for (i = 0; i < rowsA; i ++)
			matC[i] = malloc(sizeof(double complex) * colsB);
		// Call the matrix product
		ZDot(matA, matB, matC, rowsA, colsA, rowsB, colsB);
		PrintComplexArray2D(matC, "C = A . B", rowsA, colsB);
		// Free all matrices
		for (i = 0; i < rowsA; i ++)
			free(matA[i]);
		free(matA);
		for (i = 0; i < rowsB; i ++)
			free(matB[i]);
		free(matB);
		for (i = 0; i < rowsA; i ++)
			free(matC[i]);
		free(matC);
	}
	if (strncmp(argv[1], "GDot", 4) == 0){
		printf("Function: GDot.\n");
		// Assign random elements to matrices A and B.
		double **matA = malloc(sizeof(double *) * rowsA);
		for (i = 0; i < rowsA; i ++){
			matA[i] = malloc(sizeof(double) * colsA);
			for (j = 0; j < colsA; j ++)
				matA[i][j] = genrand_real3();
		}
		PrintDoubleArray2D(matA, "A", rowsA, colsA);
		double **matB = malloc(sizeof(double *) * rowsB);
		for (i = 0; i < rowsB; i ++){
			matB[i] = malloc(sizeof(double) * colsB);
			for (j = 0; j < colsB; j ++)
				matB[i][j] = genrand_real3();
		}
		PrintDoubleArray2D(matB, "B", rowsB, colsB);
		// Initialize the product
		double **matC = malloc(sizeof(double *) * rowsA);
		for (i = 0; i < rowsA; i ++)
			matC[i] = malloc(sizeof(double) * colsB);
		// Call the matrix product
		GDot(matA, matB, matC, rowsA, colsA, rowsB, colsB);
		PrintDoubleArray2D(matC, "C = A . B", rowsA, colsB);
		// Free all matrices
		for (i = 0; i < rowsA; i ++)
			free(matA[i]);
		free(matA);
		for (i = 0; i < rowsB; i ++)
			free(matB[i]);
		free(matB);
		for (i = 0; i < rowsA; i ++)
			free(matC[i]);
		free(matC);
	}
	if (strncmp(argv[1], "DiagGDotV", 9) == 0){
		printf("Function: DiagGDotV.\n");
		// Assign random elements to matrices A and B.
		double **matA = malloc(sizeof(double *) * rowsA);
		for (i = 0; i < rowsA; i ++){
			matA[i] = malloc(sizeof(double) * colsA);
			for (j = 0; j < colsA; j ++)
				matA[i][j] = genrand_real3();
		}
		PrintDoubleArray2D(matA, "A", rowsA, colsA);
		double *vecB = malloc(sizeof(double) * rowsB);
		for (i = 0; i < rowsB; i ++)
			vecB[i] = genrand_real3();
		PrintDoubleArray1D(vecB, "v", rowsB);
		// Initialize the product
		double prod = DiagGDotV(matA, vecB, rowsA, colsA, rowsB);
		printf("diag(A).v = %g.\n", prod);
		// Free all matrices
		for (i = 0; i < rowsA; i ++)
			free(matA[i]);
		free(matA);
		free(vecB);
	}
	if (strncmp(argv[1], "DiagGDotIntV", 12) == 0){
		printf("Function: DiagGDotIntV.\n");
		// Assign random elements to matrices A and B.
		double **matA = malloc(sizeof(double *) * rowsA);
		for (i = 0; i < rowsA; i ++){
			matA[i] = malloc(sizeof(double) * colsA);
			for (j = 0; j < colsA; j ++)
				matA[i][j] = genrand_real3();
		}
		PrintDoubleArray2D(matA, "A", rowsA, colsA);
		int *vecB = malloc(sizeof(int) * rowsB);
		for (i = 0; i < rowsB; i ++)
			vecB[i] = (int) (100 * genrand_real3());
		PrintIntArray1D(vecB, "v", rowsB);
		// Initialize the product
		double prod = DiagGDotIntV(matA, vecB, rowsA, colsA, rowsB);
		printf("diag(A).v = %g.\n", prod);
		// Free all matrices
		for (i = 0; i < rowsA; i ++)
			free(matA[i]);
		free(matA);
		free(vecB);
	}
	return 0;
}
*/