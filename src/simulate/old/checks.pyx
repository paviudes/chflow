#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False
import sys
try:
	import numpy as np
	cimport numpy as np
except:
	pass

cdef extern from "complex.h":
	long double creall(long double complex cnum)
	long double cimagl(long double complex cnum)
	long double cabsl(long double complex cnum)
	long double complex conjl(long double complex cnum)

cdef extern from "math.h" nogil:
	long double fabsl(long double num)

cdef int IsDiagonal(long double **matrix, int size) nogil:
	# Check is a matrix is a diagonal matrix or not.
	# If the ratio between the sum of absolute values of off-diagonal elements to that of the diagonal elements is less than some threshold, output yes.
	cdef:
		long double rtol = 10E-20
		long double diagsum = 0, offdiagsum = 0
		int r, c, isdiag = 1
	for r in range(size):
		for c in range(size):
			if (r == c):
				diagsum = diagsum + fabsl(matrix[r][c])
			else:
				offdiagsum = offdiagsum + fabsl(matrix[r][c])
	if (offdiagsum/diagsum > rtol):
		isdiag = 0
	return isdiag

######################
# Converted to C.

int IsDiagonal(long double **matrix, int size){
	// Check is a matrix is a diagonal matrix or not.
	// If the ratio between the sum of absolute values of off-diagonal elements to that of the diagonal elements is less than some threshold, output yes.
	long double rtol = 10E-20;
	long double diagsum = 0, offdiagsum = 0;
	int r, c, isdiag = 1;
	for (r = 0; r < size; r ++){
		for (c = 0; c < size; c ++){
			if (r == c){
				diagsum = diagsum + fabsl(matrix[r][c]);
			}
			else{
				offdiagsum = offdiagsum + fabsl(matrix[r][c]);
			}
		}
	}
	if (offdiagsum/diagsum > rtol){
		isdiag = 0;
	}
	return isdiag;
}
######################


cdef int IsState(complex128_t **choi, long double atol):
	# Check is a 4 x 4 matrix is a valid density matrix.
	# It must be Hermitian, have trace 1 and completely positive.
	cdef int isstate = 0, ishermitian = 0, ispositive = 0, istrace1 = 0
	ishermitian = IsHermitian(choi, atol)
	ispositive = IsPositive(choi, atol)
	istrace1 = IsTraceOne(choi, atol)
	isstate = ishermitian * istrace1 * ispositive
	return isstate

######################
# Converted to C.

int IsState(complex128_t **choi, long double atol){
	// Check is a 4 x 4 matrix is a valid density matrix.
	// It must be Hermitian, have trace 1 and completely positive.
	int isstate = 0, ishermitian = 0, ispositive = 0, istrace1 = 0;
	ishermitian = IsHermitian(choi, atol);
	ispositive = IsPositive(choi, atol);
	istrace1 = IsTraceOne(choi, atol);
	isstate = ishermitian * istrace1 * ispositive;
	return isstate;
}
######################


cdef int IsPositive(complex128_t **choi, long double atol):
	# test if an input complex 4x4 matrix is completely positive.
	# A completely positive matrix has only non-negative eigenvalues.
	cdef:
		np.ndarray[np.complex128_t, ndim = 2, mode = 'c'] choimatrix = np.zeros((4, 4), dtype = np.complex128)
		int i, j
	for i in range(4):
		for j in range(4):
			choimatrix[i, j] = creall(choi[i][j]) + 1j * cimagl(choi[i][j])
	singvals = np.linalg.svd(choimatrix, compute_uv = False)
	for i in range(4):
		if (cimagl(singvals[i]) > 0):
			return 0
		if (((-1) * creall(singvals[i])) > atol):
			return 0
	return 1

####################
# Converted to C.

int IsPositive(complex128_t **choi, long double atol){
	// Test if an input complex 4x4 matrix is completely positive.
	// A completely positive matrix has only non-negative eigenvalues.
	// We will use the LAPACK routine zgeev_ to compute the eigenvalues.
	// See http://physics.oregonstate.edu/~landaur/nacphy/lapack/codes/eigen-c.html for details.
	int i, j, dim = 4;
	double atol = 10E-8;
	// For the LAPACK method, we need to vectorize the matrix as well as represent each complex number by a tuple of reals.
	// Furthermore, we need to transpose the result since FORTRAN has a row ordering.
	double *vectorized = malloc((2 * dim * dim) * sizeof(double));
	for (i = 0; i < dim * dim; i ++){
		vectorized[2 * i] = reall(choi[i%4][i/4]);
		vectorized[2 * i + 1] = imagl(choi[i%4][i/4]);
	}
	// The following are additional inputs to the zgeev_.
	int ok, c1 = 4, c2 = 2 * dim, c3 = 1;
	char c4 = 'N';
	complex dummy[1][1], workspace[2 * dim], eigvals[dim];
	zgeev_(&c4, &c4, &c1, vectorized, &c1, eigvals, dummy, &c3, dummy, &c3, workspace, &c2, workspace, &ok);
	if (ok == 0){
		for (i = 0; i < dim; i ++){
			if (cimagl(eigvals[i]) > atol){
				return 0;
			}
		}
		return 1;
	}
	else{
		return -1;
		printf("There was a problem in computing eigenvalues.\n");
	}
	return 1;
}
####################


cdef int IsTraceOne(complex128_t **choi, long double atol):
	# check if the trace of a complex 4x4 matrix is 1.
	cdef:
		int i
		complex128_t trace = 0 + 0 * 1j
	for i in range(4):
		trace = trace + choi[i][i]
	if (fabsl(cimagl(trace)) > atol):
		return 0
	if (fabsl(creall(trace)  -  1.0) > atol):
		return 0
	return 1

####################
# Converted to C.

int IsTraceOne(complex128_t **choi, long double atol){
	// Check if the trace of a complex 4x4 matrix is 1.
	int i;
	complex128_t trace = 0 + 0 * I;
	for (i = 0; i < 4; i ++){
		trace = trace + choi[i][i];
	}
	if (fabsl(cimagl(trace)) > atol){
		return 0;
	}
	if (fabsl(creall(trace)  -  1.0) > atol){
		return 0;
	}
	return 1;
}
####################

cdef int IsHermitian(complex128_t **choi, long double atol):
	# Check is a complex 4x4 matrix is Hermitian.
	# For a Hermitian matrix A, we have: A[i][j] = (A[j][i])^*
	# atol = 10E-20
	cdef int i, j
	for i in range(4):
		for j in range(4):
			if (cabsl(choi[i][j] - conjl(choi[j][i])) > atol):
				return 0
	return 1

####################
# Converted to C.

int IsHermitian(complex128_t **choi, long double atol){
	// Check is a complex 4x4 matrix is Hermitian.
	// For a Hermitian matrix A, we have: A[i][j] = (A[j][i])^*.
	int i, j;
	for (i = 0; i < 4; i ++){
		for (j = 0; j < 4; j ++){
			if (cabsl(choi[i][j] - conjl(choi[j][i])) > atol){
				return 0;
			}
		}
	}
	return 1;
}
####################

cdef int IsPDF(long double *dist, int size, long double atol):
	# check if a given list of numbers is a normalized PDF
	cdef:
		int i
		long double norm = 0
	for i in range(size):
		norm = norm + dist[i]
		if (dist[i] < 0):
			return 0
	if (fabsl(norm - 1) > atol):
		return 0
	return 1

#####################
# Converted to C.

int IsPDF(long double *dist, int size, long double atol){
	// Check if a given list of numbers is a normalized PDF.
	int i;
	long double norm = 0;
	for (i = 0; i < size; i ++){
		norm = norm + dist[i];
		if (dist[i] < 0){
			return 0;
		}
	}
	if (fabsl(norm - 1) > atol){
		return 0;
	}
	return 1;
}
#####################
