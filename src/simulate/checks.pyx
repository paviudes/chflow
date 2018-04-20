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


cdef int IsState(complex128_t **choi, long double atol):
	# Check is a 4 x 4 matrix is a valid density matrix.
	# It must be Hermitian, have trace 1 and completely positive.
	cdef int isstate = 0, ishermitian = 0, ispositive = 0, istrace1 = 0
	ishermitian = IsHermitian(choi, atol)
	ispositive = IsPositive(choi, atol)
	istrace1 = IsTraceOne(choi, atol)
	isstate = ishermitian * istrace1 * ispositive
	return isstate


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


