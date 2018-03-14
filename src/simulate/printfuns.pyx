#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False
cimport cython

cdef extern from "complex.h":
	double creal(double complex cnum)
	double cimag(double complex cnum)


cdef int PrintComplexArray2D(complex128_t **array, char *name, int nrows, int ncols):
	# print a complex 2D array
	cdef:
		int r, c
	print("-----")
	print("Array name: %s" % (name))
	for r in range(nrows):
		for c in range(ncols):
			print("    %.4e + i %.4e" % (creal(array[r][c]), cimag(array[r][c]))),
		print("")
	print("-----")
	return 0


cdef int PrintDoubleArray2D(long double **array, char *name, int nrows, int ncols):
	# print a double 2D array
	cdef:
		int r, c
	print("-----")
	print("Array name: %s" % (name))
	for r in range(nrows):
		for c in range(ncols):
			print("    %.4e" % (array[r][c])),
		print("")
	print("-----")
	return 0

cdef int PrintDoubleArray1D(long double *array, char *name, int size):
	# print a double 2D array
	cdef:
		int r
	print("-----")
	print("Array name: %s" % (name))
	for r in range(size):
		print("    %.4e" % (array[r])),
	print("")
	print("-----")
	return 0

cdef int PrintIntArray2D(int **array, char *name, int nrows, int ncols):
	# print a int 2D array
	cdef:
		int r, c
	print("-----")
	print("Array name: %s" % (name))
	for r in range(nrows):
		for c in range(ncols):
			print("    %d" % (array[r][c])),
		print("")
	print("-----")
	return 0

cdef int PrintIntArray1D(int *array, char *name, int size):
	# print a int 2D array
	cdef:
		int r
	print("-----")
	print("Array name: %s" % (name))
	for r in range(size):
		print("    %d" % (array[r])),
	print("")
	print("-----")
	return 0

cdef int PrintLongArray2D(long **array, char *name, int nrows, int ncols):
	# print a long 2D array
	cdef:
		int r, c
	print("-----")
	print("Array name: %s" % (name))
	for r in range(nrows):
		for c in range(ncols):
			print("    %ld" % (array[r][c])),
		print("")
	print("-----")
	return 0

cdef int PrintLongArray1D(long *array, char *name, int size):
	# print a long 2D array
	cdef:
		int r
	print("-----")
	print("Array name: %s" % (name))
	for r in range(size):
		print("    %ld" % (array[r])),
	print("")
	print("-----")
	return 0