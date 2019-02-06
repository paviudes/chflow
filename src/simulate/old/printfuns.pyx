#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False
cimport cython
from libc.stdio cimport printf

cdef extern from "complex.h" nogil:
	double creal(double complex cnum)
	double cimag(double complex cnum)


cdef int PrintComplexArray2D(complex128_t **array, char *name, int nrows, int ncols) nogil:
	# print a complex 2D array
	cdef:
		int r, c
	printf("-----\n")
	printf("Array name: %s\n", (name))
	for r in range(nrows):
		for c in range(ncols):
			printf("    %Lg + i %Lg", (creal(array[r][c]), cimag(array[r][c])))
		printf("\n")
	printf("-----\n")
	return 0


cdef int PrintDoubleArray2D(long double **array, char *name, int nrows, int ncols) nogil:
	# print a double 2D array
	cdef:
		int r, c
	printf("-----\n")
	printf("Array name: %s\n", (name))
	for r in range(nrows):
		for c in range(ncols):
			printf("    %Lg", (array[r][c]))
		printf("\n")
	printf("-----\n")
	return 0

cdef int PrintDoubleArray1D(long double *array, char *name, int size) nogil:
	# print a double 2D array
	cdef:
		int r
	printf("-----\n")
	printf("Array name: %s\n", (name))
	for r in range(size):
		printf("    %Lg", (array[r]))
	printf("\n")
	printf("-----\n")
	return 0

cdef int PrintIntArray2D(int **array, char *name, int nrows, int ncols) nogil:
	# print a int 2D array
	cdef:
		int r, c
	printf("-----\n")
	printf("Array name: %s\n", (name))
	for r in range(nrows):
		for c in range(ncols):
			printf("    %d", (array[r][c]))
		printf("\n")
	printf("-----\n")
	return 0

cdef int PrintIntArray1D(int *array, char *name, int size) nogil:
	# print a int 2D array
	cdef:
		int r
	printf("-----\n")
	printf("Array name: %s\n", (name))
	for r in range(size):
		printf("    %d", (array[r]))
	printf("\n")
	printf("-----\n")
	return 0

cdef int PrintLongArray2D(long **array, char *name, int nrows, int ncols) nogil:
	# print a long 2D array
	cdef:
		int r, c
	printf("-----\n")
	printf("Array name: %s\n", (name))
	for r in range(nrows):
		for c in range(ncols):
			printf("    %ld", (array[r][c]))
		printf("\n")
	printf("-----\n")
	return 0

cdef int PrintLongArray1D(long *array, char *name, int size) nogil:
	# print a long 2D array
	cdef:
		int r
	printf("-----\n")
	printf("Array name: %s\n", (name))
	for r in range(size):
		printf("    %ld", (array[r]))
	printf("\n")
	printf("-----\n")
	return 0

##################################
# Converted in C.

void PrintComplexArray2D(complex128_t **array, char *name, int nrows, int ncols){
	// Print a complex 2D array.
	int r, c;
	printf("-----\n");
	printf("Array name: %s\n", (name));
	for (r = 0; r < nrows; r ++){
		for (c = 0; c < nrows; c ++){
			printf("    %Lg + i %Lg", (creal(array[r][c]), cimag(array[r][c])));
		}
		printf("\n");
	}
	printf("-----\n");
}

void PrintDoubleArray2D(long double **array, char *name, int nrows, int ncols){
	// Print a double 2D array.
	int r, c;
	printf("-----\n");
	printf("Array name: %s\n", (name));
	for (r = 0; r < nrows; r ++){
		for (c = 0; c < nrows; c ++){
			printf("    %Lg", (array[r][c]));
		}
		printf("\n");
	}
	printf("-----\n");
}

void PrintDoubleArray1D(long double **array, char *name, int nrows, int ncols){
	// Print a double 1D array.
	int r;
	printf("-----\n");
	printf("Array name: %s\n", (name));
	for (r = 0; r < nrows; r ++){
		printf("    %Lg", (array[r]));
	}
	printf("\n");
	printf("-----\n");
}

void PrintIntArray2D(int **array, char *name, int nrows, int ncols){
	// Print a int 2D array.
	int r, c;
	printf("-----\n");
	printf("Array name: %s\n", (name));
	for (r = 0; r < nrows; r ++){
		for (c = 0; c < nrows; c ++){
			printf("    %d", (array[r][c]));
		}
		printf("\n");
	}
	printf("-----\n");
}

void PrintIntArray1D(int **array, char *name, int nrows, int ncols){
	// Print a int 1D array.
	int r;
	printf("-----\n");
	printf("Array name: %s\n", (name));
	for (r = 0; r < nrows; r ++){
		printf("    %d", (array[r]));
	}
	printf("\n");
	printf("-----\n");
}

void PrintLongArray2D(long **array, char *name, int nrows, int ncols){
	// Print a long 2D array.
	int r, c;
	printf("-----\n");
	printf("Array name: %s\n", (name));
	for (r = 0; r < nrows; r ++){
		for (c = 0; c < nrows; c ++){
			printf("    %Ld", (array[r][c]));
		}
		printf("\n");
	}
	printf("-----\n");
}

void PrintLongArray1D(long **array, char *name, int nrows, int ncols){
	// Print a long 1D array.
	int r;
	printf("-----\n");
	printf("Array name: %s\n", (name));
	for (r = 0; r < nrows; r ++){
		printf("    %Ld", (array[r]));
	}
	printf("\n");
	printf("-----\n");
}