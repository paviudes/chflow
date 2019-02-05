ctypedef long double complex complex128_t

cdef int PrintComplexArray2D(complex128_t **array, char *name, int nrows, int ncols) nogil
cdef int PrintDoubleArray2D(long double **array, char *name, int nrows, int ncols) nogil
cdef int PrintDoubleArray1D(long double *array, char *name, int size) nogil
cdef int PrintIntArray2D(int **array, char *name, int nrows, int ncols) nogil
cdef int PrintIntArray1D(int *array, char *name, int size) nogil
cdef int PrintLongArray2D(long **array, char *name, int nrows, int ncols) nogil
cdef int PrintLongArray1D(long *array, char *name, int size) nogil