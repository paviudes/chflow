# Define the complex128_t data structure first.
ctypedef long double complex complex128_t

# Check is a 4 x 4 matrix is a valid density matrix.
cdef int IsState(complex128_t **choi, long double atol)

# Check if an input complex 4x4 matrix is completely positive.
cdef int IsPositive(complex128_t **choi, long double atol)

# Check if the trace of a complex 4x4 matrix is 1.
cdef int IsTraceOne(complex128_t **choi, long double atol)

# Check is a complex 4x4 matrix is Hermitian.
cdef int IsHermitian(complex128_t **choi, long double atol)

# Check if a given list of numbers is a normalized PDF.
cdef int IsPDF(long double *dist, int size, long double atol)

# Check is a matrix is a diagonal matrix or not.
cdef int IsDiagonal(long double **matrix, int size) nogil