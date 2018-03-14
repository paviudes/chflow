ctypedef long double complex complex128_t

cdef struct constants:
	complex128_t pauli[4][2][2]
	long double atol
	int nclifford
ctypedef constants constants_t

cdef int InitConstants(constants_t *consts)
cdef int FreeConstants(constants_t *consts)