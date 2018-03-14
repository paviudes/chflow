ctypedef long double complex complex128_t

cdef struct quantumecc:
	int N
	int K
	int D
	int nlogs
	int nstabs
	int ***algebra
	int **projector
	int ***action
	complex128_t **phases
ctypedef quantumecc qecc_t

# Allocate memory for the elements of the quantum error correcting code.
cdef int InitQECC(qecc_t *qecc, int nclifford)
# Allocate memory allocated to the elements of the quantum error correcting code.
cdef int FreeQECC(qecc_t *qecc, int nclifford)

# Compute the effective logical channel, when error correction is applied over a set of input physical channels
from constants cimport constants_t
from memory cimport simul_t
cdef int SingleShotErrorCorrection(int level, int isPauli, int frame, qecc_t *qecc, simul_t *sim, constants_t *consts) nogil
