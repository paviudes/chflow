ctypedef long double complex complex128_t

from constants cimport constants_t
from qecc cimport qecc_t
from memory cimport simul_t


cdef int Performance(qecc_t **qecc, simul_t **sims, constants_t *consts)
