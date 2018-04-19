import numpy as np
cimport numpy as np

cpdef double Obj(np.ndarray[np.float_t, ndim = 1] optvars, np.ndarray[np.float_t, ndim = 3] logErr, np.ndarray[np.float_t, ndim = 2] dists)
cpdef np.ndarray[np.float_t, ndim = 1] Jacob(np.ndarray[np.float_t, ndim = 1] optvars, np.ndarray[np.float_t, ndim = 3] logErr, np.ndarray[np.float_t, ndim = 2] dists)