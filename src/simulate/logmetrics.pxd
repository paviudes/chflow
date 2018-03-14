ctypedef long double complex complex128_t

# Compute the Fidelity between the input Choi matrix and the Choi matrix corresponding to the identity state
cdef long double Fidelity(complex128_t **choi) nogil

# Compute the average infidelity, given by: 1/6 * (4 - Tr(N)) where N is the process matrix describing a noise channel.
cdef long double ProcessFidelity(complex128_t **choi) nogil

# Compute the Frobenious norm of the difference between the input Choi matrix and the Choi matrix corresponding to the Identity channel
cdef long double FrobeniousNorm(complex128_t **choi) nogil

# Compute the Von-Neumann entropy of the input Choi matrix.
cdef long double Entropy(complex128_t **choi)

# Compute the diamond norm of the difference between an input Channel and another reference channel, which is by default, the identity channel.
cdef long double DiamondNorm(complex128_t **choi, char *chname)

# Compute all the metrics for a given channel, in the Choi matrix form
cdef int ComputeMetrics(long double *metvals, int nmetrics, char **metricsToCompute, complex128_t **choichannel, char *chname) nogil