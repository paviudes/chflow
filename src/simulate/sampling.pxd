# Given an exponent k and a probability distribution P(s), construct a new probability distribution Q(s) where
cdef int ConstructImportanceDistribution(long double* truedist, long double *impdist, int nelems, long double expo) nogil

# Given a probability distribution, construct its cumulative distribution
cdef int ConstructCumulative(long double* dist, long double *cumul, int nelems) nogil

# Sample a discrete probability distribution given its cumulative distribution
cdef int SampleCumulative(long double *cumulative, int size) nogil

# Search for an exponent k such that according to normalized distribution of P(s)^k, the probability of isincluded errors is within a desired window.
cdef long double PowerSearch(long double *distribution, int size, long double window[2], long double *searchin) nogil