# Given an exponent k and a probability distribution P(s), construct a new probability distribution Q(s) where
cdef int ConstructImportanceDistribution(long double* truedist, long double *impdist, int nelems, long double expo) nogil

# Given a probability distribution, construct its cumulative distribution
cdef int ConstructCumulative(long double* dist, long double *cumul, int nelems) nogil

# Sample a discrete probability distribution given its cumulative distribution
cdef int SampleCumulative(long double *cumulative, int size) nogil

# Search for an exponent k such that according to normalized distribution of P(s)^k, the probability of isincluded errors is within a desired window.
cdef long double PowerSearch(long double *distribution, int size, long double window[2], long double *searchin) nogil

# Compute k such that:
	# 	If Q(s) = P(s)^k/Z, where Z is the normalization, then
	# 	the variance of the average metric with respect to sampling Q(s) is smaller than the variance of the average metric with respect to P(s).
	# Identify the syndrome with the smallest (non zero) probability p*.
	# k = 1 + log(nstats)/log(p*) = 1 - log(nstats)/|log(p*)|
cdef long double PowerBound(long double *dist, int size, int nstats) nogil