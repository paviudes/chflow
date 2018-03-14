#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False
from libc.stdlib cimport malloc, free

cdef extern from "stdlib.h" nogil:
	int rand()
	long double drand48()
	void srand48(long seedval)

cdef extern from "math.h" nogil:
	long double powl(long double base, long double expo)
	long double fabsl(long double num)


cdef int ConstructImportanceDistribution(long double* truedist, long double *impdist, int nelems, long double expo) nogil:
	# Given an exponent k and a probability distribution P(s), construct a new probability distribution Q(s) where
	# Q(s) = P(s)^k/(sum_s P(s)^k), i.e, a normalized power-law scaled version of P(s).
	cdef:
		int s
		long double norm = 0.0
	for s in range(nelems):
		impdist[s] = powl(truedist[s], expo)
		norm = norm + impdist[s]
	for s in range(nelems):
		impdist[s] = impdist[s]/norm
	return 0

cdef int ConstructCumulative(long double* dist, long double *cumul, int nelems) nogil:
	# Given a probability distribution, construct its cumulative distribution
	cdef:
		int s
	cumul[0] = dist[0]
	for s in range(1, nelems):
		cumul[s] = cumul[s - 1] + dist[s]
	return 0

cdef int SampleCumulative(long double *cumulative, int size) nogil:
	# Sample a discrete probability distribution given its cumulative distribution
	# Draw a uniform random number, u, in [0,1]. Determine the interval of the cumulative distribution in which u lies.
	# http://ieeexplore.ieee.org/document/92917/
	# Additions: if frozen = -1, continue with the sampling as described above.
	# if frozen = 0, then exclude 0 from the sample.
	cdef:
		int i = 0
		long double urand = drand48()
	for i in range(size):
		if (urand < cumulative[i]):
			return i
	return i


cdef int WhereInWindow(long double number, long double *window) nogil:
	# determine is a number is within, below or above a window.
	# If it is within, return 0, if it is above, return 1 and if below, return -1.
	cdef int where = 0
	if (number < window[0]):
		return -1
	else:
		if (number > window[1]):
			return 1
	return 0


cdef long double PowerSearch(long double *dist, int size, long double window[2], long double *searchin) nogil:
	# Search for an exponent k such that according to normalized distribution of P(s)^k, the probability of isincluded errors is within a desired window.
	# the array searchin provides the exponent k in that: k = (searchin[0] + searchin[1])/2.
	# One of the three cases can occur for the distribution P(s)^k where k = (searchin[0] + searchin[1])/2.
		# 1. Probability of isincluded errors is below the window -- in this case, we return the value of the function on a new searchin wondow, given by: [searchin[0], k]. [Recursion]
		# 2. Probability of isincluded errors is within the window -- in this case we stop after returning k.
		# 3. Probability of isincluded errors is above the window -- in this case, we return the value of the function on a new searchin wondow, given by: [k, searchin[1]]. [Recursion]
	# The weight-1 errors are X_i, Z_i, Y_i for i = 1 to 7 and their syndromes are: 56, 24, 40, 8, 48, 16, 32, 7, 3, 5, 1, 6, 2, 4, 63, 54, 45, 36, 27, 18, 9, respectively.
	cdef:
		int position = 0, exit = 0, isFirst = 0
		long double norm = 0.0, pinc = 0.0, atol = 10E-10, exponent = (searchin[0] + searchin[1])/(<long double>2)
		int *incl = <int *>malloc(sizeof(int) * size)
		long double *powerdist = <long double *>malloc(sizeof(long double) * size)
	# incl is an indicator array for un-correctable errors.
	incl[0] = 0
	for i in range(1, size):
		incl[i] = 1
	
	# If searchin is NULL, then it is set to [0, 1] and we will assume it is the first call.
	if (searchin == NULL):
		isFirst = 1
		searchin = <long double *> malloc(sizeof(long double) * 2)
		searchin[0] = 0
		searchin[1] = 1

	while (exit == 0):
		if (fabsl(searchin[0] - searchin[1]) < atol):
			exit = 1
		else:
			ConstructImportanceDistribution(dist, powerdist, size, exponent)
			for i in range(size):
				if (incl[i] == 1):
					pinc = pinc + powerdist[i]
			if ((isFirst == 1) and (pinc > window[1])):
				# If the probability of included errors is above than the window to begin with, then there is no need of importance sampling.
				exit = 1
			else:
				position = WhereInWindow(pinc, window)
				if (position == 0):
					return exponent
				elif (position == -1):
					# If pinc is too small, reduce the exponent, i.e, take a higher root.
					searchin[1] = exponent
					exponent = (searchin[0] + searchin[1])/(<long double>2)
					return PowerSearch(dist, size, window, searchin)
				elif (position == 1):
					# If pinc is too large, increase the exponent, (make the exponent closer to one) so that it goes closer to its real value.
					searchin[0] = exponent
					exponent = (searchin[0] + searchin[1])/(<long double>2)
					return PowerSearch(dist, size, window, searchin)
				else:
					pass
	
	# free local variables
	free(<void *>incl)
	free(<void *>powerdist)
	free(<void *>searchin)
	return exponent