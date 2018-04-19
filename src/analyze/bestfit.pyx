#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False
import sys
import time
import numpy as np
cimport numpy as np
try:
	from scipy import optimize as opt
except Exception:
	sys.stderr.write("\033[91mThere was some problem loading SciPy. So, optimizations cannot be done.\n\033[0m")
from define import fnames as fn

cpdef double Obj(np.ndarray[np.float_t, ndim = 1] optvars, np.ndarray[np.float_t, ndim = 3] logErr, np.ndarray[np.float_t, ndim = 2] dists):
	# Objective fucntion for the different between the measured logical error rate and ansatz predicted logical error rate, on the log scale.
	# g(p, c) = sum_(i,l,e) [ log (f_(i,l,e)) - log(c_(i,l)) - a_i * d_(i,l) * log(p_e)) ]^2
	# where i is used to label the different databases, a_i 
	#		l for the levels, c_(i,l) is the number of uncorrectable errors in database i at level l.
	# 		e for the different physical channels, p_e is the physical noise rate of the channel e.
	# 		d_(i,l) = fixed value that depends on the distance of the concatenated code.
	cdef:
		int i, k, l, ndb = logErr.shape[0], nchannels = logErr.shape[1], nlevels = logErr.shape[2]
		double obj = 0.0, ansatz = 0.0
	for k in range(ndb):
		for i in range(nchannels):
			for l in range(nlevels):
				ansatz = optvars[nchannels + k * nlevels + l] + optvars[nchannels + ndb * nlevels + k] * dists[k, l] * optvars[i]
				obj = obj + np.power(logErr[k, i, l] - ansatz, 2.0)
	return obj


cpdef np.ndarray[np.float_t, ndim = 1] Jacob(np.ndarray[np.float_t, ndim = 1] optvars, np.ndarray[np.float_t, ndim = 3] logErr, np.ndarray[np.float_t, ndim = 2] dists):
	# Jacobian for the objective function Obj3.
	cdef:
		int i, k, l, ndb = logErr.shape[0], nchannels = logErr.shape[1], nlevels = logErr.shape[2]
		double ansatz = 0.0, fval = 0.0
		np.ndarray[np.float_t, ndim = 1] jacob = np.zeros(optvars.shape[0], dtype = np.float)
	for k in range(ndb):
		for i in range(nchannels):
			for l in range(nlevels):
				ansatz = optvars[nchannels + k * nlevels + l] + optvars[nchannels + ndb * nlevels + k] * dists[k, l] * optvars[i]
				fval = logErr[k, i, l] - ansatz
				# Derivatives with respect to each physical noise rate
				jacob[i] = jacob[i] + 2 * fval * (optvars[nchannels + ndb * nlevels + k] * dists[k, l]) * (-1)
				# Derivatives with respect to each level coefficient
				jacob[nchannels + k * nlevels + l] = jacob[nchannels + k * nlevels + l] + 2 * fval * (-1)
				# Derivatives with respect to each distance exponent
				jacob[nchannels + ndb * nlevels + k] = jacob[nchannels + ndb * nlevels + k] + 2 * fval * dists[k, l] * optvars[i] * (-1)
	return jacob


def FitPhysErr(pmet, lmet, *dbses):
	# Compute the physical noise rates for all channels in a database.
	# The noise rates are obtained by assuming an ansatz that related the observed logical error rates to the physical noise rate by a polynomial function.
	# We then perform least square fit to obtain the unknown parameters of the ansatz.
	# See Eqs. 3 and 4 of https://arxiv.org/abs/1711.04736 .
	# The input must be one or more databases.
	# If multiple databases are provided, it will be assumed that the underlying physical channels in all of the databases are the same.
	# Optimization variable: {p for every channel} + {c for every level in every database} + {alpha for every database}
	# Create the list of logical error rates to be used by the least squares fit. Create a 3D array L where
	# L[k, i, l] = Logical error rate of the k-th database, for physical channel i and concatenated level l.
	ndb = len(dbses)
	nchans = dbses[0].channels
	nlevels = 1 + min([dbses[i].levels for i in range(ndb)])
	atol = 10E-30
	logerr = np.zeros((ndb, nchans, nlevels), dtype = np.float)
	phyerr = np.load(fn.PhysicalErrorRates(dbses[0], pmet))
	for k in range(ndb):
		logerr[k, :, :] = np.load(fn.LogicalErrorRates(dbses[i], lmet))
		for i in range(nchans):
			for l in range(nlevels):
				if ((logerr[k, i, l] < (1 - atol)) and (logerr[k, i, l] > atol)):
					logerr[k, i, l] = np.log(logerr[k, i, l])
				else:
					logerr[k, i, l] = -100
	# For every dataset and concatenation level, store the distance of the code that was used to error correct.
	dists = np.zeros((ndb, nlevels), dtype = np.float)
	for i in range(ndb):
		dists[i, 0] = 1
		for l in range(1, nlevels):
			dists[i, l] = dbses[i].eccs[l - 1].D
	guess = np.zeros(nchans + len(dbses) * nlevels + len(dbses), dtype = np.float)
	limits = np.zeros((nchans + len(dbses) * nlevels + len(dbses), 2), dtype = np.float)
	# Bounds and initial guess for physical noise rates
	for i in range(nchans):
		limits[i, 0] = -5
		limits[i, 1] = 0
		guess[i] = phyerr[i]
	# Bounds and initial guess for the combinatorial factors
	for i in range(ndb):
		for l in range(1, nlevels):
			limits[nchans + i * nlevels + l, 1] = l * np.log(dbses[i].eccs[l - 1].N) + np.log(dbses[i].eccs[l - 1].N - 1) - np.log(np.float(2))
			# guess[nchans + i * nlevels + l] = np.random.randint(0, high = limits[nchans + i * nlevels + l, 1])
	# Bounds and initial guess for the exponent constants
	for i in range(ndb):
		limits[nchans + ndb * nlevels + i, 0] = 0
		limits[nchans + ndb * nlevels + i, 1] = 2
		guess[nchans + ndb * nlevels + i] = np.random.rand() * limits[nchans + ndb * nlevels + i, 1]

	# Objective function and Jacobian
	objective = (lambda optvars: Obj(optvars, logerr, dists))
	jacobian = (lambda optvars: Jacob(optvars, logerr, dists))
	
	start = 0.0
	fin = 0.0
	pfit = np.zeros(nchans + ndb * nlevels + ndb, dtype = np.float)
	#######
	start = time.time()
	result = opt.minimize(objective, guess, jac = jacobian, bounds = limits, method = 'L-BFGS-B', options = {'disp':True, 'maxiter':5000})
	pfit = result.x
	fin = time.time()
	#######

	if (result.success == True):
		print("\033[2mOptimization completed successfully in %d seconds. Objective function at minima is %.2e.\033[0m" % ((fin - start), result.fun))
	else:
		print("\033[2mOptimization terminated because of\n%s\nin %d seconds. Objective function at minima is %.2e.\033[0m" % (result.message, (fin - start), result.fun))

	# Write the computed physical noise rates into a file.
	for k in range(ndb):
		np.save(fn.FitPhysRates(dbses[k], lmet), np.exp(pfit[:nchans]))
		np.save(fn.FitWtEnums(dbses[k], lmet), np.exp(pfit[(nchans + k * nlevels):(nchans + (k + 1) * nlevels)]))
		np.save(fn.FitExpo(dbses[k], lmet), pfit[nchans + ndb * nlevels + k])
	return (fin - start)