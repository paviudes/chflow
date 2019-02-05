import os
import sys
import time
import numpy as np
import multiprocessing as mp
try:
	from scipy import optimize as opt
except Exception:
	sys.stderr.write("\033[91mThere was some problem loading SciPy. So, optimizations cannot be done.\n\033[0m")
from define import fnames as fn

# cpdef double Obj(np.ndarray[np.float_t, ndim = 1] optvars, np.ndarray[np.float_t, ndim = 3] logErr, np.ndarray[np.float_t, ndim = 2] dists):
def Obj(optvars, logErr, dists):
	# Objective fucntion for the different between the measured logical error rate and ansatz predicted logical error rate, on the log scale.
	# ansatz: f_(i,l,e) = c_(i,l) (p_e)^(a_i * d_(i,l))
	# g(p, c) = sum_(i,l,e) [ log (f_(i,l,e)) - log(c_(i,l)) - a_i * d_(i,l) * log(p_e)) ]^2
	# where i is used to label the different databases, a_i 
	#		l for the levels, c_(i,l) is the number of uncorrectable errors in database i at level l.
	# 		e for the different physical channels, p_e is the physical noise rate of the channel e.
	# 		d_(i,l) = fixed value that depends on the distance of the concatenated code.
	# print("X\n%s" % (np.array_str(optvars)))
	# cdef:
	# 	int i, k, l, ndb = logErr.shape[0], nchannels = logErr.shape[1], nlevels = logErr.shape[2]
	# 	double obj = 0.0, ansatz = 0.0, atol = 10E-30
	ndb = logErr.shape[0]
	nchannels = logErr.shape[1]
	nlevels = logErr.shape[2]
	obj = 0.0
	ansatz = 0.0
	atol = 10E-30
	for k in range(ndb):
		for i in range(nchannels):
			for l in range(nlevels):
				if ((logErr[k, i, l] < (1 - atol)) and (logErr[k, i, l] > atol)):
					ansatz = optvars[nchannels + k * nlevels + l] + optvars[nchannels + ndb * nlevels + k] * dists[k, l] * optvars[i]
					obj = obj + np.power(np.log(logErr[k, i, l]) - ansatz, 2.0)
	return obj


# cpdef np.ndarray[np.float_t, ndim = 1] Jacob(np.ndarray[np.float_t, ndim = 1] optvars, np.ndarray[np.float_t, ndim = 3] logErr, np.ndarray[np.float_t, ndim = 2] dists):
def Jacob(optvars, logErr, dists):
	# Jacobian for the objective function Obj().
	# cdef:
	# 	int i, k, l, ndb = logErr.shape[0], nchannels = logErr.shape[1], nlevels = logErr.shape[2]
	# 	double ansatz = 0.0, fval = 0.0, atol = 10E-30
	# 	np.ndarray[np.float_t, ndim = 1] jacob = np.zeros(optvars.shape[0], dtype = np.float)
	ansatz = 0.0
	fval = 0.0
	atol = 10E-30
	jacob = np.zeros(optvars.shape[0], dtype = np.float)
	ndb = logErr.shape[0]
	nchannels = logErr.shape[1]
	nlevels = logErr.shape[2]
	for k in range(ndb):
		for i in range(nchannels):
			for l in range(nlevels):
				if ((logErr[k, i, l] < (1 - atol)) and (logErr[k, i, l] > atol)):
					ansatz = optvars[nchannels + k * nlevels + l] + optvars[nchannels + ndb * nlevels + k] * dists[k, l] * optvars[i]
					fval = np.log(logErr[k, i, l]) - ansatz
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
	nlevels = min([dbses[i].levels for i in range(ndb)])
	logerr = np.zeros((ndb, nchans, nlevels), dtype = np.float)
	phyerr = np.load(fn.PhysicalErrorRates(dbses[0], pmet))
	for k in range(ndb):
		logerr[k, :, :] = np.load(fn.LogicalErrorRates(dbses[k], lmet))[:, 1:]
	# For every dataset and concatenation level, store the distance of the code that was used to error correct.
	dists = np.zeros((ndb, nlevels), dtype = np.float)
	for i in range(ndb):
		dists[i, 0] = dbses[i].eccs[0].D
		for l in range(1, nlevels):
			dists[i, l] = dists[i, l - 1] * dbses[i].eccs[l].D
		dists[i, :] = (dists[i, :] - 1)/np.float(2)
	guess = np.zeros(nchans + len(dbses) * nlevels + len(dbses), dtype = np.float)
	# limits = np.zeros((nchans + len(dbses) * nlevels + len(dbses), 2), dtype = np.float)
	limits = [[0, 0] for __ in range(nchans + len(dbses) * nlevels + len(dbses))]
	# Bounds and initial guess for physical noise rates
	for i in range(nchans):
		limits[i] = (None, 0)
		guess[i] = np.log(phyerr[i])
	# Bounds and initial guess for the combinatorial factors
	for i in range(ndb):
		limits[nchans + i * nlevels] = [0, np.log(dbses[i].eccs[0].N) + np.log(dbses[i].eccs[0].N - 1) - np.log(np.float(2))]
		for l in range(1, nlevels):
			limits[nchans + i * nlevels + l][0] = 0
			limits[nchans + i * nlevels + l][1] = limits[nchans + i * nlevels + l - 1][1] + (np.log(dbses[i].eccs[l].N) + np.log(dbses[i].eccs[l].N - 1) - np.log(np.float(2)))
		# guess[nchans + i * nlevels + l] = np.random.randint(0, high = limits[nchans + i * nlevels + l][1])
	# Bounds and initial guess for the exponent constants
	for i in range(ndb):
		# limits[nchans + ndb * nlevels + i, 0] = 0
		limits[nchans + ndb * nlevels + i] = (0, None)
		guess[nchans + ndb * nlevels + i] = 1

	print("guess\n%s\n%s" % (np.array_str(guess), np.array_str(dists)))
	print("limits")
	print(limits)

	# Objective function and Jacobian
	# objective = (lambda optvars: Obj(optvars, logerr, dists))
	# jacobian = (lambda optvars: Jacob(optvars, logerr, dists))
	
	# start = 0.0
	# fin = 0.0
	# pfit = np.zeros(nchans + ndb * nlevels + ndb, dtype = np.float)
	#######
	start = time.time()
	result = opt.minimize(Obj, guess, jac = Jacob, args = (logerr, dists), bounds = limits, method = 'L-BFGS-B', options = {'disp':True, 'maxiter':5000})
	pfit = result.x
	fin = time.time()
	#######

	print("pfit\n%s\nmean = %g, min = %g, max = %g." % (np.array_str(np.exp(pfit[:nchans])), np.mean(np.exp(pfit[:nchans])), np.min(np.exp(pfit[:nchans])), np.max(np.exp(pfit[:nchans]))))

	if (result.success == True):
		print("\033[2mOptimization completed successfully in %d seconds. Objective function at minima is %.2e.\033[0m" % ((fin - start), result.fun))
	else:
		print("\033[2mOptimization terminated because of\n%s\nin %d seconds. Objective function at minima is %.2e.\033[0m" % (result.message, (fin - start), result.fun))

	# Write the computed physical noise rates into a file.
	for k in range(ndb):
		np.save(fn.FitPhysRates(dbses[k], lmet), np.exp(pfit[:nchans]))
		np.save(fn.FitWtEnums(dbses[k], lmet), np.concatenate(([1], np.exp(pfit[(nchans + k * nlevels):(nchans + (k + 1) * nlevels)]))))
		np.save(fn.FitExpo(dbses[k], lmet), pfit[nchans + ndb * nlevels + k])
	return (fin - start)


# ================================================================================================================

# multiple parameters fit

def PartialObj(start, stop, comp, phychans, logerr, select, subtotals):
	# Compute the function whose minimum is reached for the compression matrix
	# f(M) = sum_(i,j) |||f_i> - |f_j>||^2 exp(- || M . (E_i> - |E_j>) ||^2)
	# Compute only the part of the sum that ranges from i=start to i=stop.
	compmat = np.reshape(comp, [select, 12])
	square = np.dot(np.transpose(compmat), compmat)
	func = 0
	for i in xrange(start, stop):
		# print("obj i = %d" % (i))
		for j in xrange(i + 1, phychans.shape[0]):
			diff = np.reshape(phychans[i, :, 1:] - phychans[j, :, 1:], [12, 1])
			# print("obj diff\n%s" % (np.array_str(diff)))
			func = func + np.power(logerr[i] - logerr[j], 2.0)/(logerr[i] * logerr[j]) * np.exp((-1) * np.squeeze(np.dot(np.dot(np.transpose(diff), square), diff)))
	# print("obj = %g" % (func))
	# print('Shape of variable is', np.shape(comp))
	subtotals.put([start, stop, func])
	return None

def CompressObj(comp, phychans, logerr, select):
	# Compute the function whose minimum is reached for the compression matrix
	# f(M) = sum_(i,j) |||f_i> - |f_j>||^2 exp(- || M . (E_i> - |E_j>) ||^2)
	# Use the PartialObj function to compute the sum in chunks
	ncpu = 8
	nproc = min(ncpu, mp.cpu_count())
	chunk = int(np.ceil(phychans.shape[0]/np.float(nproc)))
	print("obj cores = %d" % (nproc))
	processes = []
	subtotals = mp.Queue()
	for i in range(nproc):
		processes.append(mp.Process(target = PartialObj, args = (i * chunk, min(phychans.shape[0], (i + 1) * chunk), comp, phychans, logerr, select, subtotals)))
	for i in range(nproc):
		processes[i].start()
	for i in range(nproc):
		processes[i].join()

	obj = 0
	while (not subtotals.empty()):
		(start, stop, fval) = subtotals.get()
		obj = obj + fval
		print("%d to %d done." % (start, stop))

	# compmat = np.reshape(comp, [select, 12])
	# square = np.dot(np.transpose(compmat), compmat)
	# obj = 0
	# for i in xrange(phychans.shape[0]):
	# 	# print("obj i = %d" % (i))
	# 	for j in xrange(i + 1, phychans.shape[0]):
	# 		diff = np.reshape(phychans[i, :, 1:] - phychans[j, :, 1:], [12, 1])
	# 		# print("obj diff\n%s" % (np.array_str(diff)))
	# 		obj = obj + np.power(logerr[i] - logerr[j], 2.0)/(logerr[i] * logerr[j]) * np.exp((-1) * np.squeeze(np.dot(np.dot(np.transpose(diff), square), diff)))
	# print("obj = %g" % (obj))
	# print('Shape of variable is', np.shape(comp))
	return obj

# @autojit
# def CompressObj(comp, phychans, logerr, select):
# 	# Compute the function whose minimum is reached for the compression matrix
# 	# f(M) = sum_(i,j) |||f_i> - |f_j>||^2 exp(- || M . (E_i> - |E_j>) ||^2)
# 	compmat = np.reshape(comp, [select, 12])
# 	square = np.dot(np.transpose(compmat), compmat)
# 	func = 0
# 	for i in xrange(phychans.shape[0]):
# 		# print("obj i = %d" % (i))
# 		for j in xrange(i + 1, phychans.shape[0]):
# 			diff = np.reshape(phychans[i, :, 1:] - phychans[j, :, 1:], [12, 1])
# 			# print("obj diff\n%s" % (np.array_str(diff)))
# 			func = func + np.power(logerr[i] - logerr[j], 2.0)/(logerr[i] * logerr[j]) * np.exp((-1) * np.squeeze(np.dot(np.dot(np.transpose(diff), square), diff)))
# 	# print("obj = %g" % (func))
# 	# print('Shape of variable is', np.shape(comp))
# 	return func

def PartialJacob(start, stop, comp, phychans, logerr, select, subjacobs):
	# Compute the function whose minimum is reached for the compression matrix
	# f(M) = sum_(i,j) |||f_i> - |f_j>||^2 exp(- || M . (E_i> - |E_j>) ||^2)
	# Compute only the part of the sum that ranges from i=start to i=stop.
	# compmat = np.reshape(comp, [select, 12])
	# square = np.dot(np.transpose(compmat), compmat)
	partjacob = np.zeros(comp.shape[0], dtype = np.float)
	for i in xrange(start, stop):
		# print("obj i = %d" % (i))
		for j in xrange(i + 1, phychans.shape[0]):
			diff = np.reshape(phychans[i, :, 1:] - phychans[j, :, 1:], [12, 1])
			# expo = (-1) * np.squeeze(np.dot(np.dot(np.transpose(diff), square), diff))
			expo = 0.0
			for a in range(12 * 12):
				for k in range(select):
					expo = expo + (-1) * comp[k * 12 + a/12] * comp[k * 12 + np.mod(a, 12)] * diff[a/12] * diff[np.mod(a, 12)]
			# partjacob = partjacob + np.power(logerr[i] - logerr[j], 2.0) * np.exp(expo) * expo * np.reshape(np.dot(compmat, np.dot(diff, np.transpose(diff))), comp.shape[0])
			for a in range(select * 12):
				for k in range(12):
					partjacob[a] = partjacob[a] + np.power(logerr[i] - logerr[j], 2.0)/(logerr[i] * logerr[j]) * np.exp(expo) * expo * comp[a/12 * 12 + k] * diff[k] * diff[np.mod(a, 12)]
	# print("obj = %g" % (func))
	# print('Shape of variable is', np.shape(comp))
	subjacobs.put([start, stop, partjacob])
	return None

def CompressJacob(comp, phychans, logerr, select):
	# Compute the function whose minimum is reached for the compression matrix
	# f(M) = sum_(i,j) |||f_i> - |f_j>||^2 exp(- || M . (E_i> - |E_j>) ||^2)
	# Use the PartialObj function to compute the sum in chunks
	ncpu = 8
	nproc = min(ncpu, mp.cpu_count())
	chunk = int(np.ceil(phychans.shape[0]/np.float(nproc)))
	print("jacob cores = %d" % (nproc))
	processes = []
	subjacobs = mp.Queue()
	for i in range(nproc):
		processes.append(mp.Process(target = PartialJacob, args = (i * chunk, min(phychans.shape[0], (i + 1) * chunk), comp, phychans, logerr, select, subjacobs)))
	for i in range(nproc):
		processes[i].start()
	for i in range(nproc):
		processes[i].join()

	jacob = np.zeros(comp.shape[0], dtype = np.float)
	while (not subjacobs.empty()):
		(start, stop, sjac) = subjacobs.get()
		print("%d to %d done." % (start, stop))
		jacob = jacob + sjac

	# compmat = np.reshape(comp, [select, 12])
	# square = np.dot(np.transpose(compmat), compmat)
	# obj = 0
	# for i in xrange(phychans.shape[0]):
	# 	# print("obj i = %d" % (i))
	# 	for j in xrange(i + 1, phychans.shape[0]):
	# 		diff = np.reshape(phychans[i, :, 1:] - phychans[j, :, 1:], [12, 1])
	# 		# print("obj diff\n%s" % (np.array_str(diff)))
	# 		obj = obj + np.power(logerr[i] - logerr[j], 2.0)/(logerr[i] * logerr[j]) * np.exp((-1) * np.squeeze(np.dot(np.dot(np.transpose(diff), square), diff)))
	# print("obj = %g" % (obj))
	# print('Shape of variable is', np.shape(comp))
	return jacob

# def CompressJacob(comp, phychans, logerr, select):
# 	# Compute the Jacobian of the objective function in CompressObj(...).
# 	# Take the derivative with respect to the compression matrix M.
# 	# compmat = np.reshape(comp, [select, 12])
# 	jacob = np.zeros(comp.shape[0], dtype = np.float)
# 	# square = np.dot(np.transpose(compmat), compmat)
# 	# print("Compression matrix\n%s" % (np.array_str(comp)))
# 	for i in xrange(phychans.shape[0]):
# 		# print("i = %d" % (i))
# 		for j in xrange(i + 1, phychans.shape[0]):
# 			diff = np.reshape(phychans[i, :, 1:] - phychans[j, :, 1:], [12, 1])
# 			# print("jacob diff\n%s" % (np.array_str(diff)))
# 			# expo = (-1) * np.squeeze(np.dot(np.dot(np.transpose(diff), square), diff))
# 			# print "Expo"
# 			# print expo
# 			# print("expo = %g, np.reshape(np.dot(compmat, np.dot(diff, np.transpose(diff))), comp.shape[0])\n%s" % (expo, np.array_str(np.reshape(np.dot(compmat, np.dot(diff, np.transpose(diff))), comp.shape[0]))))
# 			expo = 0.0
# 			for a in range(12 * 12):
# 				for k in range(select):
# 					expo = expo + (-1) * comp[k * 12 + a/12] * comp[k * 12 + np.mod(a, 12)] * diff[a/12] * diff[np.mod(a, 12)]
# 			for a in range(select * 12):
# 				for k in range(12):
# 					jacob[a] = jacob[a] + np.power(logerr[i] - logerr[j], 2.0)/(logerr[i] * logerr[j]) * np.exp(expo) * expo * comp[a/12 * 12 + k] * diff[k] * diff[np.mod(a, 12)]
# 			# jacob = jacob + np.power(logerr[i] - logerr[j], 2.0) * np.exp(expo) * expo * np.reshape(np.dot(compmat, np.dot(diff, np.transpose(diff))), comp.shape[0])
# 	# print('Shape of variable is', np.shape(comp))
# 	# print('Shape of gradient is', np.shape(jacob))
# 	# print("jacob\n%s" % (np.array_str(jacob)))
# 	return jacob

def Compress(dbs, lmet, level, ncomp):
	# Compute the compression matrix for a database of physical channels with respect to a concatenation level.
	# Use scipy's minimize function to minimize the objective function
	# f(M) = sum_(i,j) |||f_i> - |f_j>||^2 exp(- || M . (E_i> - |E_j>) ||^2)
	# over all matrices of size c x n where c is the compressed set of features and n is the total number of features, to describe a quantum channel.
	# =====
	physical = np.zeros((dbs.channels, 4, 4), dtype = np.longdouble)
	for i in range(dbs.channels):
		physical[i, :, :] = np.load(fn.PhysicalChannel(dbs, dbs.available[i, :np.int(dbs.available.shape[1] - 1)], loc = "storage"))[np.int(dbs.available[i, dbs.available.shape[1] - 1]), :, :]
	
	if (not (os.path.isfile(fn.CompressionMatrix(dbs, lmet, level)))):
		logerr = np.load(fn.LogicalErrorRates(dbs, lmet))[:, level]
		# The compression matrix is real (c x 12) matrix where c is the input.
		# There are no bounds on the entries of the elements.
		# Initialize the compression matrix to a random real matrix.
		# print("logerr\n%s" % (np.array_str(logerr)))
		guess = np.random.rand(ncomp * 12)
		# print("guess\n%s" % (np.array_str(guess)))
		limits = [(-1.0, 1.0) for i in range(guess.shape[0])]
		#######
		# jac = CompressJacob, 
		start = time.time()
		result = opt.minimize(CompressObj, guess, args = (physical, logerr, ncomp), bounds = limits, method = 'TNC', options = {'disp':True, 'maxiter':10})
		comp = np.reshape(result.x, [ncomp, -1])
		fin = time.time()
		#######

		if (result.success == True):
			print("\033[2mOptimization completed successfully in %d seconds. Objective function at minima is %.2e.\033[0m" % ((fin - start), result.fun))
		else:
			print("\033[2mOptimization terminated because of\n%s\nin %d seconds. Objective function at minima is %.2e.\033[0m" % (result.message, (fin - start), result.fun))

		print("Compression matrix\n%s" % (np.array_str(comp)))

		# Write the compression matrix on to a file
		np.save(fn.CompressionMatrix(dbs, lmet, level), comp)

	else:
		comp = np.load(fn.CompressionMatrix(dbs, lmet, level))

	# Compute the compressed set of parameters for the physical channels and save them to a file
	compressed = np.squeeze(np.einsum('ikl,jk->ijl', np.reshape(physical[:, :, 1:], [dbs.channels, -1, 1]), comp))
	np.save(fn.CompressedParams(dbs, lmet, level), compressed)
	return None
