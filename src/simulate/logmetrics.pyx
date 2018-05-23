#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False
import sys
import numpy as np
cimport numpy as np
try:
	import picos as pic
	import cvxopt as cvx
except:
	pass

cdef extern from "math.h" nogil:
	long double fabsl(long double num)
	long double sqrtl(long double num)
	long double logl(long double num)
	long double powl(long double num)

cdef extern from "complex.h" nogil:
	long double creall(long double complex cnum)
	long double cimagl(long double complex cnum)
	long double cabsl(long double complex cnum)

cdef extern from "string.h" nogil:
	int strcmp(const char*, const char*)

cdef long double DiamondNorm(complex128_t **choi, char *chname):
	# Compute the diamond norm of the difference between an input Channel and another reference channel, which is by default, the identity channel.
	# The semidefinite program outlined in Sec. 4 of "Semidefinite Programs for Completely Bounded Norms", by John Watrous. Theory of Computing, Volume 5(11), pp. 217-238, 2009. http://www.theoryofcomputing.org/articles/v005a011/
	# See also: https://github.com/BBN-Q/matlab-diamond-norm/blob/master/src/dnorm.m
	cdef:
		int i, j
		long double dnorm = 0
	# For some known types of channels, the Diamond norm can be computed efficiently
	# 1. Depolarizing channel
	if (strcmp(chname, "dp") == 0):
		# The choi matrix of the channel is in the form
		# 1/2-p/3	0		0	1/2-(2 p)/3
		# 0			p/3		0	0
		# 0			0		p/3	0
		# 1/2-(2 p)/3	0		0	1/2-p/3
		# and it's Diamond norm is p, in other words, it is 3 * Choi[1,1]/3
		dnorm = 3 * <long double>(creall(choi[1][1]))

	# 2. Rotation about the Z axis
	elif (strcmp(chname, "rtz") == 0):
		# The Choi matrix of the Rotation channel is in the form
		# 1 						0 	0 	cos(t) - i sin(t)
		# 0 						0 	0 	0
		# 0 						0 	0 	0
		# cos(t) + i sin(t) 	0 	0 	1
		# and it's diamond norm is sin(t).
		dnorm = fabsl(cimagl(choi[3][0]))
		# print("Diamond norm of the Rotation channel\n%s\nis %g." % (np.array_str(choi), dnorm))

	else:
		## =========> This block of code has a large Python interaction -- it can be very slow. <=========
		choi = np.zeros((4, 4), dtype = np.complex128)
		for i in range(4):
			for j in range(4):
				choi[i, j] = choi[i][j]
		# Subtracting the choi matrix of the identity channel, from the input.
		# Choi matrix of the Identity channel is: 1/2 * ([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])
		choi[0, 0] = choi[0, 0] - 1/np.longdouble(2)
		choi[0, 3] = choi[0, 3] - 1/np.longdouble(2)
		choi[3, 0] = choi[3, 0] - 1/np.longdouble(2)
		choi[3, 3] = choi[3, 3] - 1/np.longdouble(2)
		## Taking a uniform superposition of a matrix and its Hermitian conjugate: (A + A^\dag)/2
		try:
			#### picos optimization problem
			prob = pic.Problem()
			# variables and parameters in the problem
			J = pic.new_param('J', cvx.matrix(choi))
			rho = prob.add_variable('rho', (2, 2), 'hermitian')
			W = prob.add_variable('W', (4, 4), 'hermitian')
			# objective function (maximize the hilbert schmidt inner product -- denoted by '|'. Here A|B means trace(A^\dagger * B))
			prob.set_objective('max', J | W)
			# adding the constraints
			prob.add_constraint(W >> 0)
			prob.add_constraint(rho >> 0)
			prob.add_constraint(('I' | rho) == 1)
			prob.add_constraint((W - ((rho & 0) // (0 & rho))) << 0)
			# solving the problem
			sol = prob.solve(verbose = 0, maxit = 100)
			dnorm = sol['obj']*2
		except:
			sys.stderr.write("\033[91mSomething went wrong in the Diamond norm computation. Setting the Diamond norm to 0.\n\033[0m")
	return dnorm


cdef long double Entropy(complex128_t **choi):
	# Compute the Von-Neumann entropy of the input Choi matrix.
	# The idea is that a pure state (which corresponds to unitary channels) will have zero entropy while any mixed state which corresponds to a channel that does not preserve the input state, has finiste entropy.
	cdef:
		int i, j
		np.ndarray[np.complex128_t, ndim = 2, mode = 'c'] choi = np.zeros((4, 4), dtype = np.complex128)
	for i in range(4):
		for j in range(4):
			choi[i, j] = choi[i][j]
	cdef:
		long double entropy = 0
		np.ndarray[np.complex128_t, ndim = 2, mode = 'c'] singvals = np.linalg.svd(choi, compute_uv = 0)
	for i in range(4):
		entropy = entropy - creall(singvals[i]) * logl(fabsl(creall(singvals[i])))
	return entropy


cdef long double Fidelity(complex128_t **choi) nogil:
	# Compute the Fidelity between the input Choi matrix and the Choi matrix corresponding to the identity state.
	# Choi matrix for the identity is: 0.5 * [[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]]
	# Returns 1-fidelity
	cdef long double infidelity = 1 - (1/(<long double>2)) * <long double>(creall(choi[0][0] + choi[3][0] + choi[0][3] + choi[3][3]))
	return infidelity

cdef long double ProcessFidelity(complex128_t **choi) nogil:
	# Compute the average infidelity, given by: 1/6 * (4 - Tr(N)) where N is the process matrix describing a noise channel.
	# The above expression for infidelity can be simplified to 2/3 * entanglement infidelity.
	return (2/(<long double> 3) * Fidelity(choi))

cdef long double FrobeniousNorm(complex128_t **choi) nogil:
	# Compute the Frobenious norm of the difference between the input Choi matrix and the Choi matrix corresponding to the Identity channel
	# Frobenious of A is defined as: sqrtl(Trace(A^\dagger . A))
	# https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
	cdef:
		int i, j
		long double frobenious = 0
	for i in range(4):
		for j in range(4):
			frobenious = frobenious + cabsl(choi[i][j]) * cabsl(choi[i][j])
	frobenious = frobenious + 1 - <long double>(creall(choi[0][0] + choi[3][0] + choi[0][3] + choi[3][3]))
	frobenious = sqrtl(fabsl(frobenious))
	return frobenious


cdef int ComputeMetrics(long double *metvals, int nmetrics, char **metricsToCompute, complex128_t **choi, char *chname) nogil:
	# Compute all the metrics for a given channel, in the Choi matrix form
	cdef int m
	for m in range(nmetrics):
		if (strcmp(metricsToCompute[m], "frb") == 0):
			metvals[m] = FrobeniousNorm(choi)
		elif (strcmp(metricsToCompute[m], "fidelity") == 0):
			metvals[m] = Fidelity(choi)
		elif (strcmp(metricsToCompute[m], "processfidelity") == 0):
			metvals[m] = ProcessFidelity(choi)
		elif (strcmp(metricsToCompute[m], "dnorm") == 0):
			metvals[m] = DiamondNorm(choi, chname)
		elif (strcmp(metricsToCompute[m], "entropy") == 0):
			metvals[m] = Entropy(choi)
		else:
			# Metrics that are not optimized in C cannot be evaluated at the Logical levels. For these, the metric value is simply zero.
			metvals[m] = 0
	return 0