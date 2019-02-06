import sys
import numpy as np
cimport numpy as np
try:
	import picos as pic
	import cvxopt as cvx
except:
	pass

from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

cdef extern from "math.h" nogil:
	long double fabsl(long double num)
	long double sqrtl(long double num)
	long double logl(long double num)
	long double powl(long double num)
	long double sinl(long double num)
	long double asinl(long double num)

cdef extern from "complex.h" nogil:
	long double creall(long double complex cnum)
	long double cimagl(long double complex cnum)
	long double cabsl(long double complex cnum)

cdef extern from "string.h" nogil:
	int strcmp(const char*, const char*)

cdef long double DiamondNorm(complex128_t **choi, char *chname) nogil:
	# Compute the diamond norm of the difference between an input Channel and another reference channel, which is by default, the identity channel.
	# The semidefinite program outlined in Sec. 4 of "Semidefinite Programs for Completely Bounded Norms", by John Watrous. Theory of Computing, Volume 5(11), pp. 217-238, 2009. http://www.theoryofcomputing.org/articles/v005a011/
	# See also: https://github.com/BBN-Q/matlab-diamond-norm/blob/master/src/dnorm.m
	cdef:
		int i, j
		long double dnorm = 0.0, angle = 0.0, pi = 3.14159265358979323846
	
	# printf("Function: DiamondNorm(choi, %s)\n", chname)

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
		# 1/2 							0 	0 	(cos(2 t) - i sin(2 t))/2
		# 0 							0 	0 	0
		# 0 							0 	0 	0
		# (cos(2 t) + i sin(2 t))/2 	0 	0 	1/2
		# and it's diamond norm is sin(t).
		# dnorm = fabsl(cimagl(choi[3][0]))
		if (creall(choi[3][0]) <= 0.0):
			angle = pi - asinl(2 * cimagl(choi[3][0]))
		else:
			angle = asinl(2 * cimagl(choi[3][0]))
		dnorm = cabsl(sinl(angle/<long double>2))
		# print("channel\n%s\n2 t = %g and dnorm = %g" % (np.array_str(choi, max_line_width = 150), (np.pi - np.arcsin(2 * np.imag(choi[3, 0]))), dnorm))
		# printf("Diamond norm of the Rotation channel is %Lg.\n", (dnorm))

	else:
		## =========> This block of code has a large Python interaction -- it can be very slow. <=========
		## =========> This block operates with GIL <======================================================
		with gil:
			choidiff = np.zeros((4, 4), dtype = np.complex128)
			for i in range(4):
				for j in range(4):
					choidiff[i, j] = choi[i][j]
			# Subtracting the choi matrix of the identity channel, from the input.
			# Choi matrix of the Identity channel is: 1/2 * ([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])
			choidiff[0, 0] = choidiff[0, 0] - 1/np.longdouble(2)
			choidiff[0, 3] = choidiff[0, 3] - 1/np.longdouble(2)
			choidiff[3, 0] = choidiff[3, 0] - 1/np.longdouble(2)
			choidiff[3, 3] = choidiff[3, 3] - 1/np.longdouble(2)
			## Taking a uniform superposition of a matrix and its Hermitian conjugate: (A + A^\dag)/2
			try:
				#### picos optimization problem
				prob = pic.Problem()
				# variables and parameters in the problem
				J = pic.new_param('J', cvx.matrix(choidiff))
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


cdef long double Entropy(complex128_t **choi) nogil:
	# Compute the Von-Neumann entropy of the input Choi matrix.
	# The idea is that a pure state (which corresponds to unitary channels) will have zero entropy while any mixed state which corresponds to a channel that does not preserve the input state, has finite entropy.
	# This function operates with GIL.
	cdef:
		int i, j
		long double entropy = 0
	with gil:
		choimat = np.zeros((4, 4), dtype = np.complex128)
		for i in range(4):
			for j in range(4):
				choimat[i, j] = choi[i][j]
		singvals = np.linalg.svd(choimat, compute_uv = 0)
		for i in range(4):
			entropy = entropy - creall(singvals[i]) * logl(fabsl(creall(singvals[i])))
	return entropy


cdef long double Infidelity(complex128_t **choi) nogil:
	# Compute the Infidelity between the input Choi matrix and the Choi matrix corresponding to the identity state.
	# Choi matrix for the identity is: 0.5 * [[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]]
	# Returns 1-fidelity
	cdef long double infidelity = 1 - (1/(<long double>2)) * <long double>(creall(choi[0][0] + choi[3][0] + choi[0][3] + choi[3][3]))
	return infidelity

cdef long double ProcessFidelity(complex128_t **choi) nogil:
	# Compute the average infidelity, given by: 1/6 * (4 - Tr(N)) where N is the process matrix describing a noise channel.
	# The above expression for infidelity can be simplified to 2/3 * entanglement infidelity.
	return (2/(<long double> 3) * Infidelity(choi))

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

#####################
# Converted to C.

long double Infidelity(complex128_t **choi){
	// Compute the Infidelity between the input Choi matrix and the Choi matrix corresponding to the identity state.
	// Choi matrix for the identity is: 0.5 * [[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]].
	// Returns 1-fidelity.
	infidelity = 1 - (1/((long double)2)) * (long double)(creall(choi[0][0] + choi[3][0] + choi[0][3] + choi[3][3]));
	return infidelity;
}

long double ProcessFidelity(complex128_t **choi){
	// Compute the average infidelity, given by: 1/6 * (4 - Tr(N)) where N is the process matrix describing a noise channel.
	// The above expression for infidelity can be simplified to 2/3 * entanglement infidelity.
	return (2/((long double)3) * Infidelity(choi));
}

long double FrobeniousNorm(complex128_t **choi){
	// Compute the Frobenious norm of the difference between the input Choi matrix and the Choi matrix corresponding to the Identity channel.
	// Frobenious of A is defined as: sqrtl(Trace(A^\dagger . A)).
	// https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm .
	int i, j;
	long double frobenious = 0;
	for (i = 0 ; i < 4; i ++){
		for (j = 0 ; j < 4; j ++){
			frobenious = frobenious + cabsl(choi[i][j]) * cabsl(choi[i][j]);
		}
	}
	frobenious = frobenious + 1 - (long double)(creall(choi[0][0] + choi[3][0] + choi[0][3] + choi[3][3]));
	frobenious = sqrtl(fabsl(frobenious));
	return frobenious;
}
#####################

cdef long double NonPauliness(complex128_t **choi) nogil:
	# Quantify the behaviour of a quantum channel by its difference from a Pauli channel
	# Convert the input Choi matrix to it's Chi-representation
	# Compute the ration between the  sum of offdiagonal entries to the sum of disgonal entries.
	# While computing the sums, consider the absolution values of the entries.
	# printf("function: NonPauliness\n")
	cdef:
		int i = 0, j = 0
		long double nonpauli = 0.0, atol = 10E-20
		complex128_t **chi = <complex128_t **>malloc(sizeof(complex128_t *) * 4)
	for i in range(4):
		chi[i] = <complex128_t *>malloc(sizeof(complex128_t) * 4)
		for j in range(4):
			chi[i][j] = 0.0 + 1j * 0.0
	ChoiToChi(choi, chi)
	nonpauli = 0.0
	for i in range(4):
		for j in range(4):
			if (not (i == j)):
				if (<long double>cabsl(chi[i][i]) * <long double>cabsl(chi[j][j]) >= atol):
					nonpauli = nonpauli + <long double>cabsl(chi[i][j]) * <long double>cabsl(chi[i][j])/(<long double>cabsl(chi[i][i]) * <long double>cabsl(chi[j][j]))
	# free memory for chi
	for i in range(4):
		free(<void *>chi[i])
	free(<void *>chi)
	# printf("non Pauliness = %Lf.\n", nonpauli)
	return nonpauli

cdef int ChoiToChi(complex128_t **choi, complex128_t **chi) nogil:
	# Convert from the Choi to the Chi representation.
	# Use the idea in ConvertRepresentations. Since this module cannot be imported into a Cython file, we essentially have to redefine the conversion here.
	# printf("function: ChoiToChi\n")
	cdef:
		int i = 0, j = 0, a = 0, b = 0
	with gil:
		invbasis = 0.25 * np.array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j, 1.+0.j, 0.+0.j,0.+0.j, 0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j,0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j,0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j,0.+0.j, 0.+1.j, 0.+0.j, 1.+0.j, 0.-0.j, 0.-0.j, 0.-0.j,0.-0.j, 1.+0.j, 0.-0.j, 0.-0.j, 0.-0.j, 0.-0.j, -1.+0.j,0.-0.j, 0.-0.j, 0.-0.j, 0.-0.j, -1.+0.j, 0.+0.j, 0.+0.j,0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,1.+0.j, 0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j,0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,1.+0.j, 0.-1.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,0.+0.j, 0.+0.j, 0.+1.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,0.+0.j, -0.-1.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,0.+1.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,0.-0.j, 0.-0.j, 0.-0.j, 0.-0.j, -1.+0.j, 0.-0.j, 0.-0.j,0.-0.j, 0.-0.j, 1.+0.j, 0.-0.j, 0.-0.j, 0.-0.j, 0.-0.j, -1.+0.j, 0.+0.j, -0.-1.j, 0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j,0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j,0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,0.-1.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+1.j,0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j,0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, -1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, -1.+0.j, 0.+0.j,0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j], dtype = np.complex128)
		Pauli = np.array([[[1, 0], [0, 1]],
						  [[0, 1], [1, 0]],
						  [[0, -1j], [1j, 0]],
						  [[1, 0], [0, -1]]], dtype = np.complex128)
		# basis = np.zeros((4, 4, 4, 4), dtype = np.complex128)
		# for i in range(4):
		# 	for j in range(4):
		# 		for a in range(4):
		# 			for b in range(4):
		# 				basis[i, j, a, b] = 0.5 * np.trace(np.dot(np.dot(np.dot(Pauli[i, :, :], Pauli[b, :, :]), Pauli[j, :, :]), Pauli[a, :, :]))
		npchoi = np.zeros((4, 4), dtype = np.complex128)
		for i in range(4):
			for j in range(4):
				npchoi[i, j] = choi[i][j]
		choivec = np.zeros((1, 16), dtype = np.complex128)
		for a in range(4):
			for b in range(4):
				choivec[0, a * 4 + b] = np.trace(np.dot(npchoi, np.kron(Pauli[a, :, :], np.transpose(Pauli[b, :, :]))))
		# npchi = np.reshape(np.dot(choivec, np.linalg.inv(np.reshape(basis, [16, 16]))), [4, 4])
		npchi = np.reshape(np.dot(choivec, np.reshape(invbasis, [16, 16])), [4, 4])
		for i in range(4):
			for j in range(4):
				chi[i][j] = <complex128_t>npchi[i, j]
	return 0

cdef int ComputeMetrics(long double *metvals, int nmetrics, char **metricsToCompute, complex128_t **choi, char *chname) nogil:
	# Compute all the metrics for a given channel, in the Choi matrix form
	cdef int m
	for m in range(nmetrics):
		if (strcmp(metricsToCompute[m], "frb") == 0):
			metvals[m] = FrobeniousNorm(choi)
		elif (strcmp(metricsToCompute[m], "infid") == 0):
			metvals[m] = Infidelity(choi)
		elif (strcmp(metricsToCompute[m], "processfidelity") == 0):
			metvals[m] = ProcessFidelity(choi)
		elif (strcmp(metricsToCompute[m], "dnorm") == 0):
			metvals[m] = DiamondNorm(choi, chname)
		elif (strcmp(metricsToCompute[m], "entropy") == 0):
			metvals[m] = Entropy(choi)
		elif (strcmp(metricsToCompute[m], "np1") == 0):
			metvals[m] = NonPauliness(choi)
		else:
			# Metrics that are not optimized in C cannot be evaluated at the Logical levels. For these, the metric value is simply zero.
			metvals[m] = 0
	return 0


######################################################################################
# Converted to C.

long double Infidelity(complex128_t **choi){
	// Compute the Infidelity between the input Choi matrix and the Choi matrix corresponding to the identity state.
	// Choi matrix for the identity is: 0.5 * [[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]].
	// Returns 1-fidelity.
	infidelity = 1 - (1/((long double)2)) * (long double)(creall(choi[0][0] + choi[3][0] + choi[0][3] + choi[3][3]));
	return infidelity;
}

long double ProcessFidelity(complex128_t **choi){
	// Compute the average infidelity, given by: 1/6 * (4 - Tr(N)) where N is the process matrix describing a noise channel.
	// The above expression for infidelity can be simplified to 2/3 * entanglement infidelity.
	return (2/((long double)3) * Infidelity(choi));
}

long double FrobeniousNorm(complex128_t **choi){
	// Compute the Frobenious norm of the difference between the input Choi matrix and the Choi matrix corresponding to the Identity channel.
	// Frobenious of A is defined as: sqrtl(Trace(A^\dagger . A)).
	// https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm .
	int i, j;
	long double frobenious = 0;
	for (i = 0 ; i < 4; i ++){
		for (j = 0 ; j < 4; j ++){
			frobenious = frobenious + cabsl(choi[i][j]) * cabsl(choi[i][j]);
		}
	}
	frobenious = frobenious + 1 - (long double)(creall(choi[0][0] + choi[3][0] + choi[0][3] + choi[3][3]));
	frobenious = sqrtl(fabsl(frobenious));
	return frobenious;
}


void ChoiToChi(complex128_t **choi, complex128_t **chi){
	// Convert from the Choi to the Chi representation.
	// Use the idea in ConvertRepresentations.
	// printf("function: ChoiToChi\n");
	int i, j, a, b;
	for (i = 0; i < 16; i ++){
		for (i = 0; i < 16; i ++){
			chi[j/4, j%4] += choi[i/4, i%4] * (consts->choichi)[i][j];
		}
	}
}

long double NonPauliness(complex128_t **choi){
	// Quantify the behaviour of a quantum channel by its difference from a Pauli channel.
	// Convert the input Choi matrix to it's Chi-representation.
	// Compute the ration between the  sum of offdiagonal entries to the sum of disgonal entries.
	// While computing the sums, consider the absolution values of the entries.
	// printf("function: NonPauliness\n");
	int i, j;
	long double nonpauli = 0.0, atol = 10E-20;
	complex128_t **chi = malloc(sizeof(complex128_t *) * 4);
	for (i = 0 ; i < 4; i ++){
		chi[i] = malloc(sizeof(complex128_t) * 4);
		for (j = 0 ; j < 4; j ++){
			chi[i][j] = 0.0 + 0.0 * I;
		}
	}
	ChoiToChi(choi, chi);
	nonpauli = 0.0;
	for (i = 0 ; i < 4; i ++){
		for (j = 0 ; j < 4; j ++){
			if (i != j){
				if ((long double)cabsl(chi[i][i]) * (long double)cabsl(chi[j][j]) >= atol){
					nonpauli = nonpauli + (long double)cabsl(chi[i][j]) * (long double)cabsl(chi[i][j])/((long double)cabsl(chi[i][i]) * (long double)cabsl(chi[j][j]));
				}
			}
		}
	}
	// Free memory for chi.
	for (i = 0 ; i < 4; i ++){
		free(chi[i]);
	}
	free(chi);
	// printf("non Pauliness = %Lf.\n", nonpauli);
	return nonpauli;
}

void ComputeMetrics(long double *metvals, int nmetrics, char **metricsToCompute, complex128_t **choi, char *chname, constants *consts){
	// Compute all the metrics for a given channel, in the Choi matrix form.
	int m;
	for (m = 0; m < nmetrics; m ++){
		if (strcmp(metricsToCompute[m], "frb") == 0){
			metvals[m] = FrobeniousNorm(choi);
		}
		else if (strcmp(metricsToCompute[m], "infid") == 0){
			metvals[m] = Infidelity(choi);
		}
		else if (strcmp(metricsToCompute[m], "processfidelity") == 0){
			metvals[m] = ProcessFidelity(choi);
		}
		else if (strcmp(metricsToCompute[m], "np1") == 0){
			metvals[m] = NonPauliness(choi, consts);
		}
		else{
			// Metrics that are not optimized in C cannot be evaluated at the Logical levels. For these, the metric value is simply zero.
			metvals[m] = 0;
		}
	}
}