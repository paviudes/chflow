try:
	import numpy as np
	from scipy import linalg as linalg
except:
	pass
import globalvars as gv
import chanreps as crep

def HermitianConjugate(mat):
	# Return the Hermitian conjugate of a matrix
	return np.conjugate(np.transpose(mat))

def RandomHermitian(dim, method = "qr"):
	# Generate a random hermitian matrix of given dimensions.
	randMat = (np.random.standard_normal(size = (dim, dim)) + 1j * np.random.standard_normal(size = (dim, dim)))
	if (method == "qr"):
		randH = (np.identity(dim) + prox * randMat +  HermitianConjugate(np.identity(dim) + prox * randMat))/np.longdouble(2)
	elif (method == "exp"):
		randH = (randMat +  HermitianConjugate(randMat))/np.longdouble(2)
	elif (method == "haar"):
		randH = (randMat +  HermitianConjugate(randMat))/np.longdouble(2)
	else:
		print("Method subscribed for random Hermitian production is unknown: \"%s\"." % (method))
		randH = np.identity(dim)
	return randH


def RandomUnitary(prox, dim, method = "qr", randH = None):
	# Generate a random unitary matrix of given dimensions and of a certain proximity to identity.
	randMat = (np.random.standard_normal(size = (dim, dim)) + 1j * np.random.standard_normal(size = (dim, dim)))
	if (method == "qr"):
		if (randH is None):
			randH = (np.identity(dim) + prox * randMat +  HermitianConjugate(np.identity(dim) + prox * randMat))/np.longdouble(2)
		(randU, __) = linalg.qr(randH)
	elif (method == "exp"):
		if (randH is None):
			randH = (randMat +  HermitianConjugate(randMat))/np.longdouble(2)
		randU = linalg.expm(1j * prox * randH)
	elif (method == "haar"):
		if (randH is None):
			randH = (randMat +  HermitianConjugate(randMat))/np.longdouble(2)
		(randU, __) = linalg.qr(randH)
	elif (method == "hyps"):
		# The random unitary is expressed as an exponential of a random Hermitian matrix.
		# The Hermitian matrix can be decomposed in the Pauli basis with real coefficients.
		# Let these coefficients be {c1, ..., cN} where N is the total number of Pauli matrices of the given dimension.
		# In order to ensure that the Unitary matrix has a certain distance, say some function f(p), from the Identity, we will ensure that \sum_i |c_i|^2 = p.
		# So, this reduces to sampling over points in a hypersphere of radius p.
		# The desired Hermitian matrix is simply: \sum_i (xi * Pi) where Pi is the i-th Pauli matrix in the basis.
		if (randH is None):
			paulibasis = np.load("codedata/paulibasis_3qubits.npy")
			nelems = np.power(dim, 2, dtype = np.int)
			hypersphere = np.zeros(nelems, dtype = np.longdouble)
			#### This part of the code is to ensure that there are only a few (determined by the number of distinct classes) degress of freedom in the random hermitian matrix.
			## Here we force the Hermitian matrix to have non-zero equal contributions from the X,Y and Z Pauli matrices.
			## If we assign the same index to two Pauli matrices in the linear combination, we essentially cut a degree of freedom.
			## Alternatively we can also set one of the indices to -1, in which case that component is set to zero.
			forcedClassification = np.arange(nelems, dtype = np.int)
			HyperSphereSampling(hypersphere, nelems, center = 0.0, radius = prox, classification = forcedClassification)
			# print("hypersphere\n%s\nsum = %g. (desired values = %g)" % (np.array_str(hypersphere, max_line_width = 150, precision = 3), np.sum(np.power(hypersphere, 2.0, dtype = np.longdouble), dtype = np.longdouble), prox))
			randH = np.zeros((dim, dim), dtype = np.complex128)
			for i in range(nelems):
				randH = randH + hypersphere[i] * paulibasis[i, :, :]
			# print("Random Hermitian\n%s" % (np.array_str(randH, max_line_width = 150, precision = 3)))
		randU = linalg.expm(1j * randH)
		# print("Random Unitary\n%s" % (np.array_str(randU, max_line_width = 150, precision = 3)))
	else:
		print("Method subscribed for random unitary production is unknown: \"%s\"." % (method))
	return randU


def RandomPauliChannel(perr):
	# A Pauli channel is defined by: E(R) = p_I R + p_X X R X + p_Y Y R Y + p_Z Z R Z.
	# Generate a random Pauli channel with a specified distance from the identity.
	# All notions of distances are identical for Pauli channels, it is essentially: 1 - p_I.
	# Generate 3 random numbers x, y, z, such that: x^2 + y^2 + z^2 = r^2, where r^2 = perr.
	# Finally, the Krauss operators are: sqrt(1 - perr) I, x X, y Y, z Z.
	pauliamps = np.zeros(3, dtype = np.longdouble)
	HyperSphereSampling(pauliamps, 3, center = 0.0, radius = np.sqrt(perr), classification = None)
	# print("p_I = %g, p_X = %g, p_Y = %g and p_Z = %g." % (1 - np.power(pauliamps[0], 2.0) - np.power(pauliamps[1], 2.0) - np.power(pauliamps[2], 2.0), np.power(pauliamps[0], 2.0), np.power(pauliamps[1], 2.0), np.power(pauliamps[2], 2.0)))
	krops = np.zeros((4, 2, 2), dtype = np.complex128)
	krops[0, :, :] = np.sqrt(1 - perr) * gv.Pauli[0, :, :]
	for i in range(3):
		krops[i + 1, :, :] = pauliamps[i] * gv.Pauli[i + 1, :, :]
	return krops


def RandomCPTP(dist, meth):
	# Generate a random CPTP map by the specified method.
	# Available methods are:
		# 1. Exponential of random Hermitian
		# 2. Diagonalization of a random Hermitian
		# 3. Haar random unitary
		# 4. Hypersphere sampling for generating a random Hermitian and then exponetial to determine unitary.
		# 5. Generating random Pauli amplitudes for X, Y and Z errors, given a probability for no error.
	availmethods = ["exp", "qr", "haar", "hyps", "pauli"]
	method = availmethods[meth]
	if (method == "pauli"):
		krauss = RandomPauliChannel(dist)
	else:
		randU = RandomUnitary(dist, 8, method, randH = None)
		krauss = crep.ConvertRepresentations(randU, 'stine', 'krauss')
	return krauss



def HyperSphereSampling(surface, npoints, center = 0.0, radius = 1.0, classification = None):
	# Sample points on a hypersphere of given radius and center.
	# We use the algorithm outlined in https://dl.acm.org/citation.cfm?id=377946.
	## Sketch of the algorithm:
		# 1. Generate n points {x1, ..., xn} distributed according to the Normal distribution with mean = center of the sphere (in our case: (0, 0, 0...)).
		# 2. For each xi, do xi -> xi/(z/p) where z = sqrt(\sum_i (xi)^2).
	## There is an additional option to reduce the degrees of freedom on the surface by ensuring that the points are concentrated into various classes.
	## See also: https://math.stackexchange.com/questions/132933/generating-3-times-3-unitary-matrices-close-to-the-identity
	if (classification is None):
		classification = np.arange(npoints, dtype = np.int)
	normalization = 0
	normalvariates = np.random.normal(loc = center, scale = 1.0, size = np.unique(classification[np.where(classification > -1)]).shape[0])
	for i in range(npoints):
		if (classification[i] == -1):
			surface[i] = 0.0
		else:
			surface[i] = normalvariates[classification[i]]
		normalization = normalization + np.power(surface[i], 2.0, dtype = np.longdouble)
	for i in range(npoints):
		surface[i] = surface[i] * radius/np.sqrt(normalization, dtype = np.longdouble)
	return None