import numpy as np
import chanreps as crep

def HermitianConjugate(mat):
	# Return the Hermitian conjugate of a matrix
	return np.conjugate(np.transpose(mat))


def IsUnital(channel, rep = "process"):
	# Test if a channel is unital, given its process matrix
	## Unital channel always maps the identity onto itself.
	# The first row of the process matrix must be: 1, 0, 0, 0
	if (np.allclose(crep.ConvertRepresentations(channel, rep, "process")[0, :], np.array([1, 0, 0, 0], dtype = np.longdouble))):
		return 1
	return 0


def IsPositiveDefinite(densmat, atol = 10E-8):
	# check if a matrix is positive definite
	eigVals = np.linalg.eigvals(densmat)
	isComplexEigVals = np.any(np.abs(np.imag(eigVals)) > atol)
	isNegetiveEigVals = np.any(np.real(eigVals) < (-1) * atol)
	isPosDef = ((isComplexEigVals + isNegetiveEigVals) == 0)
	return isPosDef


def IsQuantumChannel(channel, rep = "process"):
	# Determine if the input quantum channel, in the Choi matrix formalism is a valid CPTP map.
	# Check if the choi matrix satisfies: unit trace, positive definiteness and hermiticity.
	densmat = np.copy(channel)
	if (not (rep == "choi")):
		densmat = crep.ConvertRepresentations(channel, rep, "choi")
		rep = "choi"
	conditions = np.zeros(3, dtype = np.int8)
	# Unit trace condition
	conditions[0] = np.allclose(np.trace(densmat), 1, atol = 10E-8)
	# Positive semidefinite condition
	conditions[1] = IsPositiveDefinite(densmat)
	# Hermitian condition
	conditions[2] = np.allclose(HermitianConjugate(densmat), densmat, atol = 1E-08)
	if (np.prod(conditions, dtype = np.int8) == 0):
		print("! E = \n%s\nis not a valid quantum channel because" % (np.array_str(channel, max_line_width = 100, precision = "%.3e")))
		if (conditions[0] == 0):
			print("\tX Tr(E(rho)) is not 1.")
		if (conditions[1] == 0):
			print("\tX E is not completely positive.")
		if (conditions[2] == 0):
			print("\tX E(rho) is not Hermitian.")
	else:
		if (IsUnital(densmat, rep) == 1):
			print("_/ E is an unital CPTP map.")
		else:
			print("_/ E is a non-unital CPTP map.")
	return None