import os
import sys
try:
	import numpy as np
except:
	pass
import globalvars as gv

class QuantumErrorCorrectingCode():
	"""
	All the necessary information about a quantum error correcting code that is required for error correction and the channel simulations are given here.
	The class contains properties that can be defined for any error correcting code. Few details that are specific to an error correcting code will be speficied in a file.
	The input file must be formatted as follows.
	#####################
	code ID
	code name
	N (number of physical qubits)
	K (number of logical qubits)
	D (code distance)
	#####################
	"""
	def __init__(self, name):
		# Read the specifications of the Quantum error correcting code, from a file
		self.name = "Unspecified"
		self.N = 1
		self.K = 1
		self.D = 1
		self.S = None
		self.L = None
		self.T = None
		self.syndsigns = None
		self.lookup = None
		self.normalizer = None
		self.normphases = None
		self.conjugations = None
		self.defnfile = ("./../code/%s.txt" % (name))
		eof = 0
		with open(self.defnfile, "r") as fp:
			while (eof == 0):
				line = fp.readline()
				if (not (line)):
					eof = 1
				else:
					line = line.strip("\n").strip(" ")
					if (not (line[0] == "#")):
						if (line == "name"):
							(self.name, nkd) = fp.readline().strip("\n").strip(" ").split(" ")
							(self.N, self.K, self.D) = map(np.int8, nkd.split(","))
						elif (line == "stabilizer"):
							# Read the next N - K lines.
							self.S = PauliToOperator(fp, self.N, self.N - self.K)
						elif (line == "logical"):
							# Read the next 2 K lines.
							self.L = PauliSymbolToOperator(fp, self.N, 2 * self.K)
						elif (line == "pureerror"):
							# Read the next N - K lines.
							self.T = PauliSymbolToOperator(fp, self.N, self.N - self.K)
						else:
							pass
	################################

def Load(qecc):
	# Load all the essential information for quantum error correction
	# If the logicals or pure errors are not specified in the input file, we must construct them by Gaussian elimination.
	# Reconstructs the entire Pauli basis.
	# Only works for CSS codes
	if (qecc.S is None):
		print("\033[2mInsufficient information -- stabilizer generators not provided.\033[0m")
	else:
		if ((qecc.L is None) or (qecc.T is None)):
			if (qecc.L is None):
				ComputeLogicals(qecc)
				ComputePureErrors(qecc)
			else:
				if (IsCanonicalBasis(qecc.S, qecc.L, qecc.T, verbose = 1) == 0):
					ComputeLogicals(qecc)
				ComputePureErrors(qecc)
		else:
			if (IsCanonicalBasis(qecc.S, qecc.L, qecc.T, verbose = 1) == 0):
				ComputeLogicals(qecc)
				ComputePureErrors(qecc)
	
	# Signs in front of each stabilizer element in the syndrome projectors
	ConstructSyndProjSigns(qecc)
	# Elements in the cosets of the normalizer and their phases
	ConstructNormalizer(qecc)
	# Transformations between Pauli operators by Clifford conjugations
	PauliCliffordConjugations(qecc)
	return None

# Converting between representations of Pauli operators

def PauliToOperator(fp, nqubits, nlines):
	# Convert the Pauli string format to an operator matrix with the encoding: I -> 0, X -> 1, Y -> 2, Z -> 3.
	# Read from a file whose file pointer is given
	strenc = {"I":0, "X":1, "Y":2, "Z":3}
	sympenc = np.array([[0, 3], [1, 2]], dtype = np.int8)
	operators = np.zeros((nlines, nqubits), dtype = np.int8)
	for i in range(nlines):
		opstr = fp.readline().strip("\n").strip(" ").split(" ")
		if (len(opstr) == nqubits):
			# string encoding
			for j in range(nqubits):
				operators[i, j] = strenc[opstr[j]]
		else:
			# symplectic encoding
			for j in range(nqubits):
				operators[i, j] = sympenc[np.int8(opstr[j]), np.int8(opstr[nqubits + j])]
	return operators


def PauliOperatorToSymbol(ops):
	# Convert a set of Pauli operators in the encoding I -> 0, X -> 1, Y -> 2, Z -> 3 to a string format with I, X, Y and Z characters.
	encoding = ["I", "X", "Y", "Z"]
	opstr = ["" for i in range(ops.shape[0])]
	for i in range(ops.shape[0]):
		for j in range(ops.shape[1]):
			opstr[i] = opstr[i] + ("%s " % (encoding[ops[i, j]]))
	return opstr

def SymplecticToOperator(sympvec):
	# convert a symplectic form to an operator form
	encoding = np.array([[0, 3], [1, 2]], dtype = np.int8)
	op = np.zeros(sympvec.shape[0]/2, dtype = np.int8)
	for i in range(op.shape[0]):
		op[i] = encoding[sympvec[i], sympvec[op.shape[0] + i]]
	return op

def OperatorToSymplectic(pauliop):
	# Convert the operator in the encoding I -> 0, X -> 1, Y -> 2, Z -> 3 to its symplectic form.
	encoding = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype = np.int8)
	sympvec = np.zeros(2 * pauliop.shape[0], dtype = np.int8)
	for i in range(pauliop.shape[0]):
		(sympvec[i], sympvec[pauliop.shape[0] + i]) = (encoding[pauliop[i], 0], encoding[pauliop[i], 1])
	return sympvec

def ConvertToSympectic(gens):
	# Convert a list of stabilizer generators to X and Z symplectic matrices.
	sympmat = np.zeros((gens.shape[0], 2 * gens.shape[1]), dtype = np.int8)
	encoding = np.array(([[0, 0], [1, 0], [1, 1], [0, 1]]), dtype = np.int8)
	for i in range(gens.shape[0]):
		for j in range(gens.shape[1]):
			sympmat[i, [j, j + gens.shape[1]]] = encoding[gens[i, j], :]
	return sympmat

def ConvertToOperator(sympmat):
	# Convert a matrix of symplectic vectors to operators in the encoding: I --> 0, X --> 1, Y --> 2, Z --> 3.
	encoding = np.array([[0, 3], [1, 2]], dtype = np.int8)
	nq = sympmat.shape[1]/2
	pauliops = np.zeros((sympmat.shape[0], nq), dtype = np.int8)
	for i in range(sympmat.shape[0]):
		for j in range(nq):
			pauliops[i, j] = encoding[sympmat[i, j], sympmat[i, nq + j]]
	return pauliops

def PauliOperatorToMatrix(ops):
	# Convert a Pauli operator string to the explicit matrix form.
	pmat = gv.Pauli[ops[0], :, :]
	for i in range(1, ops.shape[0]):
		pmat = np.kron(pmat, gv.Pauli[ops[i], :, :])
	return pmat

# Commutation relations

def IsCommuting(pauli1, pauli2):
	# Determine if two of Pauli operators commute.
	lie = np.array([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]], dtype = np.int8)
	return (np.mod(np.sum(lie[pauli1, pauli2]), 2) == 0)

def SymplecticProduct(sympvec1, sympvec2):
	# return the symplectic product of two vectors.
	nq = sympvec1.shape[0]/2
	return np.mod(np.dot(sympvec1[:nq], sympvec2[nq:]) + np.dot(sympvec1[nq:], sympvec2[:nq]), 2)

# Constructing the canonical Pauli basis

def NullSpace(mat):
	# Compute a basis for the Null space of the input (binary) matrix, with respect to the symplectic product.
	# Given a matrix M = (HX | HZ), we need to compute vectors v such that: M P v = 0 where P = (0 I \\ I 0) where I is an Identity matrix.
	# Hence we need to find the kernel of (HZ | HX) over GF_2.
	# If we have a matrix A over GF_2, the kernel of A consists of vectors v such that: A.v = 0 (mod 2).
	# If we row reduce A to the format [I|P] where I is an identity matrix, then the kernel of A are columns of the matrix [P \\ I].
	nq = mat.shape[1]/2
	cols = np.arange(mat.shape[1], dtype = np.int8)
	
	# Reflect the matrix about its center column: HX <--> HZ
	for i in range(nq):
		mat[:, [i, nq + i]] = mat[:, [nq + i, i]]
	
	for i in range(mat.shape[0]):
		if (mat[i, i] == 0):
			# look for a row below that has 1 in the i-th column.
			for j in range(i + 1, mat.shape[0]):
				if (mat[j, i] == 1):
					# swap rows j and i and break
					mat[[i, j], :] = mat[[j, i], :]
					break
		if (mat[i, i] == 0):
			# look for a column to the right (up to n) that has a 1 in the i-th row.
			for j in range(i + 1, mat.shape[1]):
				if (mat[i, j] == 1):
					# swap columns i with j and columns (mat.shape[0] + i) with (mat.shape[0] + j).
					mat[:, [i, j]] = mat[:, [j, i]]
					# record the column swaps so that they can be undone at the end.
					cols[[i, j]] = cols[[j, i]]
					break
		# Row reduce the matrix
		for j in range(mat.shape[0]):
			if (not (i == j)):
				if (mat[j, i] == 1):
					mat[j, :] = np.mod(mat[i, :] + mat[j, :], 2)

	# Decude the Null space vectors and undo the column permutations
	null = np.hstack((np.transpose(mat[:, mat.shape[0]:]), np.identity((mat.shape[1] - mat.shape[0]), dtype = np.int8)))[:, np.argsort(cols)]
	return null


def NullTest(vspace, kernel):
	# test if all the vectors in the null space are orthogonal (with respect to the symplectic product) to the original space.
	print("Null space test with\nvspace\n%s\nkernel\n%s" % (np.array_str(ConvertToOperator(vspace)), np.array_str(ConvertToOperator(kernel))))
	product = np.zeros((vspace.shape[0], kernel.shape[0]), dtype = np.int8)
	for i in range(vspace.shape[0]):
		for j in range(kernel.shape[0]):
			product[i, j] = SymplecticProduct(vspace[i, :], kernel[j, :])
	print("product\n%s" % (np.array_str(product)))
	return None


def ComputeLogicals(qecc):
	# Given the stabilizer, compute a canonical basis for the normalizer consisting of the stabilizer and logical operators.
	# We will use the Symplectic Gram Schmidt Orthogonialization method mentioned in arXiv: 0903.5526v1.
	normalizer = NullSpace(ConvertToSympectic(qecc.S))
	
	used = np.zeros(normalizer.shape[0], dtype = np.int8)
	# If logs[i] = 0, the i-th normalizer is is a stabilizer.
	# If logs[i] = l, the i-th normalizer is the logical operator Z_l.
	# If logs[i] = -l, the i-th normalizer if the logical operator X_l.
	logs = np.zeros(normalizer.shape[0], dtype = np.int8)
	nlogs = 1
	for i in range(normalizer.shape[0]):
		if (used[i] == 0):
			used[i] = 1
			for j in range(normalizer.shape[0]):
				if (used[j] == 0):
					if (SymplecticProduct(normalizer[i, :], normalizer[j, :]) == 1):
						logs[i] = nlogs
						logs[j] = (-1) * nlogs
						nlogs = nlogs + 1
						used[j] = 1
						for k in range(normalizer.shape[0]):
							if (used[k] == 0):
								if (SymplecticProduct(normalizer[k, :], normalizer[j, :])):
									normalizer[k, :] = np.mod(normalizer[k, :] + normalizer[i, :], 2)
								if (SymplecticProduct(normalizer[k, :], normalizer[i, :])):
									normalizer[k, :] = np.mod(normalizer[k, :] + normalizer[j, :], 2)
						break
	qecc.S = ConvertToOperator(np.squeeze(normalizer[np.nonzero(abs(logs) == 0), :]))
	qecc.L = np.zeros((2 * qecc.K, qecc.N), dtype = np.int8)
	for i in range(nlogs - 1):
		qecc.L[[i, qecc.K + i], :] = ConvertToOperator(np.squeeze(normalizer[np.nonzero(abs(logs) == (i + 1)), :]))
	return None


def ComputePureErrors(qecc):
	# Compute the generators for the pure errors, given the stabilizer and logical generators.
	# The i-th pure error must anticommute with the i-th stabilizer generator and commute with all other operators including pure errors.
	stabgens = ConvertToSympectic(qecc.S)
	loggens = ConvertToSympectic(qecc.L)
	puregens = np.zeros((qecc.N - qecc.K, 2 * qecc.N), dtype = np.int8)
	for i in range(qecc.N - qecc.K):
		sols = NullSpace(np.vstack((loggens, stabgens[np.arange(qecc.N - qecc.K)!=i])))
		# Check which of the elements from the solutions anticommute with the i-th stabilizer.
		# Set that element to be the i-th pure error and add it to the list of constraints
		for j in range(sols.shape[0]):
			if (SymplecticProduct(sols[j, :], stabgens[i, :]) == 1):
				break
		puregens[i, :] = sols[j, :]
	# Fixing the commutation relations between pure errors.
	# 1. If Ti anti commutes with Sj, for j not equal to i, then: Ti -> Ti * Tj
	# 2. If Ti anti commutes with Lj, then: Ti -> Ti * Mj where Mj is the logical operator conjugate to Lj
	# 3. If Ti anti commutes with Tj, then Ti -> Ti * Sj
	for i in range(qecc.N - qecc.K):
		for j in range(qecc.N - qecc.K):
			if (not (i == j)):
				if (SymplecticProduct(puregens[i, :], stabgens[j, :]) == 1):
					puregens[i, :] = np.mod(puregens[i, :] + puregens[j, :], 2)
	for i in range(qecc.N - qecc.K):
		for j in range(2 * qecc.K):
			if (SymplecticProduct(puregens[i, :], loggens[j, :]) == 1):
				puregens[i, :] = np.mod(puregens[i, :] + loggens[(j + qecc.K) % (2 * qecc.K), :], 2)
	for i in range(qecc.N - qecc.K):
		for j in range(qecc.N - qecc.K):
			if (SymplecticProduct(puregens[i, :], puregens[j, :]) == 1):
				puregens[i, :] = np.mod(puregens[i, :] + stabgens[j, :], 2)
	qecc.T = ConvertToOperator(puregens)
	return None


def IsCanonicalBasis(S, L, T, verbose = 1):
	# test if a basis is a canonical basis and display the commutation relations
	(nq, nl) = (S.shape[1], S.shape[1] - S.shape[0])
	canonical = np.vstack((ConvertToSympectic(S), ConvertToSympectic(L), ConvertToSympectic(T)))
	commutation = np.zeros((canonical.shape[0], canonical.shape[0]), dtype = np.int8)
	if (verbose == 1):
		print("\033[2m")
		print("N = %d, K = %d" % (nq, nl))
		tab = 20
		stabops = PauliOperatorToSymbol(S)
		print(("{:<%d} {:<%d}" % (tab, tab)).format("Stabilizer generators", "%s" % (stabops[0])))
		for i in range(1, nq - nl):
			print(("{:<%d} {:<%d}" % (tab, tab)).format("", "%s" % (stabops[i])))
		logops = PauliOperatorToSymbol(L)
		print(("{:<%d} {:<%d}" % (tab, tab)).format("Logical generators", "%s" % (logops[0])))
		for i in range(1, 2 * nl):
			print(("{:<%d} {:<%d}" % (tab, tab)).format("", "%s" % (logops[i])))
		peops = PauliOperatorToSymbol(T)
		print(("{:<%d} {:<%d}" % (tab, tab)).format("Pure error generators", "%s" % (peops[0])))
		for i in range(1, nq - nl):
			print(("{:<%d} {:<%d}" % (tab, tab)).format("", "%s" % (peops[i])))
		print("Commutation relations")
		print("         "),
		for i in range(nq - nl):
			print("    S_%d" % (i)),
		for i in range(2 * nl):
			print("    L_%d" % (i)),
		for i in range(nq - nl):
			print("    T_%d" % (i)),
		print("")
		for i in range(2 * nq):
			if (i < (nq - nl)):
				print("S_%d    " % (i)),
			elif (i < (nq + nl)):
				print("L_%d    " % (i - (nq - nl))),
			else:
				print("T_%d    " % (i - (nq + nl))),
			for j in range(2 * nq):
				commutation[i, j] = SymplecticProduct(canonical[i, :], canonical[j, :])
				print("      %d" % (commutation[i, j])),
			print("")
		print("\033[0m")
	# testing if the commutation matrix is consistent with a canonical basis.
	# stabilizer generators must commute with all except the associated pure errors.
	for i in range(nq - nl):
		if (np.any(commutation[i, np.arange(2 * nq)!=(nq + nl + i)] > 0)):
			if (verbose == 1):
				print("\033[2mStabilizer generators do not have the right commutation relations\033[0m")
			return 0
	# logical generators must commute with all except the associated conjugte logical.
	for i in range(2 * nl):
		if (np.any(commutation[nq - nl + i, np.arange(2 * nq)!=(nq - nl + (i + nl) % (2 * nl))] > 0)):
			if (verbose == 1):
				print("\033[2mLogical generators do not have the right commutation relations\033[0m")
			return 0
	# pure error generators must commute with all except the associated stabilizer.
	for i in range(nq - nl):
		if (np.any(commutation[nq + nl + i, np.arange(2 * nq)!=i] > 0)):
			if (verbose == 1):
				print("\033[2mPure error generators do not have the right commutation relations\033[0m")
			return 0
	return 1


def Print(qecc):
	# print all the details of the error correcting code
	tab = 30
	encoding = ["I", "X", "Y", "Z"]
	print("\033[2m")
	print(("{:<%d} {:<%d}" % (tab, tab)).format("Name", qecc.name))
	print(("{:<%d} {:<%d}" % (tab, tab)).format("[[N, K, D]]", "[[%d, %d, %d]]" % (qecc.N, qecc.K, qecc.D)))
	stabops = PauliOperatorToSymbol(qecc.S)
	print(("{:<%d} {:<%d}" % (tab, tab)).format("Stabilizer generators", "%s" % (stabops[0])))
	for i in range(1, qecc.N - qecc.K):
		print(("{:<%d} {:<%d}" % (tab, tab)).format("", "%s" % (stabops[i])))
	logops = PauliOperatorToSymbol(qecc.L)
	print(("{:<%d} {:<%d}" % (tab, tab)).format("Logical generators", "%s" % (logops[0])))
	for i in range(1, 2 * qecc.K):
		print(("{:<%d} {:<%d}" % (tab, tab)).format("", "%s" % (logops[i])))
	peops = PauliOperatorToSymbol(qecc.T)
	print(("{:<%d} {:<%d}" % (tab, tab)).format("Pure error generators", "%s" % (peops[0])))
	for i in range(1, qecc.N - qecc.K):
		print(("{:<%d} {:<%d}" % (tab, tab)).format("", "%s" % (peops[i])))
	print("\033[0m")
	# IsCanonicalBasis(qecc.S, qecc.L, qecc.T, verbose = 1)
	print("\033[2m")
	if (not (qecc.lookup is None)):
		print(("{:<%d} {:<%d}" % (tab, tab)).format("Look up table", "{:<5} {:<5}".format("s", "L")))
		for i in range(2**(qecc.N - qecc.K)):
			print(("{:<%d} {:<%d}" % (tab, tab)).format("", "{:<5} {:<5}".format("%d" % (i), "%s" % (encoding[qecc.lookup[i, 0]]))))
	print("xxxxx\033[0m")
	return None


def ConstructSyndromeProjectors(qecc):
	# Construct the syndrome projectors
	# Construct the stabilizer group and then combine the stabilizer according to the signs.
	stabilizers = np.zeros((2**(qecc.N - qecc.K), 2**qecc.N, 2**qecc.N), dtype = np.complex128)
	stabilizers[0, :, :] = np.identity(2**qecc.N, dtype = np.complex128)
	for s in range(1, 2**(qecc.N - qecc.K)):
		sgens = np.array(map(np.int8, np.binary_repr(s, width = qecc.N - qecc.K)), dtype = np.int8)
		(stabop, stabph) = PauliProduct(*qecc.S[np.nonzero(sgens)])
		stabilizers[s, :, :] = stabph * PauliOperatorToMatrix(stabop)
	projectors = np.einsum("ij,jkl->ikl", qecc.syndsigns, stabilizers)
	# Save the projectors on to a file in chflow/code
	np.save("./../code/%s_syndproj.npy" % (qecc.name), projectors)
	return None

	
def ConstructPauliBasis(nqubits):
	# Construct the list of all (4^n) Pauli matrices that act on a given number (n) of qubits.
	pbasis = np.zeros((4**nqubits, 2**nqubits, 2**nqubits), dtype = np.complex128)
	for i in range(4**nqubits):
		pvec = ChangeBase(i, 4, nqubits)
		# print("Basis element %d: %s" % (i, np.array_str(pvec, max_line_width = 150)))
		# k-fold tensor product of Pauli matrices.
		element = np.zeros((2, 2), dtype = np.complex128)
		for j in range(2):
			for k in range(2):
				element[j, k] = gv.Pauli[pvec[0], j, k]
		for q in range(1, nqubits):
			element = np.kron(element, gv.Pauli[pvec[q], :, :])
		for j in range(2**nqubits):
			for k in range(2**nqubits):
				pbasis[i, j, k] = element[j, k]
	return pbasis


def ChangeBase(number, base, width):
	# change the base of a number representation from decimal to a given base.
	# Suppose the new base is b and the number in decimal is n and the number in the new base is an array Z of size ceil(log_b (number)).
	# Set m <- n, k <- ceil(log_b(number)).
	# while m >= b, do
	# A[k] = m/(b^k)
	# m <- m % (b^k)
	# k --
	# end.
	newnumber = np.zeros(width, dtype = np.int8)
	digit = width - 1
	remainder = number
	while (remainder > 0):
		newnumber[digit] = np.mod(remainder, 4)
		remainder = remainder/base
		digit = digit - 1
	return newnumber


def PauliCliffordConjugations(qecc):
	# List out all the conjugation rules for clifford operators
	# The clifford operators do a permutation of the Pauli frame axes, keeping the condition i (X Y Z) = I.
	# So we have the following conjugation rules. X -> {+,- X, +,- Y, +,- Z} and for each choice of X, we have Y -> {+,- A, +,- B} where A, B are the Pauli operators that do not include the result of the transformation of X. Finally, Z is fixed by ensuring the condition that i (X Y Z) = I.

	# C 	X |	Y |	Z
	# --------------
	# C1	X 	Y 	Z
	# C2	X 	-Y	-Z

	# C5	-X	Y 	-Z
	# C6	-X 	-Y 	Z

	# C3	X	Z 	-Y
	# C4	X 	-Z 	Y

	# C7	-X 	Z 	Y
	# C8	-X 	-Z 	-Y

	# C9	Y 	X 	-Z
	# C10	Y 	-X 	Z
	# C11	Y 	Z 	X
	# C12	Y 	-Z 	-X

	# C13	-Y 	X 	Z
	# C14	-Y 	-X 	-Z
	# C15	-Y 	Z 	X
	# C16	-Y 	-Z 	-X

	# C17	Z 	X 	Y
	# C18	Z 	-X 	-Y
	# C19	Z 	Y 	-X
	# C20	Z 	-Y 	X

	# C21	-Z 	X 	-Y
	# C22	-Z 	-X 	Y
	# C23	-Z 	Y	X
	# C24	-Z 	-Y	-X

	# The shuffling is done to ensure that the first four Clifford operators are I, X, Y and Z.
	conj = [["X", "Y",	"Z"],
			["X", "-Y",	"-Z"],
			["-X", "Y", "-Z"],
			["-X", "-Y", "Z"],
			["X", "Z", 	"-Y"],
			["X", "-Z",	"Y"],
			["-X", "Z", "Y"],
			["-X", "-Z", "-Y"],
			["Y", "X", "-Z"],
			["Y", "-X", "Z"],
			["Y", "Z", "X"],
			["Y", "-Z", "-X"],
			["-Y", "X", "Z"],
			["-Y", "-X", "-Z"],
			["-Y", "Z", "X"],
			["-Y", "-Z", "-X"],
			["Z", "X", "Y"],
			["Z", "-X", "-Y"],
			["Z", "Y", "-X"],
			["Z", "-Y", "X"],
			["-Z", "X",	"-Y"],
			["-Z", "-X", "Y"],
			["-Z", "Y",	"X"],
			["-Z", "-Y", "-X"]]
	symbmap = {"X":[1, 1], "-X":[1, -1], "Y":[2, 1], "-Y":[2, -1], "Z":[3, 1], "-Z":[3, -1]}
	qecc.conjugations = np.zeros((2, 24, 4), dtype = np.int8)
	qecc.conjugations[1, :, 0] = 1
	for ci in range(24):
		for pi in range(3):
			(qecc.conjugations[0, ci, pi + 1], qecc.conjugations[1, ci, pi + 1]) = (symbmap[conj[ci][pi]][0], symbmap[conj[ci][pi]][1])
	return qecc.conjugations


def PauliProduct(*paulis):
	# Perform a product of two n-qubit Pauli operators, along with the appropriate phase information
	# The multiplication rule for the single qubit Pauli group is given by
	###	  I 	X 	Y 	Z
	# I   I 	X 	Y 	Z
	# X   X 	I 	iZ 	-iY
	# Y   Y 	-iZ	I 	iX
	# Z   Z 	iY 	-iX I
	# We will use the encoding I --> 0, X --> 1, Y --> 2, Z --> 3.
	# print("Taking a product of %d Pauli operators:" % (len(paulis)))
	# for i in range(len(paulis)):
	# 	print("P_%d = %s" % (i, np.array_str(paulis[i])))
	mult = np.array([[0, 1, 2, 3],
					[1, 0, 3, 2],
					[2, 3, 0, 1],
					[3, 2, 1, 0]], dtype = np.int8)
	phase = np.array([[1, 1, 1, 1],
					[1, 1, 1j, -1j],
					[1, -1j, 1, 1j],
					[1, 1j, -1j, 1]], dtype = np.complex128)
	product = np.zeros(paulis[0].shape[0], dtype = np.int8)
	overall = 1
	for i in range(len(paulis)):
		overall = overall * np.prod(phase[product, paulis[i]])
		product = np.squeeze(mult[product, paulis[i]])
	return (product, overall)


def ConstructSyndProjSigns(qecc):
	# For each syndrome, construct the operators that projects on to the subspace containing Pauli errors with that syndrome
	# The syndrome projector can be expanded as a linear sum over all the stabilizers, with coefficients that depend on the syndrome.
	# Here we only need these coefficients, they are numbers in {+1, -1}. The coefficient of the stabilizer is +1(-1) if it commutes (anti-commutes) with the Pure error of that particular syndrome.
	seqs = np.array(map(lambda t: map(np.int8, np.binary_repr(t, width = (qecc.N - qecc.K))), range(2**(qecc.N - qecc.K))), dtype = np.int8)
	qecc.syndsigns = (-1)**np.dot(seqs, np.transpose(seqs))
	return None


def ConstructNormalizer(qecc):
	# For each logical class, construct the list of Pauli errors in that class, along with the appropriate global phase associated to this error
	qecc.normalizer = np.zeros((4**qecc.K, 2**(qecc.N - qecc.K), qecc.N), dtype = np.int8)
	qecc.normphases = np.zeros((4**qecc.K, 2**(qecc.N - qecc.K)), dtype = np.complex128)
	ordering = np.array([[0, 3], [1, 2]], dtype = np.int8)
	for l in range(4**qecc.K):
		lgens = np.array(map(np.int8, np.binary_repr(l, width = (2 * qecc.K))), dtype = np.int8)
		if (l > 0):
			(logop, logph) = PauliProduct(*qecc.L[np.nonzero(lgens)])
		else:
			(logop, logph) = (np.zeros(qecc.N, dtype = np.int8), 1)
		for s in range(2**(qecc.N - qecc.K)):
			if (s > 0):
				sgens = np.array(map(np.int8, np.binary_repr(s, width = (qecc.N - qecc.K))), dtype = np.int8)
				(stabop, stabph) = PauliProduct(*qecc.S[np.nonzero(sgens)])
			else:
				(stabop, stabph) = (np.zeros(qecc.N, dtype = np.int8), 1)
			# Combine the logical operator with the stabilizers to generate all the operators in the corresponding logical class
			(normop, normph) = PauliProduct(logop, stabop)
			normph = normph * stabph * logph
			if (l == 3):
				normph = normph * 1j
			qecc.normalizer[ordering[lgens[0], lgens[1]], s, :] = normop
			qecc.normphases[ordering[lgens[0], lgens[1]], s] = normph
	return None


def PrepareSyndromeLookUp(qecc):
	# Prepare a lookup table that contains logical corrections (predicted by the min-weight decoder) for every measured syndrome of the code.
	# For every syndrome
	# 	generate the pure error T
	# 	find the logical operator L such that (T.L.S) has least weight over all L and for some S.
	ordering = np.array([[0, 3], [1, 2]], dtype = np.int8)
	qecc.lookup = np.zeros((2**(qecc.N - qecc.K), 2), dtype = np.int8)
	for t in range(2**(qecc.N - qecc.K)):
		if (t > 0):
			tgens = np.array(map(np.int8, np.binary_repr(t, width = (qecc.N - qecc.K))), dtype = np.int8)
			(peop, __) = PauliProduct(*qecc.T[np.nonzero(tgens)])
		else:
			peop = np.zeros(qecc.N, dtype = np.int8)
		qecc.lookup[t, 0] = 0
		qecc.lookup[t, 1] = qecc.N
		for l in range(4**qecc.K):
			lgens = np.array(map(np.int8, np.binary_repr(l, width = (2 * qecc.K))), dtype = np.int8)
			if (l > 0):
				(logop, __) = PauliProduct(*qecc.L[np.nonzero(lgens)])
			else:
				logop = np.zeros(qecc.N, dtype = np.int8)
			for s in range(2**(qecc.N - qecc.K)):
				if (s > 0):
					sgens = np.array(map(np.int8, np.binary_repr(s, width = (qecc.N - qecc.K))), dtype = np.int8)
					(stabop, __) = PauliProduct(*qecc.S[np.nonzero(sgens)])
				else:
					stabop = np.zeros(qecc.N, dtype = np.int8)
				(correction, __) = PauliProduct(peop, logop, stabop)
				weight = np.count_nonzero(correction > 0)
				# print("correction of weight %d\n%s" % (weight, np.array_str(correction)))
				if (weight <= qecc.lookup[t, 1]):
					qecc.lookup[t, 0] = ordering[lgens[0], lgens[1]]
					qecc.lookup[t, 1] = weight
		# print("Syndrome %d: %s\n\tPure error\n\t%s\n\tCorrection\n\t%s" % (i, np.array_str(combTGens), np.array_str(pureError), np.array_str(recoveries[i])))
	return None


def GenerateGroup(gens):
	group = np.zeros(2**gens.shape[0], gens.shape[1], dtype = np.int8)
	phase = np.ones(2**gens.shape[0], dtype = np.complex128)
	for i in range(1, group.shape[0]):
		comb = np.array(map(np.int8, np.binary_repr(i, width = (2 * qecc.K))), dtype = np.int8)
		(group[i], phase[i]) = PauliProduct(*gens[np.nonzero(comb)])
	return group


if __name__ == '__main__':
	# Load a particular type of error correcting code
	codename = sys.argv[1]
	qcode = QuantumErrorCorrectingCode("%s" % (codename))
	Load(qcode)
	Print(qcode)
	IsCanonicalBasis(qcode.S, qcode.L, qcode.T, verbose = 1)
	print("logical action\n%s\n phases\n%s" % (np.array_str(qcode.normalizer), np.array_str(qcode.normphases)))
