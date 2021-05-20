import os
import sys
import numpy as np
from define import globalvars as gv
from define.QECCLfid import utils as ut

class QuantumErrorCorrectingCode:
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
		self.D = 3
		self.S = None
		self.SSym = None
		self.SGroupSym = None
		self.LSym = None
		self.L = None
		self.T = None
		self.syndsigns = None
		self.lookup = None
		self.tailored_lookup = None
		self.decoder_degens = None
		self.normalizer = None
		self.normphases = None
		self.conjugations = None
		self.Paulis_correctable = None
		self.PauliCorrectableIndices = None
		self.PauliOperatorsLST = None
		self.weight_convention = {"method": "Hamming"}
		self.weightdist = None
		self.group_by_weight = None
		self.defnfile = "%s.txt" % (name)
		self.interaction_graph = None
		eof = 0
		with open(("./../code/%s" % self.defnfile), "r") as fp:
			while eof == 0:
				line = fp.readline()
				if not (line):
					eof = 1
				else:
					line = line.strip("\n").strip(" ")
					if not (line[0] == "#"):
						if line == "name":
							(self.name, nkd) = (
								fp.readline().strip("\n").strip(" ").split(" ")
							)
							(self.N, self.K, self.D) = map(np.int8, nkd.split(","))
						elif line == "stabilizer":
							# Read the next N - K lines.
							self.S = PauliToOperator(fp, self.N, self.N - self.K)
							# print("S\n{}".format(self.S))
						elif line == "logical":
							# Read the next 2 K lines.
							self.L = PauliToOperator(fp, self.N, 2 * self.K)
							# print("L\n{}".format(self.L))
						elif line == "pureerror":
							# Read the next N - K lines.
							self.T = PauliToOperator(fp, self.N, self.N - self.K)
						else:
							pass

	################################

	def GetPositionInLST(self, op):
		r"""
		Compute the position of a Pauli operator :math:`P` in an ordering of Pauli operators expressed as :math:`LST`.

		The position can be computed by finding the logical, stabilizer and pure error operators that combine to form :math:`P`.

			- If :math:`{P,L_\ell} = 0` for some logical generator :math:`L_{\ell}`, then
				- if :math:`\ell < k`, then :math:`P` contains the logical generator :math:`L_{k + \ell}`.
				- else, :math:`P` contains te logical generator :math:`L_{\ell - k}`.

			- If :math:`{P, S_{s}} = 0` for some stabilizer generator, then :math:`P` contains the pure error generator :math:`T_{s}`.

			- If :math:`{P, T_{t}} = 0` for some stabilizer generator, then :math:`P` contains the pure error generator :math:`S_{t}`.

		For each of the logical, stabilizer and pure error generators, we form an indicator vector expressing the support of :math:`P` on each of the sets.
		Each of these indicator vectors can be converted to integers, resulting in :math:`\alpha, \beta, \gamma`.

		Finally, the position of :math:`P` in the :math:`LST` ordering is simply :math:`\gamma + 2^{n-k}\beta + \alpha 2^{2n - 2k}`.
		"""
		ordering = np.array(([0, 3], [1, 2]), dtype=np.int)
		gen_support = {
			"L": np.zeros(2 * self.K, dtype=np.int),
			"S": np.zeros(self.N - self.K, dtype=np.int),
			"T": np.zeros(self.N - self.K, dtype=np.int),
		}
		positions = {"L": 0, "S": 0, "T": 0}
		log_indices = GetCommutingInSet(self.L, op, parity=1, props="set")
		for l in log_indices:
			if l < self.K:
				gen_support["L"][l + self.K] = 1
			else:
				gen_support["L"][l - self.K] = 1
		# print("Support on L: {}".format(gen_support["L"]))
		positions["L"] = ordering[gen_support["L"][0], gen_support["L"][1]]

		stab_indices = GetCommutingInSet(self.T, op, parity=1, props="set")
		gen_support["S"][stab_indices] = 1
		# print("Support on S: {}".format(gen_support["S"]))
		positions["S"] = Bin2Int(gen_support["S"])

		pure_indices = GetCommutingInSet(self.S, op, parity=1, props="set")
		gen_support["T"][pure_indices] = 1
		# print("Support on T: {}".format(gen_support["T"]))
		positions["T"] = Bin2Int(gen_support["T"])

		position = (
			positions["T"]
			+ positions["S"] * 2 ** (self.N - self.K)
			+ positions["L"] * 2 ** (2 * self.N - 2 * self.K)
		)
		# print("Operator: {} and its position {}.".format(op, position))
		return position


def Bin2Int(binseq):
	"""
	Convert binary to integer.
	"""
	inum = np.dot(binseq, 2 ** np.arange(binseq.size)[::-1])
	return inum


def Load(qecc, lookup_load=1):
	# Load all the essential information for quantum error correction
	# If the logicals or pure errors are not specified in the input file, we must construct them by Gaussian elimination.
	# Reconstructs the entire Pauli basis.
	# Only works for CSS codes
	if qecc.S is None:
		print(
			"\033[2mInsufficient information -- stabilizer generators not provided.\033[0m"
		)
	else:
		if (qecc.L is None) or (qecc.T is None):
			if qecc.L is None:
				ComputeLogicals(qecc)
				ComputePureErrors(qecc)
			if qecc.T is None:
				ComputePureErrors(qecc)
			else:
				if IsCanonicalBasis(qecc.S, qecc.L, qecc.T, verbose=1) == 0:
					ComputeLogicals(qecc)
				ComputePureErrors(qecc)
		else:
			if IsCanonicalBasis(qecc.S, qecc.L, qecc.T, verbose=1) == 0:
				ComputeLogicals(qecc)
				ComputePureErrors(qecc)
	# Construct stabilizer symplectic form
	populate_symplectic(qecc)
	# Signs in front of each stabilizer element in the syndrome projectors
	ConstructSyndProjSigns(qecc)
	# Elements in the cosets of the normalizer and their phases
	ConstructNormalizer(qecc)
	# Transformations between Pauli operators by Clifford conjugations
	PauliCliffordConjugations(qecc)
	# PrepareSyndromeLookUp(qecc)
	if lookup_load == 1:
		# Compute the minimum weight decoding table
		PrepareSyndromeLookUp(qecc)
		# Compute correctable indices
		ComputeCorrectableIndices(qecc)
	# Generate group elements
	# qecc.stabilizers = qc.GenerateGroup(qecc.S)
	# qecc.pure_errrors = qc.GenerateGroup(qecc.T)
	# logicals_unordered = qc.GenerateGroup(qecc.L)  # I Z X Y
	# qecc.logicals = logicals_unordered[[0, 2, 3, 1]]
	return None


def populate_symplectic(qcode):
	r"""
		Populates symplectic description of all the stabilizers and logicals.
		Stores a dictionary for each stabilizer with keys "sx","sz" and values as binary lists
		"""
	qcode.SSym = []
	qcode.TSym = []
	qcode.LSym = []
	for S in qcode.S:
		dictS = {"sx": None, "sz": None}
		dictS["sx"] = list(map(lambda x: 1 if x == 1 or x == 2 else 0, S))
		dictS["sz"] = list(map(lambda x: 1 if x == 2 or x == 3 else 0, S))
		qcode.SSym.append(dictS)
	for T in qcode.T:
		dictT = {"sx": None, "sz": None}
		dictT["sx"] = list(map(lambda x: 1 if x == 1 or x == 2 else 0, T))
		dictT["sz"] = list(map(lambda x: 1 if x == 2 or x == 3 else 0, T))
		qcode.TSym.append(dictT)
	for L in qcode.L:
		dictL = {"sx": None, "sz": None}
		dictL["sx"] = list(map(lambda x: 1 if x == 1 or x == 2 else 0, L))
		dictL["sz"] = list(map(lambda x: 1 if x == 2 or x == 3 else 0, L))
		qcode.LSym.append(dictL)
	(group_S, __) = GenerateGroup(qcode.S)
	qcode.SGroupSym = []
	for S in group_S:
		dictS = {"sx": None, "sz": None}
		dictS["sx"] = list(map(lambda x: 1 if x == 1 or x == 2 else 0, S))
		dictS["sz"] = list(map(lambda x: 1 if x == 2 or x == 3 else 0, S))
		qcode.SGroupSym.append(dictS)
	qcode.TGroupSym = []
	(group_T, __) = GenerateGroup(qcode.T)
	for T in group_T:
		dictT = {"sx": None, "sz": None}
		dictT["sx"] = list(map(lambda x: 1 if x == 1 or x == 2 else 0, T))
		dictT["sz"] = list(map(lambda x: 1 if x == 2 or x == 3 else 0, T))
		qcode.TGroupSym.append(dictT)
	return None


# Converting between representations of Pauli operators


def PauliToOperator(fp, nqubits, nlines):
	# Convert the Pauli string format to an operator matrix with the encoding: I -> 0, X -> 1, Y -> 2, Z -> 3.
	# Read from a file whose file pointer is given
	strenc = {"I": 0, "X": 1, "Y": 2, "Z": 3}
	sympenc = np.array([[0, 3], [1, 2]], dtype=np.int8)
	operators = np.zeros((nlines, nqubits), dtype=np.int8)
	for i in range(nlines):
		opstr = fp.readline().strip("\n").strip(" ").split(" ")
		if len(opstr) == nqubits:
			# string encoding
			for j in range(nqubits):
				operators[i, j] = strenc[opstr[j]]
		else:
			# symplectic encoding
			for j in range(nqubits):
				operators[i, j] = sympenc[
					np.int8(opstr[j]), np.int8(opstr[nqubits + j])
				]
	return operators


def PauliOperatorToSymbol(ops):
	# Convert a set of Pauli operators in the encoding I -> 0, X -> 1, Y -> 2, Z -> 3 to a string format with I, X, Y and Z characters.
	encoding = ["I", "X", "Y", "Z"]
	opstr = ["" for i in range(ops.shape[0])]
	for i in range(ops.shape[0]):
		for j in range(ops.shape[1]):
			opstr[i] = opstr[i] + ("%s" % ("".join(encoding[ops[i, j]])))
	return opstr


def SymplecticToOperator(sympvec):
	# convert a symplectic form to an operator form
	encoding = np.array([[0, 3], [1, 2]], dtype=np.int8)
	op = np.zeros(sympvec.shape[0] // 2, dtype=np.int8)
	for i in range(op.shape[0]):
		op[i] = encoding[sympvec[i], sympvec[op.shape[0] + i]]
	return op


def OperatorToSymplectic(pauliop):
	# Convert the operator in the encoding I -> 0, X -> 1, Y -> 2, Z -> 3 to its symplectic form.
	encoding = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int8)
	sympvec = np.zeros(2 * pauliop.shape[0], dtype=np.int8)
	for i in range(pauliop.shape[0]):
		(sympvec[i], sympvec[pauliop.shape[0] + i]) = (
			encoding[pauliop[i], 0],
			encoding[pauliop[i], 1],
		)
	return sympvec


def ConvertToSympectic(gens):
	# Convert a list of stabilizer generators to X and Z symplectic matrices.
	sympmat = np.zeros((gens.shape[0], 2 * gens.shape[1]), dtype=np.int8)
	encoding = np.array(([[0, 0], [1, 0], [1, 1], [0, 1]]), dtype=np.int8)
	for i in range(gens.shape[0]):
		for j in range(gens.shape[1]):
			sympmat[i, [j, j + gens.shape[1]]] = encoding[gens[i, j], :]
	return sympmat


def ConvertToOperator(sympmat):
	# Convert a matrix of symplectic vectors to operators in the encoding: I --> 0, X --> 1, Y --> 2, Z --> 3.
	encoding = np.array([[0, 3], [1, 2]], dtype=np.int8)
	nq = sympmat.shape[1] // 2
	pauliops = np.zeros((sympmat.shape[0], nq), dtype=np.int8)
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
	lie = np.array(
		[[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]], dtype=np.int8
	)
	return np.mod(np.sum(lie[pauli1, pauli2]), 2) == 0


def SymplecticProduct(sympvec1, sympvec2):
	# return the symplectic product of two vectors.
	nq = sympvec1.shape[0] // 2
	return np.mod(
		np.dot(sympvec1[:nq], sympvec2[nq:]) + np.dot(sympvec1[nq:], sympvec2[:nq]), 2
	)


# Constructing the canonical Pauli basis


def NullSpace(mat):
	# Compute a basis for the Null space of the input (binary) matrix, with respect to the symplectic product.
	# Given a matrix M = (HX | HZ), we need to compute vectors v such that: M P v = 0 where P = (0 I \\ I 0) where I is an Identity matrix.
	# Hence we need to find the kernel of (HZ | HX) over GF_2.
	# If we have a matrix A over GF_2, the kernel of A consists of vectors v such that: A.v = 0 (mod 2).
	# If we row reduce A to the format [I|P] where I is an identity matrix, then the kernel of A are columns of the matrix [P \\ I].
	nq = mat.shape[1] // 2
	cols = np.arange(mat.shape[1], dtype=np.int8)

	# Reflect the matrix about its center column: HX <--> HZ
	for i in range(nq):
		mat[:, [i, nq + i]] = mat[:, [nq + i, i]]

	for i in range(mat.shape[0]):
		if mat[i, i] == 0:
			# look for a row below that has 1 in the i-th column.
			for j in range(i + 1, mat.shape[0]):
				if mat[j, i] == 1:
					# swap rows j and i and break
					mat[[i, j], :] = mat[[j, i], :]
					break
		if mat[i, i] == 0:
			# look for a column to the right (up to n) that has a 1 in the i-th row.
			for j in range(i + 1, mat.shape[1]):
				if mat[i, j] == 1:
					# swap columns i with j and columns (mat.shape[0] + i) with (mat.shape[0] + j).
					mat[:, [i, j]] = mat[:, [j, i]]
					# record the column swaps so that they can be undone at the end.
					cols[[i, j]] = cols[[j, i]]
					break
		# Row reduce the matrix
		for j in range(mat.shape[0]):
			if not (i == j):
				if mat[j, i] == 1:
					mat[j, :] = np.mod(mat[i, :] + mat[j, :], 2)

	# Decude the Null space vectors and undo the column permutations
	null = np.hstack(
		(
			np.transpose(mat[:, mat.shape[0] :]),
			np.identity((mat.shape[1] - mat.shape[0]), dtype=np.int8),
		)
	)[:, np.argsort(cols)]
	return null


def NullTest(vspace, kernel):
	# test if all the vectors in the null space are orthogonal (with respect to the symplectic product) to the original space.
	print(
		"Null space test with\nvspace\n%s\nkernel\n%s"
		% (
			np.array_str(ConvertToOperator(vspace)),
			np.array_str(ConvertToOperator(kernel)),
		)
	)
	product = np.zeros((vspace.shape[0], kernel.shape[0]), dtype=np.int8)
	for i in range(vspace.shape[0]):
		for j in range(kernel.shape[0]):
			product[i, j] = SymplecticProduct(vspace[i, :], kernel[j, :])
	print("product\n%s" % (np.array_str(product)))
	return None


def ComputeLogicals(qecc):
	# Given the stabilizer, compute a canonical basis for the normalizer consisting of the stabilizer and logical operators.
	# We will use the Symplectic Gram Schmidt Orthogonialization method mentioned in arXiv: 0903.5526v1.
	normalizer = NullSpace(ConvertToSympectic(qecc.S))

	used = np.zeros(normalizer.shape[0], dtype=np.int8)
	# If logs[i] = 0, the i-th normalizer is is a stabilizer.
	# If logs[i] = l, the i-th normalizer is the logical operator Z_l.
	# If logs[i] = -l, the i-th normalizer if the logical operator X_l.
	logs = np.zeros(normalizer.shape[0], dtype=np.int8)
	nlogs = 1
	for i in range(normalizer.shape[0]):
		if used[i] == 0:
			used[i] = 1
			for j in range(normalizer.shape[0]):
				if used[j] == 0:
					if SymplecticProduct(normalizer[i, :], normalizer[j, :]) == 1:
						logs[i] = nlogs
						logs[j] = (-1) * nlogs
						nlogs = nlogs + 1
						used[j] = 1
						for k in range(normalizer.shape[0]):
							if used[k] == 0:
								if SymplecticProduct(
									normalizer[k, :], normalizer[j, :]
								):
									normalizer[k, :] = np.mod(
										normalizer[k, :] + normalizer[i, :], 2
									)
								if SymplecticProduct(
									normalizer[k, :], normalizer[i, :]
								):
									normalizer[k, :] = np.mod(
										normalizer[k, :] + normalizer[j, :], 2
									)
						break
	qecc.S = ConvertToOperator(np.squeeze(normalizer[np.nonzero(abs(logs) == 0), :]))
	qecc.L = np.zeros((2 * qecc.K, qecc.N), dtype=np.int8)
	for i in range(nlogs - 1):
		qecc.L[[i, qecc.K + i], :] = ConvertToOperator(
			np.squeeze(normalizer[np.nonzero(abs(logs) == (i + 1)), :])
		)
	return None


def ComputePureErrors(qecc):
	# Compute the generators for the pure errors, given the stabilizer and logical generators.
	# The i-th pure error must anticommute with the i-th stabilizer generator and commute with all other operators including pure errors.
	stabgens = ConvertToSympectic(qecc.S)
	loggens = ConvertToSympectic(qecc.L)
	puregens = np.zeros((qecc.N - qecc.K, 2 * qecc.N), dtype=np.int8)
	for i in range(qecc.N - qecc.K):
		sols = NullSpace(
			np.vstack((loggens, stabgens[np.arange(qecc.N - qecc.K) != i]))
		)
		# Check which of the elements from the solutions anticommute with the i-th stabilizer.
		# Set that element to be the i-th pure error and add it to the list of constraints
		for j in range(sols.shape[0]):
			if SymplecticProduct(sols[j, :], stabgens[i, :]) == 1:
				break
		puregens[i, :] = sols[j, :]
	# print("Raw pure gens\n{}".format(ConvertToOperator(puregens)))
	# Fixing the commutation relations between pure errors.
	# 1. If Ti anti commutes with Sj, for j not equal to i, then: Ti -> Ti * Tj
	# 2. If Ti anti commutes with Lj, then: Ti -> Ti * Mj where Mj is the logical operator conjugate to Lj
	# 3. If Ti anti commutes with Tj, then Ti -> Ti * Sj
	for i in range(qecc.N - qecc.K):
		for j in range(qecc.N - qecc.K):
			if not (i == j):
				if SymplecticProduct(puregens[i, :], stabgens[j, :]) == 1:
					puregens[i, :] = np.mod(puregens[i, :] + puregens[j, :], 2)
	for i in range(qecc.N - qecc.K):
		for j in range(2 * qecc.K):
			if SymplecticProduct(puregens[i, :], loggens[j, :]) == 1:
				puregens[i, :] = np.mod(
					puregens[i, :] + loggens[(j + qecc.K) % (2 * qecc.K), :], 2
				)
	for i in range(qecc.N - qecc.K):
		for j in range(i + 1, qecc.N - qecc.K):
			if SymplecticProduct(puregens[i, :], puregens[j, :]) == 1:
				puregens[j, :] = np.mod(puregens[j, :] + stabgens[i, :], 2)
	qecc.T = ConvertToOperator(puregens)
	return None


def IsCanonicalBasis(S, L, T, verbose=1):
	# test if a basis is a canonical basis and display the commutation relations
	(nq, nl) = (S.shape[1], S.shape[1] - S.shape[0])
	canonical = np.vstack(
		(ConvertToSympectic(S), ConvertToSympectic(L), ConvertToSympectic(T))
	)
	commutation = np.zeros((canonical.shape[0], canonical.shape[0]), dtype=np.int8)
	if verbose == 1:
		print("\033[2m")
		print("N = %d, K = %d" % (nq, nl))
		tab = 20
		stabops = PauliOperatorToSymbol(S)
		print(
			("{:<%d} {:<%d}" % (tab, tab)).format(
				"Stabilizer generators", "%s" % (stabops[0])
			)
		)
		for i in range(1, nq - nl):
			print(("{:<%d} {:<%d}" % (tab, tab)).format("", "%s" % (stabops[i])))
		logops = PauliOperatorToSymbol(L)
		print(
			("{:<%d} {:<%d}" % (tab, tab)).format(
				"Logical generators", "%s" % (logops[0])
			)
		)
		for i in range(1, 2 * nl):
			print(("{:<%d} {:<%d}" % (tab, tab)).format("", "%s" % (logops[i])))
		peops = PauliOperatorToSymbol(T)
		print(
			("{:<%d} {:<%d}" % (tab, tab)).format(
				"Pure error generators", "%s" % (peops[0])
			)
		)
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
			if i < (nq - nl):
				print("S_%d    " % (i)),
			elif i < (nq + nl):
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
		if np.any(commutation[i, np.arange(2 * nq) != (nq + nl + i)] > 0):
			if verbose == 1:
				print(
					"\033[2mStabilizer generators do not have the right commutation relations\033[0m"
				)
			return 0
	# logical generators must commute with all except the associated conjugte logical.
	for i in range(2 * nl):
		if np.any(
			commutation[
				nq - nl + i, np.arange(2 * nq) != (nq - nl + (i + nl) % (2 * nl))
			]
			> 0
		):
			if verbose == 1:
				print(
					"\033[2mLogical generators do not have the right commutation relations\033[0m"
				)
			return 0
	# pure error generators must commute with all except the associated stabilizer.
	for i in range(nq - nl):
		if np.any(commutation[nq + nl + i, np.arange(2 * nq) != i] > 0):
			if verbose == 1:
				print(
					"\033[2mPure error generators do not have the right commutation relations\033[0m"
				)
			return 0
	return 1


def PrintQEC(qecc):
	# print all the details of the error correcting code
	tab = 30
	encoding = ["I", "X", "Y", "Z"]
	print("\033[2m")
	print(("{:<%d} {:<%d}" % (tab, tab)).format("Name", qecc.name))
	print(
		("{:<%d} {:<%d}" % (tab, tab)).format(
			"[[N, K, D]]", "[[%d, %d, %d]]" % (qecc.N, qecc.K, qecc.D)
		)
	)
	stabops = PauliOperatorToSymbol(qecc.S)
	print(
		("{:<%d} {:<%d}" % (tab, tab)).format(
			"Stabilizer generators", "%s" % (stabops[0])
		)
	)
	for i in range(1, qecc.N - qecc.K):
		print(("{:<%d} {:<%d}" % (tab, tab)).format("", "%s" % (stabops[i])))
	logops = PauliOperatorToSymbol(qecc.L)
	print(
		("{:<%d} {:<%d}" % (tab, tab)).format("Logical generators", "%s" % (logops[0]))
	)
	for i in range(1, 2 * qecc.K):
		print(("{:<%d} {:<%d}" % (tab, tab)).format("", "%s" % (logops[i])))
	peops = PauliOperatorToSymbol(qecc.T)
	print(
		("{:<%d} {:<%d}" % (tab, tab)).format(
			"Pure error generators", "%s" % (peops[0])
		)
	)
	for i in range(1, qecc.N - qecc.K):
		print(("{:<%d} {:<%d}" % (tab, tab)).format("", "%s" % (peops[i])))
	print("\033[0m")
	# IsCanonicalBasis(qecc.S, qecc.L, qecc.T, verbose = 1)
	print("\033[2m")
	if not (qecc.lookup is None):
		print(
			("{:<%d} {:<%d}" % (tab, tab)).format(
				"Look up table", "{:<5} {:<5} {:<25}".format("s", "L", "R")
			)
		)
		for i in range(2 ** (qecc.N - qecc.K)):
			print(
				("{:<%d} {:<%d}" % (tab, tab)).format(
					"",
					"{:<5} {:<5} {:<25}".format(
						"%d" % (i),
						"%s" % (encoding[qecc.lookup[i, 0]]),
						"%s"
						% (
							" ".join(
								[encoding[qecc.lookup[i, 2 + j]] for j in range(qecc.N)]
							)
						),
					),
				)
			)
	print("xxxxx\033[0m")
	return None


def ConstructSyndromeProjectors(qecc):
	# Construct the syndrome projectors
	# Construct the stabilizer group and then combine the stabilizer according to the signs.
	stabilizers = np.zeros(
		(2 ** (qecc.N - qecc.K), 2 ** qecc.N, 2 ** qecc.N), dtype=np.complex128
	)
	stabilizers[0, :, :] = np.identity(2 ** qecc.N, dtype=np.complex128)
	for s in range(1, 2 ** (qecc.N - qecc.K)):
		sgens = np.array(
			list(map(np.int8, np.binary_repr(s, width=qecc.N - qecc.K))), dtype=np.int8
		)
		(stabop, stabph) = PauliProduct(*qecc.S[np.nonzero(sgens)])
		stabilizers[s, :, :] = stabph * PauliOperatorToMatrix(stabop)
	projectors = np.einsum("ij,jkl->ikl", qecc.syndsigns, stabilizers)
	# Save the projectors on to a file in chflow/code
	np.save("./../code/%s_syndproj.npy" % (qecc.name), projectors)
	return None


def ConstructPauliBasis(nqubits):
	# Construct the list of all (4^n) Pauli matrices that act on a given number (n) of qubits.
	pbasis = np.zeros((4 ** nqubits, 2 ** nqubits, 2 ** nqubits), dtype=np.complex128)
	for i in range(4 ** nqubits):
		pvec = ChangeBase(i, 4, nqubits)
		# print("Basis element %d: %s" % (i, np.array_str(pvec, max_line_width = 150)))
		# k-fold tensor product of Pauli matrices.
		element = np.zeros((2, 2), dtype=np.complex128)
		for j in range(2):
			for k in range(2):
				element[j, k] = gv.Pauli[pvec[0], j, k]
		for q in range(1, nqubits):
			element = np.kron(element, gv.Pauli[pvec[q], :, :])
		for j in range(2 ** nqubits):
			for k in range(2 ** nqubits):
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
	newnumber = np.zeros(width, dtype=np.int8)
	digit = width - 1
	remainder = number
	while remainder > 0:
		newnumber[digit] = np.mod(remainder, 4)
		remainder = remainder / base
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
	conj = [
		["X", "Y", "Z"],
		["X", "-Y", "-Z"],
		["-X", "Y", "-Z"],
		["-X", "-Y", "Z"],
		["X", "Z", "-Y"],
		["X", "-Z", "Y"],
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
		["-Z", "X", "-Y"],
		["-Z", "-X", "Y"],
		["-Z", "Y", "X"],
		["-Z", "-Y", "-X"],
	]
	symbmap = {
		"X": [1, 1],
		"-X": [1, -1],
		"Y": [2, 1],
		"-Y": [2, -1],
		"Z": [3, 1],
		"-Z": [3, -1],
	}
	qecc.conjugations = np.zeros((2, 24, 4), dtype=np.int8)
	qecc.conjugations[1, :, 0] = 1
	for ci in range(24):
		for pi in range(3):
			(qecc.conjugations[0, ci, pi + 1], qecc.conjugations[1, ci, pi + 1]) = (
				symbmap[conj[ci][pi]][0],
				symbmap[conj[ci][pi]][1],
			)
	return qecc.conjugations


def PauliProduct(*paulis):
	# Perform a product of many n-qubit Pauli operators, along with the appropriate phase information
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
	mult = np.array(
		[[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]], dtype=np.int8
	)
	phase = np.array(
		[[1, 1, 1, 1], [1, 1, 1j, -1j], [1, -1j, 1, 1j], [1, 1j, -1j, 1]],
		dtype=np.complex128,
	)
	product = np.zeros(paulis[0].shape[0], dtype=np.int8)
	overall = 1
	for i in range(len(paulis)):
		overall = overall * np.prod(phase[product, paulis[i]])
		product = np.squeeze(mult[product, paulis[i]])
	return (product, overall)


def ConstructSyndProjSigns(qecc):
	# For each syndrome, construct the operators that projects on to the subspace containing Pauli errors with that syndrome
	# The syndrome projector can be expanded as a linear sum over all the stabilizers, with coefficients that depend on the syndrome.
	# Here we only need these coefficients, they are numbers in {+1, -1}. The coefficient of the stabilizer is +1(-1) if it commutes (anti-commutes) with the Pure error of that particular syndrome.
	seqs = np.array(
		list(
			map(
				lambda t: list(
					map(np.int8, np.binary_repr(t, width=(qecc.N - qecc.K)))
				),
				range(2 ** (qecc.N - qecc.K)),
			)
		),
		dtype=np.int8,
	)
	qecc.syndsigns = (-1) ** np.dot(seqs, np.transpose(seqs))
	return None


def ConstructNormalizer(qecc):
	# For each logical class, construct the list of Pauli errors in that class, along with the appropriate global phase associated to this error
	qecc.normalizer = np.zeros(
		(4 ** qecc.K, 2 ** (qecc.N - qecc.K), qecc.N), dtype=np.int8
	)
	qecc.normphases = np.zeros(
		(4 ** qecc.K, 2 ** (qecc.N - qecc.K)), dtype=np.complex128
	)
	ordering = np.array([[0, 3], [1, 2]], dtype=np.int8)
	for l in range(4 ** qecc.K):
		lgens = np.array(
			list(map(np.int8, np.binary_repr(l, width=(2 * qecc.K)))), dtype=np.int8
		)
		if l > 0:
			(logop, logph) = PauliProduct(*qecc.L[np.nonzero(lgens)])
		else:
			(logop, logph) = (np.zeros(qecc.N, dtype=np.int8), 1)
		if l == 3:
			logph = logph * 1j
		for s in range(2 ** (qecc.N - qecc.K)):
			if s > 0:
				sgens = np.array(
					list(map(np.int8, np.binary_repr(s, width=(qecc.N - qecc.K)))),
					dtype=np.int8,
				)
				(stabop, stabph) = PauliProduct(*qecc.S[np.nonzero(sgens)])
			else:
				(stabop, stabph) = (np.zeros(qecc.N, dtype=np.int8), 1)
			# Combine the logical operator with the stabilizers to generate all the operators in the corresponding logical class
			(normop, normph) = PauliProduct(logop, stabop)
			normph = normph * stabph * logph
			qecc.normalizer[ordering[lgens[0], lgens[1]], s, :] = normop
			qecc.normphases[ordering[lgens[0], lgens[1]], s] = normph
	return None


def PrepareSyndromeLookUp(qecc):
	# Prepare a lookup table that contains logical corrections (predicted by the min-weight decoder) for every measured syndrome of the code.
	# For every syndrome
	# 	generate the pure error T
	# 	find the logical operator L such that (T.L.S) has least weight over all L and for some S.
	# Each row of the constructed lookup table corresponds to a syndrome outcome.
	# The row entries are the logical correction, weight of the minimum weight operator and the full minimum weight correction.
	# print("Computing lookup for {}".format(qecc.weight_convention))
	nstabs = 2 ** (qecc.N - qecc.K)
	nlogs = 4 ** qecc.K
	ordering = np.array([[0, 3], [1, 2]], dtype=np.int8)
	qecc.lookup = np.zeros((nstabs, 2 + qecc.N), dtype=np.double)
	qecc.PauliOperatorsLST = np.zeros((4 ** qecc.N, qecc.N), dtype=np.int)
	qecc.weightdist = np.zeros(4 ** qecc.N, dtype=np.int)
	for t in range(nstabs):
		if t > 0:
			tgens = np.array(
				list(map(np.int8, np.binary_repr(t, width=(qecc.N - qecc.K)))),
				dtype=np.int8,
			)
			(peop, __) = PauliProduct(*qecc.T[np.nonzero(tgens)])
		else:
			peop = np.zeros(qecc.N, dtype=np.int8)
		qecc.lookup[t, 0] = 0
		qecc.lookup[t, 1] = -1
		for l in range(nlogs):
			lgens = np.array(
				list(map(np.int8, np.binary_repr(l, width=(2 * qecc.K)))), dtype=np.int8
			)
			if l > 0:
				(logop, __) = PauliProduct(*qecc.L[np.nonzero(lgens)])
			else:
				logop = np.zeros(qecc.N, dtype=np.int8)
			for s in range(nstabs):
				if s > 0:
					sgens = np.array(
						list(map(np.int8, np.binary_repr(s, width=(qecc.N - qecc.K)))),
						dtype=np.int8,
					)
					(stabop, __) = PauliProduct(*qecc.S[np.nonzero(sgens)])
				else:
					stabop = np.zeros(qecc.N, dtype=np.int8)
				(correction, __) = PauliProduct(peop, logop, stabop)
				# weight = np.count_nonzero(correction > 0)
				(hamming_weight, error_weight) = ErrorWeight(correction, qecc.weight_convention)
				if (qecc.lookup[t, 1] == -1):
					qecc.lookup[t, 0] = ordering[lgens[0], lgens[1]]
					qecc.lookup[t, 1] = error_weight
					qecc.lookup[t, 2:] = correction
				else:
					if (error_weight <= qecc.lookup[t, 1]):
						qecc.lookup[t, 0] = ordering[lgens[0], lgens[1]]
						qecc.lookup[t, 1] = error_weight
						qecc.lookup[t, 2:] = correction
				# Record the weight and the correction.
				qecc.weightdist[
					ordering[lgens[0], lgens[1]] * nstabs * nstabs + s * nstabs + t
				] = hamming_weight
				qecc.PauliOperatorsLST[
					(ordering[lgens[0], lgens[1]] * nstabs * nstabs + s * nstabs + t), :
				] = correction
	# print("Lookup table\n{}".format(qecc.lookup))
	# Group errors by weight
	qecc.group_by_weight = {}
	for w in range(1 + qecc.N):
		(qecc.group_by_weight[w],) = np.nonzero(qecc.weightdist == w)
	return None


def ErrorWeight(pauli_error, convention=None):
	# Compute the weight of a Pauli error with respect to a decoding technique.
	# 1. Hamming: corresponds to the number of non-identity 2 x 2 Pauli matrices in the tensor product decomposition.
	# 2. Bias: Here we will assume relative importance for I, X, Y and Z are given.
	hamming_weight = np.count_nonzero(pauli_error)
	if convention is None:
		error_weight = hamming_weight
	else:
		if convention["method"] == "Hamming":
			error_weight = hamming_weight
		elif convention["method"] == "bias":
			# We need the relative importance of I, X, Y and Z.
			# These numbers can be >= 1, with the larger number indicating higher probability of a given type of error.
			# The weight of the error will be computed by: multiplying the number of Paauli matrices of a given type (I, X, Y or Z) by the inverse of its relative importance.
			# print("Function: ErrorWeight({}, {})".format(pauli_error, convention))
			error_weight = 0
			paulis = ["X", "Y", "Z"]
			for p in range(3):
				error_weight += np.count_nonzero(pauli_error == (1 + p)) * convention["weights"][paulis[p]]
			# print("Modified weight of {} = {}.".format(pauli_error, weight))
		else:
			pass
	return (hamming_weight, error_weight)


def ComputeCorrectableIndices(qcode):
	r"""
	Compute the indices of correctable errors in a code.
	"""
	minwt_reps = list(map(ut.convert_Pauli_to_symplectic, qcode.lookup[:, 2:].astype(np.int)))
	degeneracies = [
		ut.prod_sym(unique_rep, stab)
		for unique_rep in minwt_reps
		for stab in qcode.SGroupSym
	]
	qcode.Paulis_correctable = np.array(
		list(map(ut.convert_symplectic_to_Pauli, degeneracies)), dtype=np.int
	)
	qcode.PauliCorrectableIndices = np.array(
		list(map(lambda op: qcode.GetPositionInLST(op), qcode.Paulis_correctable)),
		dtype=np.int,
	)
	# print("Pauli correctable indices : {}".format(list(qcode.PauliCorrectableIndices)))
	return None


def ComputeAdaptiveDecoder(qecc, pauli_probs, method="ML"):
	if method == "MP":
		print("Using Max probability decoder to construct lookup")
		ComputeMaxProbAdaptive(qecc, pauli_probs)
	else:
		print("Using Max Likelihood decoder to construct lookup")
		ComputeMLAdaptive(qecc, pauli_probs)
	# print("Tailored lookup = {}".format(qecc.tailored_lookup))
	return None


def ComputeMLAdaptive(qecc, pauli_probs):
	# Prepare a lookup table that contains logical corrections (predicted by the min-weight decoder) for every measured syndrome of the code.
	# For every syndrome
	#   generate the pure error T
	#   find the logical operator L such that (T.L.S) has least weight over all L and for some S.
	# Each row of the constructed lookup table corresponds to a syndrome outcome.
	# The row entries are the logical correction, weight of the minimum weight operator and the full minimum weight correction.
	nstabs = 2 ** (qecc.N - qecc.K)
	nlogs = 4 ** qecc.K
	ordering = np.array([[0, 3], [1, 2]], dtype=np.int8)
	qecc.tailored_lookup = np.zeros((nstabs, 2 + qecc.N), dtype=np.double)
	for t in range(nstabs):
		if t > 0:
			tgens = np.array(
				list(map(np.int8, np.binary_repr(t, width=(qecc.N - qecc.K)))),
				dtype=np.int8,
			)
			(peop, __) = PauliProduct(*qecc.T[np.nonzero(tgens)])
		else:
			peop = np.zeros(qecc.N, dtype=np.int8)
		for l in range(nlogs):
			lgens = np.array(
				list(map(np.int8, np.binary_repr(l, width=(2 * qecc.K)))), dtype=np.int8
			)
			if l > 0:
				(logop, __) = PauliProduct(*qecc.L[np.nonzero(lgens)])
			else:
				logop = np.zeros(qecc.N, dtype=np.int8)
			prob = 0
			for s in range(nstabs):
				if s > 0:
					sgens = np.array(
						list(map(np.int8, np.binary_repr(s, width=(qecc.N - qecc.K)))),
						dtype=np.int8,
					)
					(stabop, __) = PauliProduct(*qecc.S[np.nonzero(sgens)])
				else:
					stabop = np.zeros(qecc.N, dtype=np.int8)
				(correction, __) = PauliProduct(peop, logop, stabop)
				# weight = np.count_nonzero(correction > 0)
				prob += pauli_probs[
					ordering[lgens[0], lgens[1]] * nstabs * nstabs + s * nstabs + t
				]

			if prob >= qecc.tailored_lookup[t, 1]:
				qecc.tailored_lookup[t, 0] = ordering[lgens[0], lgens[1]]
				qecc.tailored_lookup[t, 1] = prob
				qecc.tailored_lookup[t, 2:] = correction
	return None


def ComputeMaxProbAdaptive(qecc, pauli_probs):
	# Prepare a lookup table that contains logical corrections (predicted by the min-weight decoder) for every measured syndrome of the code.
	# For every syndrome
	#   generate the pure error T
	#   find the logical operator L such that (T.L.S) has least weight over all L and for some S.
	# Each row of the constructed lookup table corresponds to a syndrome outcome.
	# The row entries are the logical correction, weight of the minimum weight operator and the full minimum weight correction.
	nstabs = 2 ** (qecc.N - qecc.K)
	nlogs = 4 ** qecc.K
	ordering = np.array([[0, 3], [1, 2]], dtype=np.int8)
	qecc.tailored_lookup = np.zeros((nstabs, 2 + qecc.N), dtype=np.double)
	for t in range(nstabs):
		if t > 0:
			tgens = np.array(
				list(map(np.int8, np.binary_repr(t, width=(qecc.N - qecc.K)))),
				dtype=np.int8,
			)
			(peop, __) = PauliProduct(*qecc.T[np.nonzero(tgens)])
		else:
			peop = np.zeros(qecc.N, dtype=np.int8)
		for l in range(nlogs):
			lgens = np.array(
				list(map(np.int8, np.binary_repr(l, width=(2 * qecc.K)))), dtype=np.int8
			)
			if l > 0:
				(logop, __) = PauliProduct(*qecc.L[np.nonzero(lgens)])
			else:
				logop = np.zeros(qecc.N, dtype=np.int8)
			for s in range(nstabs):
				if s > 0:
					sgens = np.array(
						list(map(np.int8, np.binary_repr(s, width=(qecc.N - qecc.K)))),
						dtype=np.int8,
					)
					(stabop, __) = PauliProduct(*qecc.S[np.nonzero(sgens)])
				else:
					stabop = np.zeros(qecc.N, dtype=np.int8)
				(correction, __) = PauliProduct(peop, logop, stabop)
				# weight = np.count_nonzero(correction > 0)
				prob = pauli_probs[
					ordering[lgens[0], lgens[1]] * nstabs * nstabs + s * nstabs + t
				]
				if prob >= qecc.tailored_lookup[t, 1]:
					qecc.tailored_lookup[t, 0] = ordering[lgens[0], lgens[1]]
					qecc.tailored_lookup[t, 1] = prob
					qecc.tailored_lookup[t, 2:] = correction
	return None


def GenerateGroup(gens):
	group = np.zeros((2 ** gens.shape[0], gens.shape[1]), dtype=np.int8)
	phase = np.ones(2 ** gens.shape[0], dtype=np.complex128)
	for i in range(1, group.shape[0]):
		comb = np.array(
			list(map(np.int8, np.binary_repr(i, width=gens.shape[0]))), dtype=np.int8
		)
		(group[i], phase[i]) = PauliProduct(*gens[np.nonzero(comb)])
	return (group, phase)


def GetCommuting(log_op, stab_op, lgens, sgens, tgens):
	r"""
	Given a Pauli operator P, determine the Pauli operators in the LST ordering that commute and anticommute with P.
	For each operator P of the form TLS we need to compute:

	.. math::
		\begin{gather}
		\Gamma_{P,P} = \sum_{A\in S_{C}}p_{A} - \sum_{B\in S_{A}}p_{B}
		\end{gather}

	where :math:`S_{C}` denotes the set of all operators that commute with :math:`P`, while :math:`S_{A}` is the set of all operators that anticommute with :math:`P`.

	Note that if :math:`P = LS`, then all operators of the form

	.. math::
		\begin{gather}
		Q_{1} = L^{C}_{\ell} S_{i} T^{C}_{t}\\
		Q_{2} = L^{A}_{\ell} S_{i} T^{A}_{t}
		\end{gather}

	commute with :math:`P` while those of the form

	.. math::
		\begin{gather}
		Q_{3} = L^{C}_{\ell} S_{i} T^{A}_{t} \\
		Q_{4} = L^{A}_{\ell} S_{i} T^{C}_{t}
		\end{gather}

	anticommute with :math:`P`. In both of the cases,

		- :math:`S_{i}` is a stabilizer.
		- :math:`L^{C}_{\ell}` is a logical operator that commutes with :math:`L`.
		- :math:`L^{A}_{\ell}` is a logical operator that anti-commutes with :math:`L`.
		- :math:`T^{C}_{t}` is a pure error which commutes with :math:`S`.
		- :math:`T^{A}_{t}` is a pure error which anticommutes with :math:`S`.

	The above operators can be related to their positions in the :math:`LST` ordering.
	The operators of the form :math:`Q_{i}` have indices: :math:`\ell * 2^{2n - 2k} + i * 2^{n-k} + t`.
	"""
	# 0  1   2  3
	# 00 01 10 11
	# I  Z  X  Y
	# 0  3  1  2
	ordering = np.array([0, 3, 1, 2], dtype=np.int8)
	k = lgens.shape[0] // 2
	n = lgens.shape[1]
	# print("n = {}, k = {}".format(n, k))
	supports = {}
	supports.update({"LC": GetCommutingInSet(lgens, log_op, 0, props="group")})
	# Convert to right ordering
	supports["LC"] = ordering[supports["LC"]]
	supports.update({"LA": GetCommutingInSet(lgens, log_op, 1, props="group")})
	# Convert to right ordering
	supports["LA"] = ordering[supports["LA"]]

	supports.update({"TC": GetCommutingInSet(tgens, stab_op, 0, props="group")})
	supports.update({"TA": GetCommutingInSet(tgens, stab_op, 1, props="group")})
	# print(
	#     "LC = {}\nLA = {}\nTC = {}\nTA = {}".format(
	#         supports["LC"], supports["LA"], supports["TC"], supports["TA"]
	#     )
	# )
	indicators = {
		"commuting": np.zeros(4 ** n, dtype=np.int),
		"anticommuting": np.zeros(4 ** n, dtype=np.int),
	}
	for l in supports["LC"]:
		for t in supports["TC"]:
			# print(
			#     "index for l = {}, s = {}, t = {} is = {}".format(
			#         l, s, t, l * 2 ** (2 * n - 2 * k) + s * 2 ** (n - k) + t
			#     )
			# )
			indicators["commuting"][
				l * 2 ** (2 * n - 2 * k)
				+ np.arange(2 ** (n - k), dtype=np.int) * 2 ** (n - k)
				+ t
			] = 1
	# print("LC indicator: {}".format(np.nonzero(indicators["commuting"])))
	for l in supports["LA"]:
		for t in supports["TA"]:
			# print(
			#     "index for l = {}, s = {}, t = {} is = {}".format(
			#         l, s, t, l * 2 ** (2 * n - 2 * k) + s * 2 ** (n - k) + t
			#     )
			# )
			indicators["commuting"][
				l * 2 ** (2 * n - 2 * k)
				+ np.arange(2 ** (n - k), dtype=np.int) * 2 ** (n - k)
				+ t
			] = 1
	# print("LA indicator: {}".format(np.nonzero(indicators["commuting"])))
	for l in supports["LC"]:
		for t in supports["TA"]:
			# print(
			#     "index for l = {}, s = {}, t = {} is = {}".format(
			#         l, s, t, l * 2 ** (2 * n - 2 * k) + s * 2 ** (n - k) + t
			#     )
			# )
			indicators["anticommuting"][
				l * 2 ** (2 * n - 2 * k)
				+ np.arange(2 ** (n - k), dtype=np.int) * 2 ** (n - k)
				+ t
			] = 1
	# print("TC indicator: {}".format(np.nonzero(indicators["anticommuting"])))
	for l in supports["LA"]:
		for t in supports["TC"]:
			# print(
			#     "index for l = {}, s = {}, t = {} is = {}".format(
			#         l, s, t, l * 2 ** (2 * n - 2 * k) + s * 2 ** (n - k) + t
			#     )
			# )
			indicators["anticommuting"][
				l * 2 ** (2 * n - 2 * k)
				+ np.arange(2 ** (n - k), dtype=np.int) * 2 ** (n - k)
				+ t
			] = 1
	# print("TA indicator: {}".format(np.nonzero(indicators["anticommuting"])))
	# print(
	#     "1 - commuting - anticommuting\n{}".format(
	#         np.nonzero(1 - indicators["commuting"] - indicators["anticommuting"])
	#     )
	# )
	indices = {"commuting": None, "anticommuting": None}
	(indices["commuting"],) = np.nonzero(indicators["commuting"])
	(indices["anticommuting"],) = np.nonzero(indicators["anticommuting"])
	return indices


def GetCommutingInSet(gens, op, parity=0, props="set"):
	# Compute the indices of group elements that (anti-)commute with a given operator.
	if props == "group":
		size = 2 ** gens.shape[0]
	else:
		size = gens.shape[0]
	indicator = np.zeros(size, dtype=np.int)
	for i in range(size):
		if props == "group":
			select = np.array(
				list(map(np.int8, np.binary_repr(i, width=gens.shape[0]))),
				dtype=np.int8,
			)
			if i == 0:
				group_op = np.zeros(gens.shape[1], dtype=np.int)
			else:
				(group_op, __) = PauliProduct(*gens[np.nonzero(select)])
		else:
			group_op = gens[i, :]
		indicator[i] = (parity + int(IsCommuting(op, group_op))) % 2
		# print(
		#     "O = {}, G = {}, commuting = {}.".format(
		#         op, group_op, qc.IsCommuting(op, group_op)
		#     )
		# )
	(indices,) = np.nonzero(indicator)
	return indices


def ComputeLSTOrdering(qcode, ops):
	"""
	Compute the ordering of the given Pauli operators in the Pauli group indexed by LST.
	"""
	ordering = np.zeros(len(ops), dtype=np.int)
	for i in range(len(ops)):
		ordering[i] = qcode.GetPositionInLST(qcode, ops[i])
	return ordering


def GetOperatorsForLSTIndex(qcode, indices):
	r"""
	Get the operators for the index in LST ordering.
	The index of logical operators:
	0 --> I = 00 = 0
	1 --> X = 10 = 2
	2 --> Y = 11 = 3
	3 --> Z = 01 = 1
	"""
	log_ordering = np.array([0, 2, 3, 1], dtype=np.int)

	nstabs = 2 ** (qcode.N - qcode.K)
	ops = np.zeros((len(indices), qcode.N), dtype=np.int)
	phases = np.zeros(len(indices), dtype=np.complex128)
	for i in range(len(indices)):
		pure_index = indices[i] % nstabs
		(pure_op, pure_ph) = GetElementInGroup(pure_index, qcode.T)
		stab_index = (indices[i] // nstabs) % nstabs
		(stab_op, stab_ph) = GetElementInGroup(stab_index, qcode.S)
		log_index = indices[i] // (nstabs * nstabs)
		(log_op, log_ph) = GetElementInGroup(log_ordering[log_index], qcode.L)
		if log_ordering[log_index] == 3:
			log_ph *= 1j
		# print(
		#     "pure_index = {}, pure_op = {}, stab_index = {}, stab_op = {}, log_index = {}, log_op = {}".format(
		#         pure_index, pure_op, stab_index, stab_op, log_index, log_op
		#     )
		# )
		(ops[i, :], phases[i]) = PauliProduct(pure_op, stab_op, log_op)
		phases[i] *= stab_ph * log_ph * pure_ph
	return (ops, phases)


def GetOperatorsForTLSIndex(qcode, indices):
	r"""
	Get the operators for the index in TLS ordering.
	The index of logical operators:
	0 --> I = 00 = 0
	1 --> X = 10 = 2
	2 --> Y = 11 = 3
	3 --> Z = 01 = 1
	"""
	log_ordering = np.array([0, 2, 3, 1], dtype=np.int)

	nstabs = 2 ** (qcode.N - qcode.K)
	nlogs = 4 ** qcode.K
	ops = np.zeros((len(indices), qcode.N), dtype=np.int)
	phases = np.zeros(len(indices), dtype=np.complex128)
	for i in range(len(indices)):
		stab_index = indices[i] % nstabs
		(stab_op,stab_ph) = GetElementInGroup(stab_index, qcode.S)

		log_index = (indices[i] // nstabs) % nlogs
		(log_op, log_ph) = GetElementInGroup(log_ordering[log_index], qcode.L)
		if log_ordering[log_index] == 3:
			log_ph *= 1j

		pure_index = indices[i] // (nlogs * nstabs)
		(pure_op, pure_ph) = GetElementInGroup(pure_index, qcode.T)

		# print(
		#     "pure_index = {}, pure_op = {}, stab_index = {}, stab_op = {}, log_index = {}, log_op = {}".format(
		#         pure_index, pure_op, stab_index, stab_op, log_index, log_op
		#     )
		# )
		(ops[i, :], phases[i]) = PauliProduct(pure_op, stab_op, log_op)
		phases[i] *= stab_ph * log_ph * pure_ph
	return (ops, phases)


def GetElementInGroup(group_index, gens):
	"""
	Get element given its index and weight_enumerators
	"""
	if group_index == 0:
		return (np.zeros(gens.shape[1], dtype=np.int),1)
	include_gens = np.array(
		list(map(np.int8, np.binary_repr(group_index, width=gens.shape[0]))),
		dtype=np.int8,
	)
	# print(
	#     "Binary = {}, include_gens = {}".format(
	#         np.binary_repr(group_index, width=gens.shape[0]), include_gens
	#     )
	# )
	(element,phase) = PauliProduct(*gens[np.nonzero(include_gens)])
	return (element,phase)


if __name__ == "__main__":
	# Load a particular type of error correcting code
	codename = sys.argv[1]
	qcode = QuantumErrorCorrectingCode("%s" % (codename))
	Load(qcode)
	Print(qcode)
	IsCanonicalBasis(qcode.S, qcode.L, qcode.T, verbose=1)
	# print(
	#     "logical action\n%s\n phases\n%s"
	#     % (np.array_str(qcode.normalizer), np.array_str(qcode.normphases))
	# )
