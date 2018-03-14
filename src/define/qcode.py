import os
import sys
import numpy as np

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
		self.defnfile = ("./../code/%s/%s.txt" % (name, name))
		with open(self.defnfile, 'r') as fp:
			infoline = 0
			for line in fp:
				if (not (line[0] == "#")):
					(param, value) = map(lambda val: val.strip(" "), line.strip("\n").strip(" ").split(" "))
					if (param == "name"):
						self.name = value
					elif (param == "N"):
						self.N = int(value)
					elif (param == "K"):
						self.K = int(value)
					elif (param == "D"):
						self.D = int(value)
					else:
						pass
		# Operator basis for the quantum code
		self.paulibasis = np.zeros((4**(self.K + 2), 2**(self.K + 2), 2**(self.K + 2)), dtype = np.complex128)
		self.conjugations = np.zeros((2, 4**self.K, 4**self.K), dtype = np.int)
		self.StabilizerGenerators = np.zeros((self.N - self.K, self.N), dtype = np.complex128)
		self.LogicalGenerators = np.zeros((2 * self.K, self.N), dtype = np.complex128)
		self.PureErrorGenerators = np.zeros((self.N - self.K, self.N), dtype = np.complex128)
		# Preprocessing for error correction
		self.StabilizerSyndromeSigns = np.zeros((2**(self.N - self.K), 2**(self.N - self.K)), dtype = np.int)
		self.LogicalActionOperators = np.zeros((4**self.K, 2**(self.N - self.K), self.N), dtype = np.int)
		self.LogicalActionPhases = np.zeros((4**self.K, 2**(self.N - self.K)), dtype = np.complex128)
		self.hardLookUp = np.zeros((2**(self.N - self.K), 2), dtype = np.int)
		self.LGensFname = ("./../code/%s/%s_LogicalGenerators.npy" % (self.name, self.name))
		self.SGensFname = ("./../code/%s/%s_StabilizerGenerators.npy" % (self.name, self.name))
		self.TGensFname = ("./../code/%s/%s_PureErrorGenerators.npy" % (self.name, self.name))
		self.stabSyndSignsFname = ("./../code/%s/%s_StabilizerSyndromeSigns.npy" % (self.name, self.name))
		self.LAOpsFname = ("./../code/%s/%s_LogicalActionOperators.npy" % (self.name, self.name))
		self.LAPhaseFname = ("./../code/%s/%s_LogicalActionPhases.npy" % (self.name, self.name))
		self.lookupFname = ("./../code/%s/%s_syndLookUp.npy" % (self.name, self.name))
		self.conjfname = ("./../code/%s/Clifford_conjugations.npy" % (self.name))
		self.pbasisfname = ("./../code/%s/paulibasis_3qubits.npy" % (self.name))
	
	def Load(self):
		# Load all the essential elements for a QECC from file
		# Load the generators of the code
		if (os.path.isfile(self.SGensFname)):
			# print("\033[93mReading from %s.\033[0m" % (self.SGensFname))
			# print np.load(self.SGensFname)
			self.StabilizerGenerators = np.load(self.SGensFname)
		else:
			# Read the symbolic generators from a file
			symbolicSGensFname = ("%s.txt" % self.SGensFname[:-4])
			self.StabilizerGenerators = InterprettPauliSymbols(symbolicSGensFname)

		if (os.path.isfile(self.LGensFname)):
			self.LogicalGenerators = np.load(self.LGensFname)
		else:
			# Read the symbolic generators from a file
			symbolicLGensFname = ("%s.txt" % self.LGensFname[:-4])
			self.LogicalGenerators = InterprettPauliSymbols(symbolicLGensFname)

		if (os.path.isfile(self.TGensFname)):
			self.PureErrorGenerators = np.load(self.TGensFname)
		else:
			# Read the symbolic generators from a file
			symbolicTGensFname = ("%s.txt" % self.TGensFname[:-4])
			self.PureErrorGenerators = InterprettPauliSymbols(symbolicTGensFname)

		# Sings in front of each stabilizer element in the syndrome projectors
		if (not (os.path.isfile(self.stabSyndSignsFname))):
			ConstructSyndromeProjectors(self)
			np.save(self.stabSyndSignsFname, self.StabilizerSyndromeSigns)
		self.StabilizerSyndromeSigns = np.load(self.stabSyndSignsFname)

		# Load the logical actions with the appropriate phase information
		if (not os.path.isfile(self.LAOpsFname)):
			ConstructLogicalAction(self)
			np.save(self.LAOpsFname, self.LogicalActionOperators)
			np.save(self.LAPhaseFname, self.LogicalActionPhases)
		self.LogicalActionOperators = np.load(self.LAOpsFname)
		self.LogicalActionPhases = np.load(self.LAPhaseFname)

		# Load the syndrome lookup table
		if (not (os.path.isfile(self.lookupFname))):
			PrepareSyndromeLookUp(self)
			np.save(self.lookupFname, self.hardLookUp)
		self.hardLookUp = np.load(self.lookupFname)

		# Load the clifford conjugations.
		if (not os.path.isfile(self.conjfname)):
			np.save(self.conjfname, PauliCliffordConjugations())
		self.conjugations = np.load(self.conjfname)

		# Load the 3 qubit Pauli basis.
		if (not os.path.isfile(self.pbasisfname)):
			np.save(self.pbasisfname, ConstructPauliBasis(nqubits = 3))
		self.paulibasis = np.load(self.pbasisfname)
		return None


	def Save(self):
		# Save all the pre-computed data for the ECC
		# Write down the computed quantities for future use
		np.save(self.SGensFname, self.StabilizerGenerators)
		np.save(self.LGensFname, self.LogicalGenerators)
		np.save(self.TGensFname, self.PureErrorGenerators)
		np.save(self.stabSyndSignsFname, self.StabilizerSyndromeSigns)
		np.save(self.LAOpsFname, self.LogicalActionOperators)
		np.save(self.LAPhaseFname, self.LogicalActionPhases)
		np.save(self.lookupFname, self.hardLookUp)
		np.save(self.conjfname, self.conjugations)
		np.save(self.pbasisfname, self.paulibasis)
		return None

	def __str__(self):
		# Override the str(...) method for the Quantum error correcting code object
		# Display the details of the error correcting code
		details = ("Name: %s\nParameters: N = %d, K = %d\nGenerating sets:\nStabilizer\n%s\nLogical\n%s\nPure Error\n%s\n\nPreprocessing for error correction:\nStabilizer syndrome signs\n%s\nLogical Action\n%s\nLogical Action Phases\n%s\nSyndrome look up and weights\n%s"
			% (self.name,
			   self.N,
			   self.K,
			   np.array_str(self.StabilizerGenerators),
			   np.array_str(self.LogicalGenerators),
			   np.array_str(self.PureErrorGenerators),
			   np.array_str(self.StabilizerSyndromeSigns),
			   np.array_str(self.LogicalActionOperators),
			   np.array_str(self.LogicalActionPhases),
			   np.array_str(self.hardLookUp)))
		return details

def ConstructPauliBasis(nqubits):
	# Construct the list of all (4^n) Pauli matrices that act on a given number (n) of qubits.
	Pauli = np.array([[[1, 0], [0, 1]],
					  [[0, 1], [1, 0]],
					  [[0, -1j], [1j, 0]],
					  [[1, 0], [0, -1]]], dtype = np.complex128)
	pbasis = np.zeros((4**nqubits, 2**nqubits, 2**nqubits), dtype = np.complex128)
	for i in range(4**nqubits):
		pvec = ChangeBase(i, 4, nqubits)
		# print("Basis element %d: %s" % (i, np.array_str(pvec, max_line_width = 150)))
		# k-fold tensor product of Pauli matrices.
		element = np.zeros((2, 2), dtype = np.complex128)
		for j in range(2):
			for k in range(2):
				element[j, k] = Pauli[pvec[0], j, k]
		for q in range(1, nqubits):
			element = np.kron(element, Pauli[pvec[q], :, :])
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
	newnumber = np.zeros(width, dtype = int)
	digit = width - 1
	remainder = number
	while (remainder > 0):
		newnumber[digit] = np.mod(remainder, 4)
		remainder = remainder/base
		digit = digit - 1
	return newnumber


def PauliCliffordConjugations():
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
	conjugations = np.zeros((2, 24, 4), dtype = int)
	conjugations[1, :, 0] = 1
	for ci in range(24):
		for pi in range(3):
			(conjugations[0, ci, pi + 1], conjugations[1, ci, pi + 1]) = (symbmap[conj[ci][pi]][0], symbmap[conj[ci][pi]][1])
	return conjugations


def InterprettPauliSymbols(symbolsFname):
	# Read a set of Pauli operators from thier symbolic representation, along with the phase information
	encoding = {"I":0, "X":1, "Y":2, "Z":3}
	with open(symbolsFname, 'r') as sPFid:
		# print("\033[93mOpeneing file %s.\033[0m" % symbolsFname)
		(nOps, nQubits) = map(int, sPFid.readline().strip("\n").strip(" ").split(" "))
		operators = np.zeros((nOps, nQubits + 1), dtype = complex)
		for oi in range(nOps):
			operatorInfo = sPFid.readline().strip("\n").strip(" ").split(" ")
			operators[oi, 0] = complex(operatorInfo[0])
			for qi in range(nQubits):
				operators[oi, qi + 1] = int(encoding[operatorInfo[qi + 1]])
	return operators



def PauliProduct(pauli1, pauli2):
	# Perform a product of two n-qubit Pauli operators, along with the appropriate phase information
	# The multiplication rule for the single qubit Pauli group is given by
	###	  I 	X 	Y 	Z
	# I   I 	X 	Y 	Z
	# X   X 	I 	iZ 	-iY
	# Y   Y 	-iZ	I 	iX
	# Z   Z 	iY 	-iX I
	# We always use the encoding I --> 0, X --> 1, Y --> 2, Z --> 3
	# print("\033[95mTaking a product of the Pauli operators\nP1 = %s\nP2 = %s.\033[0m" % (np.array_str(pauli1, precision = 1, suppress_small = True), np.array_str(pauli2, precision = 1, suppress_small = True)))
	multiplicationTable = np.array([[[0, 1, 2, 3],
							   		 [1, 0, 3, 2],
							   		 [2, 3, 0, 1],
							   		 [3, 2, 1, 0]],
							  		[[1, 1, 1, 1],
							  		 [1, 1, 1j, -1j],
							  		 [1, -1j, 1, 1j],
							  		 [1, 1j, -1j, 1]]], dtype = complex)
	if (pauli1.shape == pauli2.shape):
		product = np.squeeze(np.real(multiplicationTable[0, pauli1, pauli2]).astype(int))
		phase = np.prod(multiplicationTable[1, pauli1, pauli2])

	# if (not (phase.imag == 0)):
		# print("\033[95mTaking a product of the Pauli operators\nP1 = %s\nP2 = %s.\033[0m" % (np.array_str(pauli1, precision = 1, suppress_small = True), np.array_str(pauli2, precision = 1, suppress_small = True)))
	# 	print("\033[95mP1 . P2 = (%f + i %.4e) * %s\033[0m" % (np.real(phase), np.imag(phase), np.array_str(product, precision = 1, suppress_small = True)))

	return (product, phase)



def CollapsePaulis(pauliOperators, globalPhases):
	# Compute the product of a list of Pauli operators
	print("\033[95mGoing to collapse a set of %s Pauli operators.\033[0m" % (np.array_str(pauliOperators.shape, precision = 1, suppress_small = True)))
	(nOps, nQubits) = pauliOperators.shape
	product = np.zeros(nQubits, dtype = int)
	productPhase = 1
	for oi in range(nOps):
		(product, phase) = PauliProduct(product, operators[oi, :])
		productPhase = productPhase * phase * globalPhases[oi]
	return (product, productPhase)


def ConstructSyndromeProjectors(qecc):
	# For each syndrome, construct the operators that projects on to the subspace containing Pauli errors with that syndrome
	# The syndrome projector can be expanded as a linear sum over all the stabilizers, with coefficients that depend on the syndrome.
	# Here we only need these coefficients, they are numbers in {+1, -1}. The coefficient of the stabilizer is +1(-1) if it commutes (anti-commutes) with the Pure error of that particular syndrome.
	qecc.StabilizerSyndromeSigns = np.zeros((2**(qecc.N - qecc.K), 2**(qecc.N - qecc.K)), dtype = int)
	for ti in range(2**(qecc.N - qecc.K)):
		# Interate over the Pure error generators
		combTGens = np.array(map(int, np.binary_repr(ti, width = (qecc.N - qecc.K))), dtype = int)
		for si in range(2**(qecc.N - qecc.K)):
			# Interate over the stabilizer generators
			combSGens = np.array(map(int, np.binary_repr(si, width = (qecc.N - qecc.K))), dtype = int)
			qecc.StabilizerSyndromeSigns[ti, si] = (-1)**np.dot(combTGens, combSGens)
	print("\033[92m\r%d%% done.\033[0m" % (100 * (ti + 1)/float(2**(qecc.N - qecc.K)))),
	sys.stdout.flush()
	print("")
	return None


def ConstructLogicalAction(qecc):
	# For each logical class, construct the list of Pauli errors in that class, along with the appropriate global phase associated to this error
	qecc.LogicalActionOperators = np.zeros((4**qecc.K, 2**(qecc.N - qecc.K), qecc.N), dtype = int)
	qecc.LogicalActionPhases = np.zeros((4**qecc.K, 2**(qecc.N - qecc.K)), dtype = complex)
	ordering = np.array([[0, 3], [1, 2]], dtype = int)
	for li in range(4**qecc.K):
		combLGens = np.array(map(int, np.binary_repr(li, width = (2 * qecc.K))), dtype = int)
		logical = np.zeros(qecc.N, dtype = int)
		logPhase = (1j)**np.prod(combLGens)
		if(li > 0):
			# print("Logical generator combinations: %s." % (np.array_str(combLGens)))
			for lgi in range(2 * qecc.K):
				if (combLGens[lgi] == 1):
					(logical, phase) = PauliProduct(logical, np.real(qecc.LogicalGenerators[lgi, 1:]).astype(int))
					logPhase = logPhase * qecc.LogicalGenerators[lgi, 0] * phase

		for si in range(2**(qecc.N - qecc.K)):
			combSGens = np.array(map(int, np.binary_repr(si, width = (qecc.N - qecc.K))), dtype = int)
			stabilizer = np.zeros(qecc.N, dtype = int)
			stabPhase = 1
			if(si > 0):
				for sgi in range(qecc.N - qecc.K):
					if (combSGens[sgi] == 1):
						(stabilizer, phase) = PauliProduct(stabilizer, np.real(qecc.StabilizerGenerators[sgi, 1:]).astype(int))
						stabPhase = stabPhase * qecc.StabilizerGenerators[sgi, 0] * phase

			# Combine the logical operator with the stabilizers to generate all the operators in the corresponding logical class
			(normalizer, normPhase) = PauliProduct(logical, stabilizer)
			normPhase = normPhase * logPhase * stabPhase

			# print("Generator combinations\nLogical: (%g + i %g) %s\nStabilizer: (%g + i %g) %s\nNormalizer: (%g + i %g) %s." % (np.real(logPhase), np.imag(logPhase), np.array_str(logical), np.real(stabPhase), np.imag(stabPhase), np.array_str(stabilizer), np.real(normPhase), np.imag(normPhase), np.array_str(normalizer)))

			qecc.LogicalActionOperators[ordering[combLGens[0], combLGens[1]], si, :] = normalizer
			qecc.LogicalActionPhases[ordering[combLGens[0], combLGens[1]], si] = normPhase

		print("\r\033[92m%d%% done.\033[0m" % (100 * (li + 1)/float(4**qecc.K))),
		sys.stdout.flush()
	print("")
	return None

def PrepareSyndromeLookUp(qecc):
	# Prepare a lookup table that contains logical corrections (predicted by the min-weight decoder) for every measured syndrome of the code.
	# For every syndrome
	# 	generate the pure error T
	# 	find the logical operator L such that (T.L.S) has least weight over all L and for some S.
	ordering = np.array([[0, 3], [1, 2]], dtype = int)
	qecc.hardLookUp = np.zeros((2**(qecc.N - qecc.K), 2), dtype = int)
	# recoveries = np.zeros((2**(qecc.N - qecc.K), qecc.N), dtype = int)
	for i in range(2**(qecc.N - qecc.K)):
		combTGens = np.array(map(int, np.binary_repr(i, width = (qecc.N - qecc.K))), dtype = int)
		pureError = np.zeros(qecc.N, dtype = int)
		for j in range(qecc.N - qecc.K):
			if (combTGens[j] == 1):
				(pureError, __) = PauliProduct(pureError, np.real(qecc.PureErrorGenerators[j, 1:]).astype(int))
		qecc.hardLookUp[i, 0] = 0
		qecc.hardLookUp[i, 1] = qecc.N
		for l in range(4**qecc.K):
			combLGens = np.array(map(int, np.binary_repr(l, width = (2 * qecc.K))), dtype = int)
			logical = np.zeros(qecc.N, dtype = int)
			for j in range(2 * qecc.K):
				if (combLGens[j] == 1):
					(logical, __) = PauliProduct(logical, np.real(qecc.LogicalGenerators[j, 1:]).astype(int))
			for s in range(2**(qecc.N - qecc.K)):
				combSGens = np.array(map(int, np.binary_repr(s, width = (qecc.N - qecc.K))), dtype = int)
				stabilizer = np.zeros(qecc.N, dtype = int)
				for j in range(qecc.N - qecc.K):
					if (combSGens[j] == 1):
						(stabilizer, __) = PauliProduct(stabilizer, np.real(qecc.StabilizerGenerators[j, 1:]).astype(int))
				correction = np.zeros(qecc.N, dtype = int)
				(correction, __) = PauliProduct(correction, stabilizer)
				(correction, __) = PauliProduct(correction, logical)
				(correction, __) = PauliProduct(correction, pureError)
				weight = np.count_nonzero(correction.astype(int) > 0)
				# print("correction of weight %d\n%s" % (weight, np.array_str(correction)))
				if (weight <= qecc.hardLookUp[i, 1]):
					qecc.hardLookUp[i, 0] = ordering[combLGens[0], combLGens[1]]
					qecc.hardLookUp[i, 1] = weight
					# for q in range(qecc.N):
					# 	recoveries[i, q] = correction[q]
		# print("Syndrome %d: %s\n\tPure error\n\t%s\n\tCorrection\n\t%s" % (i, np.array_str(combTGens), np.array_str(pureError), np.array_str(recoveries[i])))

	return None


def Supports(qecc):
	# Test if there is any Logical operator L and a stabilizer S such that the support of the logical operator is contained entirely within the support of the Stabilizer
	for sti in range(2**(qecc.N - qecc.K)):
		supportStab = np.array(np.nonzero(qecc.LogicalActionOperators[0, sti, :]))[0, :]
		for li in range(1, 4**qecc.K):
			for si in range(2**(qecc.N - qecc.K)):
				supportLog = np.array(np.nonzero(qecc.LogicalActionOperators[li, si, :]))[0, :]
				tracePreserving = 1 - IsSubset(supportLog, supportStab)
				# for qi in range(qecc.N):
				# 	if ((qecc.LogicalActionOperators[li, si, qi] > 0) and (qecc.LogicalActionOperators[0, sti, qi] == 0)):
				# 		tracePreserving = 1
				if (tracePreserving == 0):
					print("\033[93mStabilizer\n%s\nLogical\n%s.\033[0m" % (np.array_str(qecc.LogicalActionOperators[0, sti, :], precision = 1, suppress_small = True), np.array_str(qecc.LogicalActionOperators[li, si, :], precision = 1, suppress_small = True)))
	return None


def IsSubset(array1, array2):
	# Test if array1 is a subset of array2
	uniqueArray1 = np.unique(array1)
	uniqueArray2 = np.unique(array2)
	common = np.intersect1d(uniqueArray1, uniqueArray2)
	if (common.shape[0] == uniqueArray1.shape[0]):
		return 1
	else:
		return 0
	return None

def GenerateGroup(generators):
	group = np.zeros((generators.shape[0], generators.shape[1], generators.shape[1]), dtype = float)
	for gi in range(2**generators.shape[0]):
		gselect = map(int, eval("'{:0" + str(generators.shape[0]) + "b}'" + ".format(" + str(gi) + ")"))
		group[gi, :, :] = Product(generators, gselect)
	return group

def Product(generators, gselect):
	product = np.identity(generators.shape[1])
	for gi in range(generators.shape[0]):
		if gselect[gi] == 1:
			product = np.dot(product, generators[gi, :, :])
	return product

def Distinct(elements):
	# output elements of the input set that are distinct, up to a constant multiplicative factor
	duplicate = np.zeros(elements.shape[0], dtype = int)
	for ei in range(elements.shape[0]):
		for oei in range(ei + 1, elements.shape[0]):
			duplicate[ei] = duplicate[ei] + IsSame(elements[ei, :, :], elements[oei, :, :])
	# take only those elements whose corresponding place in the duplicate array is 0.
	unique = elements[np.array(np.nonzero(duplicate == 0)[0]), :, :]
	return unique

def IsSame(matrix1, matrix2):
	# determine if two matrices are same up to a constant multiplicative factor
	if (np.any(matrix1 == 0).shape[0] + np.any(matrix2 == 0).shape[0] > 0):
		return 0
	else:
		if (np.linalg.matrix_rank(np.row_stack((np.ravel(matrix1), np.ravel(matrix2)))) < 2):
			return 0
	return 1



if __name__ == '__main__':
	# Load a particular type of error correcting code
	codename = sys.argv[1]
	qcode = QuantumErrorCorrectingCode("code/%s/%s.txt" % (codename, codename))
	qcode.Load()

	# print("Stabilizer generators")
	# print np.array_str(qcode.StabilizerGenerators, precision = 1, suppress_small = True)

	# print("Logical generators")
	# print np.array_str(qcode.LogicalGenerators, precision = 1, suppress_small = True)

	# print("Pure error generators")
	# print np.array_str(qcode.PureErrorGenerators, precision = 1, suppress_small = True)

	# ConstructSyndromeProjectors(qcode)
	# print("Signs in the syndrome projector")
	# print np.array_str(qcode.StabilizerSyndromeSigns, precision = 1, suppress_small = True)

	# ConstructLogicalAction(qcode)

	# print("Syndrome look up table")
	# PrepareSyndromeLookUp(qcode)
	# print np.array_str(qcode.hardLookUp, precision = 1, suppress_small = True)

	print("\033[92mDetails of the error correcting code which is to be used for this simulation\n**********************\n%s\n**********************\033[0m" % (str(qcode)))
	qcode.Save()

	# print("\033[92m*********************************\nChecking if any logical operator is supported entirely in the supoort of any stabilizer.\033[0m")
	# Supports(qcode)
