import numpy as np







def SymplecticToOperator(sympvec):
	# convert a symplectic form to an operator form
	encoding = np.array([[0, 3], [1, 2]], dtype = np.int8)
	op = np.zeros(sympvec.shape[0]/2, dtype = np.int8)
	for i in range(op.shape[0]):
		op[i] = encoding[sympvec[i], sympvec[op.shape[0] + i]]
	return op

def OperatorToSymplectic(pauliop):
	# Convert the operator in the encoding I -> 0, X -> 1, Y -> 2, Z -> 3 to its symplectic form.
	# print("operator = %s" % (pauliop))
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
	# print("symplectic\n%s" % (np.array_str(sympmat)))
	return sympmat

def ConvertToOperator(sympmat):
	# Convert a matrix of symplectic vectors to operators in the encoding: I --> 0, X --> 1, Y --> 2, Z --> 3.
	encoding = np.array([[0, 3], [1, 2]], dtype = np.int8)
	nq = sympmat.shape[1]/2
	pauliops = np.zeros((sympmat.shape[0], nq), dtype = np.int8)
	# print("sympmat\n%s\nnq = %d" % (sympmat, nq))
	for i in range(sympmat.shape[0]):
		for j in range(nq):
			pauliops[i, j] = encoding[sympmat[i, j], sympmat[i, nq + j]]
	return pauliops


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
					# print("r%d <--> r%d" % (i, j))
					mat[[i, j], :] = mat[[j, i], :]
					break
		if (mat[i, i] == 0):
			# look for a column to the right (up to n) that has a 1 in the i-th row.
			for j in range(i + 1, mat.shape[1]):
				if (mat[i, j] == 1):
					# swap columns i with j and columns (mat.shape[0] + i) with (mat.shape[0] + j).
					# print("c%d <--> c%d" % (i, j))
					mat[:, [i, j]] = mat[:, [j, i]]
					# record the column swaps so that they can be undone at the end.
					cols[[i, j]] = cols[[j, i]]
					# colswap[swaps, 0] = i
					# colswap[swaps, 1] = j
					# swaps = swaps + 1
					break
		# Row reduce the matrix
		for j in range(mat.shape[0]):
			if (not (i == j)):
				if (mat[j, i] == 1):
					# print("r%d --> r%d + r%d" % (j, i, j))
					mat[j, :] = np.mod(mat[i, :] + mat[j, :], 2)
		# print("mat\n%s" % (np.array_str(mat)))
	# print("swaps = %d\nRow reduced form\n%s" % (swaps, np.array_str(mat)))

	# print("mat\n%s\ndim = %d x %d.\nraw kernel\n%s" % (np.array_str(mat), mat.shape[0], mat.shape[1], np.array_str(np.hstack((np.transpose(mat[:, mat.shape[0]:]), np.identity((mat.shape[1] - mat.shape[0]), dtype = np.int8))))))
	# print("raw prod = %s" % np.array_str(np.mod(np.dot(mat, np.transpose(np.hstack((np.transpose(mat[:, mat.shape[0]:]), np.identity((mat.shape[1] - mat.shape[0]), dtype = np.int8))))), 2)))

	# print("column ordering\n%s" % (np.array_str(cols)))
	# Decude the Null space vectors and undo the column permutations
	# print("np.transpose(mat.shape[:, mat.shape[0]:])\n%s" % (np.array_str(np.transpose(mat[:, mat.shape[0]:]))))
	null = np.hstack((np.transpose(mat[:, mat.shape[0]:]), np.identity((mat.shape[1] - mat.shape[0]), dtype = np.int8)))[:, cols]
	# print("null\n%s" % (np.array_str(null)))
	return null


def ComputeLogicals(normalizer):
	# Given the normalizer, compute a canonical basis for the logical operators.
	# We will use the Symplectic Gram Schmidt Orthogonialization method mentioned in arXiv: 0903.5526v1.
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
	# print("nlogs = %d\ncanonical N(S)\n%s\nLogical indices\n%s" % (nlogs, np.array_str(ConvertToOperator(normalizer)), np.array_str(logs)))
	logicals = np.zeros((2 * (nlogs - 1), normalizer.shape[1]), dtype = np.int8)
	for i in range(nlogs - 1):
		logicals[[i, nlogs - 1 + i], :] = normalizer[np.nonzero(abs(logs) == (i + 1)), :]
	stabilizers = np.squeeze(normalizer[np.nonzero(abs(logs) == 0), :])
	# print("stabilizers\n%s\nlogicals\n%s" % (np.array_str(stabilizers), np.array_str(logicals)))
	return np.vstack((stabilizers, logicals))


def ComputePureErrors(stabgens, loggens):
	# Compute the generators for the pure errors, given the stabilizer and logical generators.
	# The i-th pure error must anticommute with the i-th stabilizer generator and commute with all other operators including pure errors.
	print("Function: ComputePureErrors")
	# print("stabgens\n%s\nloggens\n%s" % (np.array_str(stabgens), np.array_str(loggens)))
	nq = stabgens.shape[1]/2
	nl = nq - stabgens.shape[0]
	# print("nq = %d, nl = %d" % (nq, nl))
	puregens = np.zeros((nq - nl, 2 * nq), dtype = np.int8)
	for i in range(nq - nl):
		# cons = np.vstack((loggens, stabgens[np.arange(nq - nl)!=i], puregens[:i, :]))
		# print("cons at i = %d\n%s" % (i, np.array_str(ConvertToOperator(cons))))
		sols = NullSpace(np.vstack((loggens, stabgens[np.arange(nq - nl)!=i])))
		# print("operators that commute with all but the %d-th stabilizer generator\n%s" % (i, np.array_str(sols)))
		# Check which of the elements from the solutions anticommute with the i-th stabilizer.
		# Set that element to be the i-th pure error and add it to the list of constraints
		for j in range(sols.shape[0]):
			if (SymplecticProduct(sols[j, :], stabgens[i, :]) == 1):
				# commute = 1
				# for k in range(nq - nl):
				# 	if (not (k == i)):
				# 		if (SymplecticProduct(sols[j, :], stabgens[k, :]) == 1):
				# 			commute = 0
				# 			break
				# for k in range(2 * nl):
				# 	if (SymplecticProduct(sols[j, :], loggens[k, :]) == 1):
				# 		commute = 0
				# 		break
				# for k in range(i):
				# 	if (SymplecticProduct(sols[j, :], puregens[k, :]) == 1):
				# 		commute = 0
				# 		break
				# if (commute == 1):
				break
		puregens[i, :] = sols[j, :]
		NullTest(np.vstack([loggens, stabgens[np.arange(nq - nl)!=i], puregens[:i, :]]), puregens[i, np.newaxis, :])
		print("pure errors\n%s" % (np.array_str(ConvertToOperator(puregens[:(i + 1), :]))))
	# Fixing the commutation relations between pure errors.
	# 1. If Ti anti commutes with Sj, for j not equal to i, then: Ti -> Ti * Tj
	# 2. If Ti anti commutes with Lj, then: Ti -> Ti * Mj where Mj is the logical operator conjugate to Lj
	# 3. If Ti anti commutes with Tj, then Ti -> Ti * Sj
	for i in range(nq - nl):
		for j in range(nq - nl):
			if (not (i == j)):
				if (SymplecticProduct(puregens[i, :], stabgens[j, :]) == 1):
					puregens[i, :] = np.mod(puregens[i, :] + puregens[j, :], 2)
	for i in range(nq - nl):
		for j in range(2 * nl):
			if (SymplecticProduct(puregens[i, :], loggens[j, :]) == 1):
				puregens[i, :] = np.mod(puregens[i, :] + loggens[(j + nl) % (2 * nl), :], 2)
	for i in range(nq - nl):
		for j in range(nq - nl):
			if (SymplecticProduct(puregens[i, :], puregens[j, :]) == 1):
				puregens[i, :] = np.mod(puregens[i, :] + stabgens[j, :], 2)

	# print("Pure error generators\n%s" % (np.array_str(puregens)))
	return puregens


def IsCanonicalBasis(S, L, T, verbose = 1):
	# test if a basis is a canonical basis and display the commutation relations
	(nq, nl) = (S.shape[1], S.shape[1] - S.shape[0])
	canonical = np.vstack((ConvertToSympectic(S), ConvertToSympectic(L), ConvertToSympectic(T)))
	commutation = np.zeros((canonical.shape[0], canonical.shape[0]), dtype = np.int8)
	if (verbose == 1):
		print("\033[2m")
		print("N = %d, K = %d" % (nq, nl))
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
				print("commutation[i, %s] = %s\n%s" % (np.array_str(np.arange(2 * nq)!=(nq + 2 * nl + i)), np.array_str(commutation[i, np.arange(2 * nq)!=(nq + 2 * nl + i)]), np.array_str(commutation[i, np.arange(2 * nq)!=(nq + 2 * nl + i)])))
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


def NullTest(vspace, kernel):
	# test if all the vectors in the null space are orthogonal (with respect to the symplectic product) to the original space.
	print("Null space test with\nvspace\n%s\nkernel\n%s" % (np.array_str(ConvertToOperator(vspace)), np.array_str(ConvertToOperator(kernel))))
	product = np.zeros((vspace.shape[0], kernel.shape[0]), dtype = np.int8)
	for i in range(vspace.shape[0]):
		for j in range(kernel.shape[0]):
			product[i, j] = SymplecticProduct(vspace[i, :], kernel[j, :])
	print("product\n%s" % (np.array_str(product)))
	return None

def ShortestPathTest():
	# from ./../define import chanreps as crep
	## Test the Shortest path algorithm.
	reprs = ["krauss", "choi", "chi", "process", "stine"]
	# Mapping functions:
	# 1. Choi to process
	# 2. process to choi
	# 3. stine to krauss
	# 4. krauss to stine
	# 5. krauss to process
	# 6. krauss to choi
	# 7. choi to krauss
	# 8. process to chi
	# 9. chi to process
	# 			Krauss 	Choi 	Chi 	Process 	Stine
	# Krauss 	 0 		 6 		-1 		 5 			 4
	# Choi 	 	 7		 0 		-1 		 1 			-1
	# Chi 	 	-1		-1		 0 		 9 			-1
	# Process  	-1		 2		 8		 0 			-1
	# Stine 	 3		-1		-1		-1			 0

	mappings = np.array([[0, 6, -1, 5, 4],
						 [7, 0, -1, 1, -1],
						 [-1, -1, 0, 9, -1],
						 [-1, 2, 8, 0, -1],
						 [3, -1, -1, -1, 0]], dtype = np.int8)
	costs = np.array([[0, 1, -1, 1, 5],
					  [1, 0, -1, 1, -1],
					  [-1, -1, 0, 1, -1],
					  [-1, 1, 1, 0, -1],
					  [5, -1, -1, -1, 0]], dtype = np.int8)

	initial = "process"
	final = "krauss"
	
	map_process = crep.ShortestPath(costs, initial, final, reprs)
	print("Mapping procedure: %s" % (" -> ".join(map_process)))
	return None


def CanonicalBasisTest():
	# Test the Null space algorithm
	S = ConvertToSympectic(np.real(np.load("Steane_StabilizerGenerators.npy")).astype(np.int8)[:, 1:])
	nq = S.shape[1]/2
	nl = nq - S.shape[0]
	# print("S\n%s" % (np.array_str(ConvertToOperator(S))))
	# null = NullSpace(np.copy(S))
	# print("N(S)\n%s" % (np.array_str(ConvertToOperator(null))))
	# product = np.zeros((nq - nl, nq + nl), dtype = np.int8)
	# nullops = ConvertToOperator(null)
	# Sops = ConvertToOperator(S)
	# for i in range(nq - nl):
		# for j in range(nq + nl):
			# product[i, j] = SymplecticProduct(S[i, :], null[j, :])
			# print("[%s, %s] = %d" % (np.array_str(Sops[i, :]), np.array_str(nullops[j, :]), product[i, j]))
	# print("product\n%s" % (np.array_str(product)))
	normalizer = ComputeLogicals(NullSpace(S))
	print("S\n%s\nL\n%s" % (np.array_str(ConvertToOperator(normalizer[:(nq - nl), :])), np.array_str(ConvertToOperator(normalizer[(nq - nl):, :]))))
	
	puregens = ComputePureErrors(normalizer[:(nq - nl), :], normalizer[(nq - nl):, :])
	print("Pure error generators\n%s" % (np.array_str(ConvertToOperator(puregens))))
	
	canonical = np.vstack((normalizer, puregens))
	print("nq = %d, nl = %d\ncanonical\n%s" % (nq, nl, np.array_str(canonical)))
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
			print("      %d" % (SymplecticProduct(canonical[i, :], canonical[j, :]))),
		print("")
	return None

if __name__ == '__main__':
	CanonicalBasisTest()