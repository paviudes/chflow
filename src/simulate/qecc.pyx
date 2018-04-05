#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False
from libc.stdlib cimport malloc, free
from constants cimport constants_t
from memory cimport simul_t
from printfuns cimport PrintDoubleArray1D, PrintComplexArray2D

cdef extern from "math.h" nogil:
	long double powl(long double base, long double expo)
	
cdef extern from "complex.h" nogil:
	long double creall(long double complex cnum)
	long double cimagl(long double complex cnum)
	long double complex conjl(long double complex cnum)


cdef int InitQECC(qecc_t *qecc, int nclifford):
	# Initialize the elements of a quantum error correcting code
	qecc[0].nlogs = 4**qecc[0].K
	qecc[0].nstabs = 2**(qecc[0].N - qecc[0].K)

	cdef int i, s, r, c, g
	qecc[0].projector = <int **>malloc(qecc[0].nstabs * sizeof(int *))
	for s in range(qecc[0].nstabs):
		qecc[0].projector[s] = <int *>malloc(qecc[0].nstabs * sizeof(int))
		for g in range(qecc[0].nstabs):
			qecc[0].projector[s][g] = 0

	qecc[0].action = <int ***>malloc(qecc[0].nlogs * sizeof(int **))
	for i in range(qecc[0].nlogs):
		qecc[0].action[i] = <int **>malloc(qecc[0].nstabs * sizeof(int *))
		for g in range(qecc[0].nstabs):
			qecc[0].action[i][g] = <int *>malloc(qecc[0].N * sizeof(int))
			for r in range(qecc[0].N):
				qecc[0].action[i][g][r] = 0

	qecc[0].phases = <complex128_t **>malloc(qecc[0].nlogs * sizeof(complex128_t *))
	for i in range(qecc[0].nlogs):
		qecc[0].phases[i] = <complex128_t *>malloc(qecc[0].nstabs * sizeof(complex128_t))
		for g in range(qecc[0].nstabs):
			qecc[0].phases[i][g] = 0 + 0 * 1j

	qecc[0].algebra = <int ***>malloc(2 * sizeof(int **))
	for i in range(2):
		qecc[0].algebra[i] = <int **>malloc(nclifford * sizeof(int *))
		for r in range(nclifford):
			qecc[0].algebra[i][r] = <int *>malloc(qecc[0].nlogs * sizeof(int))
			for c in range(qecc[0].nlogs):
				qecc[0].algebra[i][r][c] = 0
	return 0


cdef int FreeQECC(qecc_t *qecc, int nclifford):
	# free the memory assigned to a quantum error correcting code
	cdef int i, s, r, c, g
	for s in range(qecc[0].nstabs):
		free(<void *>qecc[0].projector[s])
	free(<void *>qecc[0].projector)

	for i in range(qecc[0].nlogs):
		for g in range(qecc[0].nstabs):
			free(<void *>qecc[0].action[i][g])
		free(<void *>qecc[0].action[i])
	free(<void *>qecc[0].action)

	for i in range(qecc[0].nlogs):
		free(<void *>qecc[0].phases[i])
	free(<void *>qecc[0].phases)

	for i in range(2):
		for r in range(nclifford):
			free(<void *>qecc[0].algebra[i][r])
		free(<void *>qecc[0].algebra[i])
	free(<void *>qecc[0].algebra)
	return 0


cdef int SingleShotErrorCorrection(int level, int isPauli, int frame, qecc_t *qecc, simul_t *sim, constants_t *consts) nogil:
	# Compute the effective logical channel, when error correction is applied over a set of input physical channels
	# with gil:
		# print("Constructing the full process matrix")
	GetFullProcessMatrix(qecc, sim, isPauli, level)
	## Compute the probabilities of all the syndromes
	ComputeSyndromeDistribution(qecc, sim, isPauli, consts[0].atol)
	## Maximum Likelihood Decoding (MLD) -- For every syndrome, compute the probabilities of the logical classes and pick the one that is most likely.
	# with gil:
		# print("Maximum likelihood decoding")
	MLDecoder(qecc, sim, consts, frame, isPauli)
	## For every syndrome, apply the correction and compute the new effective choi matrix of the single qubit logical channel
	# with gil:
		# print("Computing the effective logical channels")
	ComputeEffectiveStates(qecc, sim, consts, isPauli)
	## Convert the effective channels from choi representation to the process matrix form
	cdef int s
	for s in range(qecc[0].nstabs):
		ChoiToProcess(sim[0].effprocess[s], sim[0].effective[s], consts[0].pauli)
	return 0


cdef int ChoiToProcess(long double **process, complex128_t **choi, complex128_t pauli[4][2][2]) nogil:
	# Convert from the Choi matrix to the Chi matrix, of a quantum channel
	# CHI[a,b] = Trace( Choi * (Pb \otimes Pa^T) )
	cdef:
		int i, j, k, l
		complex128_t contribution
	for i in range(4):
		for j in range(4):
			contribution = 0 + 0 * 1j
			for k in range(4):
				for l in range(k + 1):
					contribution = contribution + choi[k][l] * pauli[j][l/2][k/2] * pauli[i][k%2][l%2]
				for l in range(k + 1, 4):
					contribution = contribution + conjl(choi[l][k]) * pauli[j][l/2][k/2] * pauli[i][k%2][l%2]
			process[i][j] = <long double>(creall(contribution))
	return 0

cdef int ComputeEffectiveStates(qecc_t *qecc, simul_t *sim, constants_t *consts, int isPauli) nogil:
	# Compute the Choi matrix of the effective logical channel, after applying noise + error correction.
	# The effective Choi matrix (state) is given by: L Ts PI_s E(r) PI_s T_s L which can be written in its unencoded form by expanding in the Pauli basis
	# un-encoded state = sum_(a,b) Tr( L Ts PI_s E(r) PI_s T_s L . (PI_0 Pa PI_0 \o Pb))
	# CHOI[a,b] = sum_(i: Pi is in the logical class of Pb) sum_(j: Pj is in the logical class of [L Pa L]) CHI[i,j] * (-1)^(P_j) * (-1)**(if {L, Pb} == 0)
	cdef:
		int r, c, s, a, b, i, j
		complex128_t coeff
	for s in range(qecc[0].nstabs):
		for r in range(qecc[0].nlogs):
			for c in range(qecc[0].nlogs):
				sim[0].effective[s][r][c] = 0.0 + 0.0 * 1j
		if (sim[0].syndprobs[s] > consts[0].atol):
			if (isPauli == 0):
				for a in range(qecc[0].nlogs):
					for b in range(qecc[0].nlogs):
						coeff = 0 + 0 * 1j
						for i in range(qecc[0].nstabs):
							for j in range(qecc[0].nstabs):
								coeff = coeff + sim[0].process[b][qecc[0].algebra[0][sim[0].corrections[s]][a]][i][j] * qecc[0].projector[s][j]
						for r in range(qecc[0].nlogs):
							for c in range(1 + r):
								sim[0].effective[s][r][c] = sim[0].effective[s][r][c] + coeff * powl(-1, (<int>(b == 2))) * qecc[0].algebra[1][sim[0].corrections[s]][a] * consts[0].pauli[a][r/2][c/2] * consts[0].pauli[b][r%2][c%2]
			else:
				for a in range(qecc[0].nlogs):
					coeff = 0.0 + 0.0 * 1j
					for i in range(qecc[0].nstabs):
						coeff = coeff + sim[0].process[a][qecc[0].algebra[0][sim[0].corrections[s]][a]][i][i] * qecc[0].projector[s][i]
					for r in range(qecc[0].nlogs):
						for c in range(1 + r):
							sim[0].effective[s][r][c] = sim[0].effective[s][r][c] + coeff * powl(-1, (<int>(a == 2))) * qecc[0].algebra[1][sim[0].corrections[s]][a] * consts[0].pauli[a][r/2][c/2] * consts[0].pauli[a][r%2][c%2]
			# Normalization and rounding
			for r in range(qecc[0].nlogs):
				for c in range(1 + r):
					if (creall(sim[0].effective[s][r][c]) < (consts[0].atol)):
						sim[0].effective[s][r][c] = 0 + 1j * cimagl(sim[0].effective[s][r][c])
					if (cimagl(sim[0].effective[s][r][c]) < (consts[0].atol)):
						sim[0].effective[s][r][c] = creall(sim[0].effective[s][r][c]) + 1j * 0.0
			for r in range(qecc[0].nlogs):
				for c in range(1 + r, qecc[0].nlogs):
					sim[0].effective[s][r][c] = conjl(sim[0].effective[s][c][r])

			for r in range(qecc[0].nlogs):
				for c in range(qecc[0].nlogs):
					sim[0].effective[s][r][c] = sim[0].effective[s][r][c]/(sim[0].syndprobs[s] * (4 * <long double>(qecc[0].nstabs)))
			# if (IsState(sim[0].effective[s], consts[0].atol) == 0):
			# 	print("The following density matrix is not a valid quantum state!")
			# 	PrintComplexArray2D(sim[0].effective[s], ("rho_%d, P(s) = %.4e, isPauli = %d" % (s, sim[0].syndprobs[s], isPauli)), 4, 4)
	return 0


cdef int MLDecoder(qecc_t *qecc, simul_t *sim, constants_t *consts, int currentframe, int isPauli) nogil:
	# Perform maximum likelihood decoding.
	# Compute the probabilities of the logical classes, considitioned on a particular syndrome
	# The ML Decoder picks the logical error which belongs to the class that has the maximum probability.
	# The probability of a logical class is P(L|s) = Tr( L r L . Ts PI_s E(r) PI_s Ts )/P(s) which can be simplified to
	# P(L|s) = 1/P(s) * sum_(u: Paulis) sum_(i: P_i is in the [u] logical class) sum_(j: Pj is in the [L u L] logical class) CHI[i,j] * (-1)^(P_j)
	# inputs: nqecc, kqecc, chi, algebra (conjugations)
	cdef:
		int i, j, u, l, s
		long double prob, maxprob, contrib
	for s in range(qecc[0].nstabs):
		if (sim[0].syndprobs[s] > consts[0].atol):
			sim[0].corrections[s] = 0
			maxprob = 0
			for l in range(currentframe):
				prob = 0
				if (isPauli == 0):
					for u in range(qecc[0].nlogs):
						contrib = 0.0
						for i in range(qecc[0].nstabs):
							for j in range(qecc[0].nstabs):
								contrib = contrib + sim[0].process[u][qecc[0].algebra[0][l][u]][i][j] * qecc[0].projector[s][j]
						prob = prob + qecc[0].algebra[1][l][u] * contrib
				else:
					for u in range(qecc[0].nlogs):
						contrib = 0
						for i in range(qecc[0].nstabs):
							contrib = contrib + sim[0].process[u][u][i][i] * qecc[0].projector[s][i]
						prob = prob + qecc[0].algebra[1][l][u] * contrib
				if (prob > maxprob):
					sim[0].corrections[s] = l
					maxprob = prob
	return 0


cdef int ComputeSyndromeDistribution(qecc_t *qecc, simul_t *sim, int isPauli, long double atol) nogil:
	# Compute the probability of all the syndromes in the qecc code, for the given error channel and input state. Sample a syndrome from the resulting probability distribution of the syndromes.
	# Probability of a syndrome s, denoted by P(s) is given by the following expression.
	## P(s) = 1/2^(n-k) * sum_(i,j: P_i and P_j are stabilizers) CHI[i,j] * (-1)^sign(P_j)
	cdef int s
	for s in range(qecc[0].nstabs):
		sim[0].syndprobs[s] = 0.0
		if (isPauli == 0):
			for i in range(qecc[0].nstabs):
				for j in range(qecc[0].nstabs):
					sim[0].syndprobs[s] = sim[0].syndprobs[s] + sim[0].process[0][0][i][j] * qecc[0].projector[s][j]
		else:
			for i in range(qecc[0].nstabs):
				sim[0].syndprobs[s] = sim[0].syndprobs[s] + sim[0].process[0][0][i][i] * qecc[0].projector[s][i]
		if (sim[0].syndprobs[s] < atol):
			sim[0].syndprobs[s] = 0.0
		sim[0].syndprobs[s] = sim[0].syndprobs[s]/(<long double>qecc[0].nstabs)
		if (s == 0):
			sim[0].cumulative[s] = sim[0].syndprobs[s]
		else:
			sim[0].cumulative[s] = sim[0].cumulative[s - 1] + sim[0].syndprobs[s]
	# with gil:
	# 	PrintDoubleArray1D(sim[0].syndprobs, "Fresh syndrome probabilities", qecc[0].nstabs)
	return 0


cdef int GetFullProcessMatrix(qecc_t *qcode, simul_t *sim, int isPauli, int level) nogil:
	# For each pair of logical operators, we only need the entries of the Chi matrix that correspond to Pauli operators from different logical classes.
	# We construct the sections of the Chi matrices that correspond to rows and columns in the logical classes
	# cdef np.ndarray[np.longdouble_t, ndim = 4, mode = 'c'] chi = np.empty((4**kqecc, 4**kqecc, 2**(nqecc - kqecc), 2**(nqecc - kqecc)), dtype = np.longdouble)
	cdef:
		unsigned int i, j, k, q
		long double contribution = 1.0
		complex128_t phase
	
	if (isPauli == 0):
		for i in range(qcode[0].nlogs):
			for j in range(qcode[0].nlogs):
				for k in range(qcode[0].nstabs):
					for l in range(qcode[0].nstabs):
						contribution = 1.0
						for q in range(qcode[0].N):
							contribution = contribution * sim[0].virtual[q][qcode[0].action[i][k][q]][qcode[0].action[j][l][q]]
						sim[0].process[i][j][k][l] = creall(qcode[0].phases[i][k] * qcode[0].phases[j][l]) * contribution
	else:
		# For a Pauli channel, the process matrix is diagonal
		for i in range(qcode[0].nlogs):
			for j in range(qcode[0].nstabs):
				contribution = 1.0
				for q in range(qcode[0].N):
					contribution = contribution * sim[0].virtual[q][qcode[0].action[i][j][q]][qcode[0].action[i][j][q]]
				sim[0].process[i][i][j][j] = <long double>(qcode[0].phases[i][j] * qcode[0].phases[i][j]) * contribution
	return 0
