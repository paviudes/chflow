import os
import numpy as np
from scipy import linalg as linalg
from scipy.stats import poisson
from scipy.special import comb
from define import globalvars as gv
from define import chanreps as crep
from define.QECCLfid import utils as ut
from define import qcode as qc

# from define.qcode import PrepareSyndromeLookUp


def HermitianConjugate(mat):
	# Return the Hermitian conjugate of a matrix
	return np.conjugate(np.transpose(mat))


def RandomHermitian(dim, method="qr"):
	# Generate a random hermitian matrix of given dimensions.
	randMat = np.random.standard_normal(
		size=(dim, dim)
	) + 1j * np.random.standard_normal(size=(dim, dim))
	if method == "qr":
		randH = (
			np.identity(dim)
			+ prox * randMat
			+ HermitianConjugate(np.identity(dim) + prox * randMat)
		) / np.longdouble(2)
	elif method == "exp":
		randH = (randMat + HermitianConjugate(randMat)) / np.longdouble(2)
	elif method == "haar":
		randH = (randMat + HermitianConjugate(randMat)) / np.longdouble(2)
	else:
		print(
			'Method subscribed for random Hermitian production is unknown: "%s".'
			% (method)
		)
		randH = np.identity(dim)
	return randH


def RandomUnitary(prox, dim, method="qr", randH=None):
	# Generate a random unitary matrix of given dimensions and of a certain proximity to identity.
	randMat = np.random.standard_normal(
		size=(dim, dim)
	) + 1j * np.random.standard_normal(size=(dim, dim))
	if method == "qr":
		if randH is None:
			randH = (
				np.identity(dim)
				+ prox * randMat
				+ HermitianConjugate(np.identity(dim) + prox * randMat)
			) / np.longdouble(2)
		(randU, __) = linalg.qr(randH)
	elif method == "exp":
		if randH is None:
			randH = (randMat + HermitianConjugate(randMat)) / np.longdouble(2)
		randU = linalg.expm(1j * prox * randH)
	elif method == "haar":
		if randH is None:
			randH = (randMat + HermitianConjugate(randMat)) / np.longdouble(2)
		(randU, __) = linalg.qr(randH)
	elif method == "hyps":
		# The random unitary is expressed as an exponential of a random Hermitian matrix.
		# The Hermitian matrix can be decomposed in the Pauli basis with real coefficients.
		# Let these coefficients be {c1, ..., cN} where N is the total number of Pauli matrices of the given dimension.
		# In order to ensure that the Unitary matrix has a certain distance, say some function f(p), from the Identity, we will ensure that \sum_i |c_i|^2 = p.
		# So, this reduces to sampling over points in a hypersphere of radius p.
		# The desired Hermitian matrix is simply: \sum_i (xi * Pi) where Pi is the i-th Pauli matrix in the basis.
		if randH is None:
			# paulibasis = np.load("codedata/paulibasis_3qubits.npy")
			npoints = np.power(dim, 2, dtype=np.int)
			#### This part of the code is to ensure that there are only a few (determined by the number of distinct classes) degress of freedom in the random hermitian matrix.
			## Here we force the Hermitian matrix to have non-zero equal contributions from the X,Y and Z Pauli matrices.
			## If we assign the same index to two Pauli matrices in the linear combination, we essentially cut a degree of freedom.
			## Alternatively we can also set one of the indices to -1, in which case that component is set to zero.
			hypersphere = HyperSphereSampling(npoints, center=0.0, radius=prox)
			# print("hypersphere\n%s\nsum = %g. (desired values = %g)" % (np.array_str(hypersphere, max_line_width = 150, precision = 3), np.sum(np.power(hypersphere, 2.0, dtype = np.longdouble), dtype = np.longdouble), prox))
			# randH = np.zeros((dim, dim), dtype = np.complex128)
			# for i in range(nelems):
			# 	randH = randH + hypersphere[i] * paulibasis[i, :, :]
			# print("hypersphere = {}".format(hypersphere))
			randH = np.einsum(
				"i,ikl->kl",
				hypersphere.astype(np.complex128),
				gv.paulibasis,
				dtype=np.complex128,
			)
			# print("Random Hermitian\n%s" % (np.array_str(randH, max_line_width = 150, precision = 3)))
		randU = linalg.expm(1j * randH)
		# print("Random Unitary\n%s" % (np.array_str(randU, max_line_width = 150, precision = 3)))
	else:
		print(
			'Method subscribed for random unitary production is unknown: "%s".'
			% (method)
		)
	return randU


def CreateIIDPauli(infid, qcode):
	# Create an IID Pauli distribution for a given infidelity.
	single_qubit_errors = np.array([1 - infid, infid / 3, infid / 3, infid / 3], dtype=np.double)
	qubit_errors = np.tile(single_qubit_errors, [qcode.N, 1])
	# print("Single qubit errors: {}".format(single_qubit_errors))
	if qcode.PauliOperatorsLST is None:
		qc.PrepareSyndromeLookUp(qcode)
	# iid_error_dist_old = ut.GetErrorProbabilities(qcode.PauliOperatorsLST, single_qubit_errors, 0)
	iid_error_dist = np.prod(qubit_errors[range(qcode.N), qcode.PauliOperatorsLST[:, range(qcode.N)]], axis=1)
	# print("Matching:\nOld: {}\nNew: {}".format(iid_error_dist_old, iid_error_dist_new))
	return iid_error_dist


def ReconstructPauliChannel(pauli_probs, qcode):
	# Reconstruct the n-qubit Pauli channel if we had only weight-1 errors.
	if qcode.group_by_weight is None:
		PrepareSyndromeLookUp(qcode)
	
	single_qubit_errors = qcode.PauliOperatorsLST[qcode.group_by_weight[1], :]
	single_qubit_probs = pauli_probs[qcode.group_by_weight[1]]
	# print("Reconstructing Pauli channel")
	# print("===========")
	# print("single_qubit_probs\n{}".format(single_qubit_probs))
	# print("===========")
	
	# Extract the marginal distribution of pI, pX, pY and pZ on each qubit.
	qubit_pauli_probs = np.zeros((qcode.N, 4), dtype = np.double)
	# Fill the identity probs
	qubit_pauli_probs[:, 0] = np.power(pauli_probs[0], 1/qcode.N)
	# Retrieve the single qubit error probabilities
	for i in range(qcode.group_by_weight[1].size):
		supp, = np.nonzero(single_qubit_errors[i, :])
		err_type = single_qubit_errors[i, supp]
		qubit_pauli_probs[supp, err_type] = single_qubit_probs[i]

	# If the probability of an error type for a qubit is 0, use (1 - pI)/3 instead.
	for q in range(qcode.N):
		for err_type in range(1, 4):
			if (qubit_pauli_probs[q, err_type] == 0):
				# print("Single qubit error rate of {} for qubit {} is missing." % (err_type, q))
				qubit_pauli_probs[q, err_type] = (1 - qubit_pauli_probs[q, 0])/3

	# print("Qubit Pauli probs before normalization\n{}".format(qubit_pauli_probs))

	# Normalize so that the marginal distributions add up to 1.
	for q in range(qcode.N):
		# qubit_pauli_probs[q, 1:] = (1 - qubit_pauli_probs[q, 0]) * qubit_pauli_probs[q, 1:]/np.sum(qubit_pauli_probs[q, 1:])
		qubit_pauli_probs[q, 0] = 1 - np.sum(qubit_pauli_probs[q, 1:])

	# print("Qubit Pauli probs after normalization\n{}".format(qubit_pauli_probs))
	# print("===========")

	# Create an n-qubit Pauli channel where the probabilities of n-qubit errors are constructed using the i.i.d ansatz from the single qubit error probabilities.
	pauli_dist = np.prod(qubit_pauli_probs[range(qcode.N), qcode.PauliOperatorsLST[:, range(qcode.N)]], axis=1)
	
	return pauli_dist


def IIDWtihCrossTalk(infid, qcode, iid_fraction, subset_fraction):
	# Generate a Pauli correlated channel as a weighted sum of IID and two-qubit error distributions.
	atol = 1E-14
	q1 = iid_fraction
	q2 = 1 - q1
	# print("IID fraction: {}, CORR fraction: {}".format(q1, q2))
	n = qcode.N
	# Construct the IID channel as a n-qubit Depolarizing channel.
	# print("Infidelity = {}".format(infid))
	iid_error_dist = CreateIIDPauli(infid, qcode)
	# print("qcode.PauliOperatorsLST = {}".format(qcode.PauliOperatorsLST))
	# print("iid_error_dist = {}".format(iid_error_dist))
	full_process_infid = 1 - iid_error_dist[0]
	# print(
	#     "Sum of IID error probabilities = {}, Infidelity = {}.".format(
	#         np.sum(iid_error_dist), full_process_infid
	#     )
	# )
	### Constructed the purely corelated channel.
	# Add a random sumset of 10% of all two qubit errors
	weights_to_boost = [2, 3, 4]
	subset_fraction_weights = {2: subset_fraction}
	for w in weights_to_boost:
		if w not in subset_fraction_weights:
			subset_fraction_weights.update(
				{
					w: 1
					/ 3
					* comb(qcode.N, w - 1)
					/ comb(qcode.N, w)
					* subset_fraction_weights[w - 1]
				}
			)
	# print("subset_fraction_weights = {}".format(subset_fraction_weights))
	n_errors = np.cumsum(
		[
			max(1, np.int(subset_fraction_weights[w] * qcode.group_by_weight[w].size))
			for w in weights_to_boost
		]
	)
	# print("n_errors: {}".format(n_errors))
	errors_to_boost = np.zeros(n_errors[-1], dtype=np.int)
	corr_error_dist = np.zeros(iid_error_dist.size, dtype=np.double)
	for i in range(len(weights_to_boost)):
		w = weights_to_boost[i]
		if i == 0:
			errors_to_boost[: n_errors[i]] = np.random.choice(
				qcode.group_by_weight[weights_to_boost[i]], size=n_errors[i]
			)
			mq_errors = n_errors[i]
			selected_errors = np.arange(n_errors[i])
		else:
			errors_to_boost[n_errors[i - 1] : n_errors[i]] = np.random.choice(
				qcode.group_by_weight[weights_to_boost[i]],
				size=n_errors[i] - n_errors[i - 1],
			)
			mq_errors = size = n_errors[i] - n_errors[i - 1]
			selected_errors = np.arange(n_errors[i - 1], n_errors[i])
		# corr_error_dist[errors_to_boost[selected_errors]] = np.random.normal(
		#     (0.1 ** (w - 1)) * 4 ** n * full_process_infid,
		#     (0.1 ** (w - 1)) * 4 ** n * full_process_infid,
		#     size=(mq_errors,),
		# )
		##########
		# Temporary patch for strong correlations
		corr_error_dist[errors_to_boost[selected_errors]] = np.random.normal(
			full_process_infid,
			selected_errors.size,
			size=(mq_errors,),
		)
		##########

	# print("errors_to_boost: {}".format(errors_to_boost))
	# The probability distribution within this subset is Gaussian with mean = 0.1 * 4^n * full_process_infid
	# corr_error_dist = np.zeros(iid_error_dist.size, dtype=np.double)
	# corr_error_dist[errors_to_boost] = np.random.normal(0.1 * 4 ** n * full_process_infid,0.1 * 4 ** n * full_process_infid,size=(n_errors[-1],),)

	# Setting negative numbers to 0.
	corr_error_dist[errors_to_boost] = np.where(
		corr_error_dist[errors_to_boost] >= atol, corr_error_dist[errors_to_boost], 0
	)
	# print(
	#     "corr_error_dist[errors_to_boost] = {}".format(corr_error_dist[errors_to_boost])
	# )
	corr_error_dist[errors_to_boost] = corr_error_dist[errors_to_boost] * (
		corr_error_dist[errors_to_boost] >= atol
	)
	# The infidelity of the purely correlated channel is adjusted to be similar to the infidelity of the IID channel.
	corr_error_dist[0] = 1 - full_process_infid
	corr_error_dist[errors_to_boost] = (
		full_process_infid
		* corr_error_dist[errors_to_boost]
		/ np.sum(corr_error_dist[errors_to_boost])
	)
	# Explicitly normalize the purely correlated distribution -- this is needed because there are some numerical approximation errors for high noise regime.
	corr_error_dist = corr_error_dist / np.sum(corr_error_dist)
	# print(
	#     "Sum of CORR error probabilities = {}, Infidelity = {}".format(
	#         np.sum(corr_error_dist), 1 - corr_error_dist[0]
	#     )
	# )
	#### Take a linear combination of IID and purely correlated distributions.
	pauli_error_dist = q1 * iid_error_dist + q2 * corr_error_dist
	# print("Pauli error distribution:\n{}".format(pauli_error_dist))
	return pauli_error_dist


def AnIsotropicRandomPauli(infid, max_weight, qcode):
	# Generate a random Pauli channel with a specified fidelity to the identity channel.
	# We will generate uniformly random numbers to denote the probability of a non-identity Pauli error.
	# Furthermore, we will ensure that the probability of the non-identity Pauli error add up to a given infidelity value.
	# A Pauli channel is defined by: E(R) = p_I R + p_X X R X + p_Y Y R Y + p_Z Z R Z.
	# For 1 qubit channels, we will return the Kraus operators (Pauli matrices).
	# For multi-qubit channels we will simply return the probability distribution on the Pauli errors.
	# print("Infidelity = {}".format(infid))
	single_qubit_errors = np.concatenate(([1 - infid], np.random.uniform(size=3)))
	# # Set X and Z to be roughly similar
	# single_qubit_errors[[1, 3]] = (single_qubit_errors[1] + single_qubit_errors[3]) / 2
	# # Set Y to be 10 times lower than X and Z
	# single_qubit_errors[2] = single_qubit_errors[1] / 10
	# # Normalize
	single_qubit_errors[1:] = infid * single_qubit_errors[1:] / np.sum(single_qubit_errors[1:])
	# print("Single qubit error rates: {}".format(single_qubit_errors))
	iid_error_dist = ut.GetErrorProbabilities(qcode.PauliOperatorsLST, single_qubit_errors, 0)
	corr_error_dist = np.zeros_like(iid_error_dist)
	corr_error_dist = iid_error_dist[:]
	# print("iid_error_dist = {}".format(np.sort(iid_error_dist)))

	# sum_probs_by_weight = np.zeros(1 + max_weight, dtype=np.double)
	# sum_probs_by_weight[0] = 1 - iid_error_dist[0]
	mean_probs_by_weight = np.zeros(1 + max_weight, dtype=np.double)
	mean_probs_by_weight[0] = 1 - iid_error_dist[0]
	# min_probs_by_weight = np.zeros(1 + max_weight, dtype=np.double)
	# min_probs_by_weight[0] = 1 - iid_error_dist[0]
	# max_probs_by_weight = np.zeros(1 + max_weight, dtype=np.double)
	# max_probs_by_weight[0] = 1 - iid_error_dist[0]
	# sum_probs_by_weight[1] = np.sum(iid_error_dist[qcode.group_by_weight[1]])
	boost = 0.3
	for w in range(1, 1 + max_weight):
		if w > 2:
			boost = np.power(10, w/2)
		# sum_probs_by_weight[w] = np.sum(iid_error_dist[qcode.group_by_weight[w]])
		# min_probs_by_weight[w] = np.min(iid_error_dist[qcode.group_by_weight[w]])
		mean_probs_by_weight[w] = np.mean(iid_error_dist[qcode.group_by_weight[w]])
		# bias = boost * sum_probs_by_weight[w - 1] / sum_probs_by_weight[w]
		# bias = boost * min_probs_by_weight[w - 1] / max_probs_by_weight[w]
		bias = boost * mean_probs_by_weight[w - 1] / mean_probs_by_weight[w]
		# Boost the probability of multi-qubit errors.
		anisotropic_errors = np.array(list(map(IsAnisotropicOperator, qcode.PauliOperatorsLST[qcode.group_by_weight[w]])), dtype = np.int)
		selected_errors, = np.nonzero(anisotropic_errors)
		# print("Errors whose probabilities are boosted: {}".format(np.nonzero(anisotropic_errors)))
		corr_error_dist[qcode.group_by_weight[w][selected_errors]] *= bias

	# Normalize to ensure that the probability of non-identity errors add up to the n-qubit infid.
	corr_error_dist[1:] = (1 - corr_error_dist[0]) * corr_error_dist[1:] / np.sum(corr_error_dist[1:])
	return iid_error_dist


def IsAnisotropicOperator(pauli_op):
	"""
	Determine if a multi-qubit Pauli operator is homogeoneous with single qubit terms or not.
	"""
	isanositropic = np.unique(pauli_op[np.nonzero(pauli_op)]).size > 1
	# print("Error: {}, isanositropic = {}".format(pauli_op, isanositropic))
	return isanositropic


def PoissonRandomPauli(infid, mean_correlation_length, subset_fraction, qcode):
	"""
	Assign probabilities to Pauli errors that follow a Poission distribution.
	The mean of the Poisson distribution is the average weight of correctable errors.
	1. Construct the Poisson PMF of a given mean, pmf, for values from 0 to W, the maximum weight of an error.
	2. For each of the numbers, from w = 0 to w = W, do:
	3.      Assign uniformly random probabilities to errors of weight w. Ensure that these probabilities add up to pmf[w].
	4. End for.
	"""
	error_dist = CreateIIDPauli(infid, qcode)
	# Limit the number of errors of a given weight.
	# n_selected = np.array(comb(qcode.N, np.arange(qcode.N + 1)), dtype=np.int) * np.power(np.arange(qcode.N + 1), 2)
	
	# Generate a Poisson distribution for probability of an error having a weight w.
	weight_dist = poisson.pmf(np.arange(1 + qcode.N, dtype=np.int), mean_correlation_length)
	# Set the probability of the identity error to be 1 - infid.
	weight_dist[0] = 1 - infid
	# Force the total probabilities of errors of weights w > 0 to be equal to infid.
	weight_dist[1:] = infid * weight_dist[1:] / np.sum(weight_dist[1:])
	# print("Weight distribution: {}\nsum = {}".format(weight_dist, np.sum(weight_dist)))
	
	for w in range(1 + qcode.N):
		# Limit the number of errors of a given weight.
		mask = np.zeros(qcode.group_by_weight[w].size, dtype=np.int)
		if (w < 2):
			boost = 1
			n_selected = qcode.group_by_weight[w].size
		else:
			boost = 1/np.sqrt(infid)
			n_selected = max(1, int(subset_fraction * qcode.group_by_weight[w].size))
		mask[: n_selected] = 1
		# Choose a random subset of N errors of weights w.
		np.random.shuffle(mask)
		# Boost probabilities for the chosen errors.
		errors_to_boost, = np.nonzero(mask)
		# print("mask = {}\nqcode.group_by_weight[{}]\n{}".format(errors_to_boost, w, qcode.group_by_weight[w]))
		error_dist[qcode.group_by_weight[w][errors_to_boost]] *= boost
		# Renormalize the probabilities of all errors of weight w
		error_dist[qcode.group_by_weight[w]] = weight_dist[w] * error_dist[qcode.group_by_weight[w]]/np.sum(error_dist[qcode.group_by_weight[w]])

	# print("Error distribution conditions:\nerror_dist[0] = {}, np.max(error_dist) = {}, np.min(error_dist) = {}, np.sum(error_dist) = {}.".format(error_dist[0], np.max(error_dist), np.min(error_dist), np.sum(error_dist)))
	return error_dist


def RandomPauliChannel(kwargs):
	# Generate a random Pauli channel on n qubits using one of the few methods available.
	# print("args = {}".format(kwargs))
	available_methods = ["uniform", "crosstalk", "poisson"]
	method = "uniform"
	if "method" in kwargs:
		method = available_methods[kwargs["method"]]

	if kwargs["qcode"].PauliOperatorsLST is None:
		qc.PrepareSyndromeLookUp(kwargs["qcode"])

	# print("Method = {}".format(method))
	if method == "uniform":
		return AnIsotropicRandomPauli(kwargs["infid"], int(kwargs["iid_fraction"]), kwargs["qcode"])
	elif method == "crosstalk":
		return IIDWtihCrossTalk(kwargs["infid"], kwargs["qcode"], kwargs["iid_fraction"], kwargs["subset_fraction"])
	elif method == "poisson":
		return PoissonRandomPauli(kwargs["infid"], kwargs["iid_fraction"], kwargs["subset_fraction"], kwargs["qcode"])
	else:
		pass
	return None


def UncorrelatedRandomPauli(infid):
	"""
	Kraus operators for an single qubit Pauli channel.
	"""
	probs = np.random.rand(4)
	probs[0] = 1 - infid
	probs[1:] = infid * probs[1:] / np.sum(probs[1:])
	krauss = np.zeros((4, 2, 2), dtype=np.complex128)
	for i in range(krauss.shape[0]):
		krauss[i, :, :] = np.sqrt(probs[i]) * gv.Pauli[i, :, :]
	return krauss


def RandomCPTP(dist, meth):
	# Generate a random CPTP map by the specified method.
	# Available methods are:
	# 1. Exponential of random Hermitian
	# 2. Diagonalization of a random Hermitian
	# 3. Haar random unitary
	# 4. Hypersphere sampling for generating a random Hermitian and then exponetial to determine unitary.
	# 5. Generating random Pauli amplitudes for X, Y and Z errors, given a probability for no error.
	# print("RandomCPTP({}, {})".format(dist, meth))
	availmethods = ["exp", "qr", "haar", "hyps", "pauli"]
	method = availmethods[meth]
	if method == "pauli":
		krauss = RandomPauliChannel(dist)
	else:
		randU = RandomUnitary(dist, 8, method, None)
		krauss = crep.ConvertRepresentations(randU, "stine", "krauss")
	return krauss


def HyperSphereSampling(npoints, center=0.0, radius=1.0, classification=None):
	# Sample points on a hypersphere of given radius and center.
	# We use the algorithm outlined in https://dl.acm.org/citation.cfm?id=377946.
	## Sketch of the algorithm:
	# 1. Generate n points {x1, ..., xn} distributed according to the Normal distribution with mean = center of the sphere (in our case: (0, 0, 0...)).
	# 2. For each xi, do xi -> xi/(z/p) where z = sqrt(\sum_i (xi)^2).
	## There is an additional option to reduce the degrees of freedom on the surface by ensuring that the points are concentrated into various classes.
	## See also: https://math.stackexchange.com/questions/132933/generating-3-times-3-unitary-matrices-close-to-the-identity
	if classification is None:
		classification = np.arange(npoints, dtype=np.int)
	normalization = 0
	normalvariates = np.random.normal(
		loc=center,
		scale=1.0,
		size=np.unique(classification[np.where(classification > -1)]).shape[0],
	)
	surface = np.zeros(npoints, dtype=np.double)
	for i in range(npoints):
		if classification[i] == -1:
			surface[i] = 0.0
		else:
			surface[i] = normalvariates[classification[i]]
		normalization = normalization + np.power(surface[i], 2.0)
	for i in range(npoints):
		surface[i] = surface[i] * radius / np.sqrt(normalization)
	return surface


def RandomPauliTransfer(pauliprobs, maxterms=-1):
	r"""
	Generate the Pauli transfer matrix of a random Pauli channel.
	Note that the Pauli transfer matrix need not be of the specified input Pauli channel, it can be another random Pauli channel.

	A diagonal element of the Pauli transfer matrix, corresponding to the Pauli operator :math:`P`, written as :math:`\Gamma_{P,P}`, can be expressed as follows.

	..math::
		\begin{gather}
		\Gamma_{P,P} = \sum_{A\in S_{C}}p_{A} - \sum_{B\in S_{A}}p_{B}
		\end{gather}

	where :math:`S_{C}` denotes the set of all operators that commute with :math:`P`, while :math:`S_{A}` is the set of all operators that anticommute with :math:`P`.

	Note that for any Pauli operator :math:`P`, the set :math:`S_{C}` contains half of all the Pauli operators, other than :math:`I` and :math:`P`. The set :math:`S_{A}` contains the other half. Hence, to generate random elements of the Pauli transfer matrix, we simply need to divide the set of Pauli operators into two halfs and compute the above expression.
	"""
	if maxterms == -1:
		maxterms = pauliprobs.shape[0]
	ptm = np.zeros(maxterms, dtype=np.double)
	ptm[0] = 1
	partition = np.zeros(pauliprobs.shape[0] - 2, dtype=np.int)
	partition[: (partition.shape[0] // 2)] = 1
	for i in range(maxterms - 1):
		np.random.shuffle(partition)
		relations = np.concatenate(([1], partition[:i], [1], partition[i:]))
		commute = np.dot(relations, pauliprobs)
		anticommute = np.dot((1 - relations), pauliprobs)
		ptm[i + 1] = commute - anticommute
	return ptm
