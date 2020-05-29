import os

try:
    import numpy as np
    from scipy import linalg as linalg
    from scipy.stats import poisson
    from scipy.special import comb
except:
    pass
from define import globalvars as gv
from define import chanreps as crep
from define.QECCLfid import utils as ut


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


def IIDWtihCrossTalk(infid, qcode, iid_fraction):
    # Generate a Pauli correlated channel as a weighted sum of IID and two-qubit error distributions.
    atol = 10e-14
    q1 = iid_fraction
    q2 = 1 - q1
    # print("IID fraction: {}, CORR fraction: {}".format(q1, q2))
    n = qcode.N
    # Construct the IID channel as a n-qubit Depolarizing channel.
    # print("Infidelity = {}".format(infid))
    single_qubit_errors = np.array(
        [1 - infid, infid / 3, infid / 3, infid / 3], dtype=np.double
    )
    # print("Single qubit errors: {}".format(single_qubit_errors))
    iid_error_dist = ut.GetErrorProbabilities(
        qcode.PauliOperatorsLST, single_qubit_errors, 0
    )
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
    n_two_qubit_errors = np.int(0.1 * qcode.group_by_weight[2].size)
    two_qubit_errors = np.random.choice(
        qcode.group_by_weight[2], size=n_two_qubit_errors
    )
    # print("Two qubit errors: {}".format(two_qubit_errors))
    # The probability distribution within this subset is Gaussian with mean = 0.1 * 4^n * full_process_infid
    corr_error_dist = np.zeros(iid_error_dist.size, dtype=np.double)
    corr_error_dist[two_qubit_errors] = np.random.normal(
        0.1 * 4 ** n * full_process_infid,
        0.1 * 4 ** n * full_process_infid,
        size=(n_two_qubit_errors,),
    )
    corr_error_dist[two_qubit_errors] = np.where(
        corr_error_dist[two_qubit_errors] >= atol, corr_error_dist[two_qubit_errors], 0
    )
    # print(
    #     "corr_error_dist[two_qubit_errors] = {}".format(
    #         corr_error_dist[two_qubit_errors]
    #     )
    # )
    corr_error_dist[two_qubit_errors] = corr_error_dist[two_qubit_errors] * (
        corr_error_dist[two_qubit_errors] >= atol
    )
    # The infidelity of the purely correlated channel is adjusted to be similar to the infidelity of the IID channel.
    corr_error_dist[0] = 1 - full_process_infid
    corr_error_dist[two_qubit_errors] = (
        full_process_infid
        * corr_error_dist[two_qubit_errors]
        / np.sum(corr_error_dist[two_qubit_errors])
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


def IsotropicRandomPauli(infid, qcode):
    # Generate a random Pauli channel with a specified fidelity to the identity channel.
    # We will generate uniformly random numbers to denote the probability of a non-identity Pauli error.
    # Furthermore, we will ensure that the probability of the non-identity Pauli error add up to a given infidelity value.
    # A Pauli channel is defined by: E(R) = p_I R + p_X X R X + p_Y Y R Y + p_Z Z R Z.
    # For 1 qubit channels, we will return the Kraus operators (Pauli matrices).
    # For multi-qubit channels we will simply return the probability distribution on the Pauli errors.
    single_qubit_errors = np.concatenate(
        ([1 - infid], infid * np.random.uniform(size=3))
    )
    # # Set X and Z to be roughly similar
    # single_qubit_errors[[1, 3]] = (single_qubit_errors[1] + single_qubit_errors[3]) / 2
    # # Set Y to be 10 times lower than X and Z
    # single_qubit_errors[2] = single_qubit_errors[1] / 10
    # # Normalize
    single_qubit_errors = single_qubit_errors / np.sum(single_qubit_errors)

    iid_error_dist = ut.GetErrorProbabilities(
        qcode.PauliOperatorsLST, single_qubit_errors, 0
    )
    # print("iid_error_dist = {}".format(iid_error_dist))

    max_bias_weight = 2
    mean_probs_by_weight = np.zeros(1 + max_bias_weight, dtype=np.double)
    boost = np.abs(np.random.normal(0.25, 0.1))
    for w in range(1 + max_bias_weight):
        mean_probs_by_weight[w] = np.mean(iid_error_dist[qcode.group_by_weight[w]])
    bias = boost * mean_probs_by_weight[1] / mean_probs_by_weight[2]

    multi_qubit_errors = np.array(
        list(
            map(
                lambda erridx: IsAnisotropicOperator(qcode.PauliOperatorsLST[erridx]),
                qcode.group_by_weight[2],
            )
        ),
        dtype=np.int,
    )
    # print("Errors whose probabilities are boosted: {}".format(np.nonzero(multi_qubit_errors)))
    iid_error_dist[np.nonzero(multi_qubit_errors)] *= bias

    # alpha = 0.90
    # global_fraction = 0.2
    # total_selected_errors = int(multi_qubit_errors.size * global_fraction)
    #
    # multi_qubit_correctable = np.intersect1d(
    #     multi_qubit_errors, qcode.PauliCorrectableIndices
    # )
    # multi_qubit_uncorrectable = np.setdiff1d(
    #     multi_qubit_errors, multi_qubit_correctable
    # )
    #
    # errors_to_boost = np.concatenate(
    #     (
    #         np.random.choice(
    #             multi_qubit_correctable, int(alpha * total_selected_errors)
    #         ),
    #         np.random.choice(
    #             multi_qubit_uncorrectable, int((1 - alpha) * total_selected_errors)
    #         ),
    #     )
    # )
    # iid_error_dist[errors_to_boost] *= bias

    iid_error_dist = iid_error_dist / np.sum(iid_error_dist)
    return iid_error_dist


def IsAnisotropicOperator(pauli_op):
    """
	Determine if a multi-qubit Pauli operator is homogeoneous with single qubit terms or not.
	"""
    isanositropic = np.unique(pauli_op[np.nonzero(pauli_op)]).size > 1
    # print("Error: {}, isanositropic = {}".format(pauli_op, isanositropic))
    return isanositropic


def PoissonRandomPauli(infid, average_correctable_weight, weights):
    """
	Assign probabilities to Pauli errors that follow a Poission distribution.
	The mean of the Poisson distribution is the average weight of correctable errors.
	1. Construct the Poisson PMF of a given mean, pmf, for values from 0 to W, the maximum weight of an error.
	2. For each of the numbers, from w = 0 to w = W, do:
	3.      Assign uniformly random probabilities to errors of weight w. Ensure that these probabilities add up to pmf[w].
	4. End for.
	"""
    error_dist = np.zeros(weights.shape[0], dtype=np.double)
    max_weight = int(np.max(weights))
    # Limit the number of errors of a given weight.
    max_errors_of_weight = np.array(
        comb(max_weight, np.arange(max_weight + 1)), dtype=np.int
    ) * np.power(np.arange(max_weight + 1), 2)
    # print("Maximum errors by weight: {}".format(max_errors_of_weight))
    # Generate a Poisson distribution for probability of an error having a weight w.
    # We want to set the most prevalent weight of an error.
    # However, we will not directly fix its value but let it be a normally distributed value with controllable mean and standard deviation equal to the minimum of 0.5 and mean/3.
    weight_dist = poisson.pmf(
        np.arange(1 + max_weight, dtype=np.int), average_correctable_weight
    )
    weight_dist = weight_dist / np.sum(weight_dist)
    # print("Weight distribution: {}\nsum = {}".format(weight_dist, np.sum(weight_dist)))
    # Group errors by weight and within weight-w errors, assign uniformly random probabilities.
    group_by_weight = {w: None for w in range(weight_dist.size)}
    for w in range(weight_dist.size):
        (group_by_weight[w],) = np.nonzero(weights == w)
        if group_by_weight[w].size > 0:
            mask = np.zeros(group_by_weight[w].size, dtype=np.int)
            mask[: max_errors_of_weight[w]] = 1
            np.random.shuffle(mask)
            error_dist[group_by_weight[w]] = (
                np.random.uniform(0, 1, size=group_by_weight[w].size) * mask
            )
            error_dist[group_by_weight[w]] = (
                weight_dist[w]
                * error_dist[group_by_weight[w]]
                / np.sum(error_dist[group_by_weight[w]])
            )
    # print("error_dist: {}\nsum = {}".format(error_dist, np.sum(error_dist)))
    error_dist[0] = 1 - infid
    error_dist[1:] = infid * error_dist[1:] / np.sum(error_dist[1:])
    return error_dist


def RandomPauliChannel(kwargs):
    # Generate a random Pauli channel on n qubits using one of the few methods available.
    # print("args = {}".format(kwargs))
    available_methods = ["uniform", "crosstalk", "poisson"]
    method = "uniform"
    if "method" in kwargs:
        method = available_methods[kwargs["method"]]

    # print("Method = {}".format(method))
    if method == "uniform":
        return IsotropicRandomPauli(kwargs["infid"], kwargs["qcode"])
    elif method == "poisson":
        return PoissonRandomPauli(kwargs["infid"], 1.66, kwargs["qcode"].weightdist)
    elif method == "crosstalk":
        return IIDWtihCrossTalk(
            kwargs["infid"], kwargs["qcode"], kwargs["iid_fraction"]
        )
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
