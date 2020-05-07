import os

try:
    import numpy as np
    from scipy import linalg as linalg
    from scipy.stats import poisson
except:
    pass
from define import globalvars as gv
from define import chanreps as crep


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


def IsotropicRandomPauli(infid, nqubits=1):
    # Generate a random Pauli channel with a specified fidelity to the identity channel.
    # We will generate uniformly random numbers to denote the probability of a non-identity Pauli error.
    # Furthermore, we will ensure that the probability of the non-identity Pauli error add up to a given infidelity value.
    # A Pauli channel is defined by: E(R) = p_I R + p_X X R X + p_Y Y R Y + p_Z Z R Z.
    # For 1 qubit channels, we will return the Kraus operators (Pauli matrices).
    # For multi-qubit channels we will simply return the probability distribution on the Pauli errors.
    print(
        "Random Pauli channel on {} qubits with infidelity {}.".format(nqubits, infid)
    )
    pauliprobs = np.random.uniform(size=(4 ** nqubits - 1,))
    pauliprobs = infid * pauliprobs / np.sum(pauliprobs)
    # pauliamps = HyperSphereSampling(4 ** nqubits - 1, center=0.0, radius=np.sqrt(infid))
    # print("p_I = %g, p_X = %g, p_Y = %g and p_Z = %g." % (1 - np.sum(np.power(pauliamps, 2.0)), np.power(pauliamps[0], 2.0), np.power(pauliamps[1], 2.0), np.power(pauliamps[2], 2.0)))
    # print("dist shape: {}".format(pauliprobs.shape))
    if nqubits > 1:
        return np.concatenate(([1 - infid], pauliprobs))
    krops = np.zeros((4 ** nqubits, 2 ** nqubits, 2 ** nqubits), dtype=np.complex128)
    krops[0, :, :] = np.sqrt(1 - infid) * gv.Pauli[0, :, :]
    for i in range(1, 4):
        krops[i, :, :] = np.sqrt(pauliprobs[i]) * gv.Pauli[i, :, :]
    return krops


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
            error_dist[group_by_weight[w]] = np.random.uniform(
                0, 1, size=group_by_weight[w].size
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
    available_methods = ["uniform", "poisson"]
    method = "uniform"
    if "method" in kwargs:
        method = available_methods[kwargs["method"]]

    # print("Method = {}".format(method))

    if method == "uniform":
        return IsotropicRandomPauli(kwargs["infid"], kwargs["n"])
    elif method == "poisson":
        return PoissonRandomPauli(kwargs["infid"], 1.5, kwargs["weightdist"])
    else:
        pass
    return None


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
