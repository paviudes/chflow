Channels = {
    "idn": {
        "name": "Identity channel",
        "params": [],
        "latex": [],
        "color": "white",
        "Pauli": 1,
        "corr": 0,
    },
    "ad": {
        "name": "Amplitude damping",
        "params": ["Damping rate"],
        "latex": ["$\\lambda$"],
        "color": "forestgreen",
        "Pauli": 1,
        "corr": 0,
    },
    "bf": {
        "name": "Bit flip channel",
        "params": ["X error rate"],
        "latex": ["p_{X}"],
        "color": "sienna",
        "Pauli": 1,
        "corr": 0,
    },
    "pd": {
        "name": "Dephasing channel",
        "params": ["Dephasing rate"],
        "latex": ["$p$"],
        "color": "deeppink",
        "Pauli": 1,
        "corr": 0,
    },
    "bpf": {
        "name": "Bit phase flip",
        "params": ["Y error rate"],
        "latex": ["$p_{Y}$"],
        "color": "darkkhaki",
        "Pauli": 1,
        "corr": 0,
    },
    "dp": {
        "name": "Depolarizing channel",
        "params": ["Depolarizing rate"],
        "latex": ["$p$"],
        "color": "red",
        "Pauli": 1,
        "corr": 0,
    },
    "pauli": {
        "name": "Pauli channel",
        "params": ["Depolarizing rate"],
        "latex": ["$p$"],
        "color": "red",
        "Pauli": 1,
        "corr": 0,
    },
    "rtx": {
        "name": "Rotation about the X-axis",
        "params": ["Rotation angle"],
        "latex": ["\\delta/2\\pi"],
        "color": "darkviolet",
        "Pauli": 1,
        "corr": 0,
    },
    "rty": {
        "name": "Rotation about Y-axis",
        "params": ["Rotation angle"],
        "latex": ["\\delta/2\\pi"],
        "color": "darkorchid",
        "Pauli": 1,
        "corr": 0,
    },
    "rtz": {
        "name": "Rotation about the Z-axis",
        "params": ["Rotation angle"],
        "latex": ["\\delta/2\\pi"],
        "color": "black",
        "Pauli": 1,
        "corr": 0,
    },
    "rtnp": {
        "name": "Rotation about an arbitrary axis of the Bloch sphere",
        "params": [
            "phi (azimuthal angle)",
            "theta (angle with Z-axis)",
            "rotation angle as a fraction of pi (the rotation axis is decribed by (phi, theta)).",
        ],
        "latex": ["$\\theta/2\\pi$", "\\phi/2\\pi", "\\delta"],
        "color": "indigo",
        "Pauli": 1,
        "corr": 0,
    },
    "rtas": {
        "name": "Asymmetric rotations about arbitrary axes of the Bloch sphere",
        "params": [
            "Number of qubits",
            "Average angle of rotation",
            "Standard deviation for rotation angle",
            "Standard deviation for angle with Z-axis",
            "Standard deviation for azimuthal angle",
        ],
        "latex": [
            "N",
            "$\\mu_{\\delta}$",
            "\\Delta_{\\delta}",
            "\\Delta_{\\theta}",
            "\\Delta_{\\phi}",
        ],
        "color": "midnightblue",
        "Pauli": 1,
        "corr": 0,
    },
    "rand": {
        "name": "Random CPTP map",
        "params": ["Interaction time of Hamiltonian on system and environment"],
        "latex": ["$t$"],
        "color": "peru",
        "Pauli": 1,
        "corr": 0,
    },
    "randunit": {
        "name": "Random unitary channel",
        "params": ["Interaction time"],
        "latex": ["$t$"],
        "color": "goldenrod",
        "Pauli": 1,
        "corr": 0,
    },
    "pcorr": {
        "name": "Random correlated Pauli channel",
        "params": "Infidelity",
        "latex": ["$1 - p_{I}$"],
        "color": "limegreen",
        "Pauli": 1,
        "corr": 0,
    },
    "pl": {
        "name": "Photon loss channel",
        "params": ["number of photons (alpha)", "decoherence rate (gamma)"],
        "latex": ["$\\alpha$", "$\\gamma$"],
        "color": "steelblue",
        "Pauli": 1,
        "corr": 0,
    },
    "gd": {
        "name": "Generalized damping",
        "params": ["Relaxation", "dephasing"],
        "latex": ["$\\lambda$", "$p$"],
        "color": "green",
        "Pauli": 1,
        "corr": 0,
    },
    "gdt": {
        "name": "Generalized time dependent damping",
        "params": ["Relaxation (t/T1)", "ratio (T2/T1)"],
        "latex": ["$\\frac{t}{T_{1}}$", "$\\frac{T_{2}}{T_{1}}$"],
        "color": "darkgreen",
        "Pauli": 1,
        "corr": 0,
    },
    "gdtx": {
        "name": "Explicit generalized time dependent damping",
        "params": ["T1", "T2", "t"],
        "latex": ["$t$", "$T_{1}$", "$T_{2}$"],
        "color": "olive",
        "Pauli": 1,
        "corr": 0,
    },
}

Metrics = {
    "dnorm": {
        "phys": "Diamond distance of the physical channel",
        "log": "Diamond distance of the logical channel",
        "latex": "$|| \\mathcal{E} - \\mathsf{id} ||_{\\diamondsuit}$",
        "marker": u"+",
        "color": "crimson",
        "desc": "See Sec. 4 of DOI: 10.4086/toc.2009.v005a011.",
        "func": "lambda J, kwargs: DiamondNorm(J, kwargs)",
    },
    "errp": {
        "phys": "Error probability of the physical channel",
        "log": "Error probability of the logical channel",
        "latex": "p_{err}(\\mathcal{E})$",
        "marker": u"*",
        "color": "darkblue",
        "desc": "1 - p* where p* is the maximum value between 0 and 1 such that J - (p*)*J_id is a valid choi matrix, up to normalization.",
        "func": "lambda J, kwargs: ErrorProbability(J, kwargs)",
    },
    "entropy": {
        "phys": "Entropy of the physical channel",
        "log": "Entropy of the logical channel",
        "latex": "$S(\\mathcal{J})$",
        "marker": u"p",
        "color": "brown",
        "desc": "Von-Neumann entropy of the channel's Choi matrix.",
        "func": "lambda J, kwargs: Entropy(J, kwargs)",
    },
    "infid": {
        "phys": "Infidelity of the physical channel",
        "log": "Infidelity of the logical channel",
        "latex": "$1 - F$",
        "marker": u"s",
        "color": "forestgreen",
        "desc": "1 - Fidelity between the input Choi matrix and the Choi matrix corresponding to the identity state.",
        "func": "lambda J, kwargs: Infidelity(J, kwargs)",
    },
    "np1": {
        "phys": "Non-Pauliness of the physical channel",
        "log": "Non-Pauliness of the logical channel",
        "latex": "$\\mathcal{W}(\\mathcal{E})$",
        "marker": u"v",
        "color": "lavender",
        "desc": "L2 norm of the difference between the channel's Chi matrix and it's twirled approximation.",
        "func": "lambda J, kwargs: NonPaulinessChi(J, kwargs)",
    },
    "np2": {
        "phys": "Non-Pauliness of the physical channel",
        "log": "Non-Pauliness of the logical channel",
        "latex": "$np_{2}(\\mathcal{E})$",
        "marker": u"1",
        "color": "maroon",
        "desc": "Least fidelity between the channel's Choi matrix and a bell state.",
        "func": "lambda J, kwargs: NonPaulinessChoi(J, kwargs)",
    },
    "np4": {
        "phys": "Non-Pauliness of the physical channel",
        "log": "Non-Pauliness of the logical channel",
        "latex": "$np_{4}(\\mathcal{E})$",
        "marker": u"3",
        "color": "turquoise",
        "desc": 'Maximum "amount" of Pauli channel that can be subtracted from the input Pauli channel, such that what remains is still a valid quantum channel.',
        "func": "lambda J, kwargs: NonPaulinessRemoval(J, kwargs)",
    },
    "trn": {
        "phys": "Trace-distance of the physical channel",
        "log": "Trace-distance of the logical channel",
        "latex": "$\\left|\\left|\\mathcal{J} - \\mathsf{id}\\right|\\right|_{1}$",
        "marker": u"8",
        "color": "black",
        "desc": "Trace norm of the difference between the channel's Choi matrix and the input Bell state, Trace norm of A is defined as: Trace(Sqrt(A^\\dagger . A)).",
        "func": "lambda J, kwargs: TraceNorm(J, kwargs)",
    },
    "frb": {
        "phys": "Frobenious norm of the physical channel",
        "log": "Frobenious norm of the logical channel",
        "latex": "$\\left|\\left|\\mathcal{J} - \\mathsf{id}\\right|\\right|_{2}$",
        "marker": u"d",
        "color": "chocolate",
        "desc": "Frobenious norm of the difference between the channel's Choi matrix and the input Bell state, Frobenious norm of A is defined as: Sqrt(Trace(A^\\dagger . A)).",
        "func": "lambda J, kwargs: FrobeniousNorm(J, kwargs)",
    },
    "bd": {
        "phys": "Bures distance of the physical channel",
        "log": "Bures distance of the logical channel",
        "latex": "$\\Delta_{B}(\\mathcal{J})$",
        "marker": u"<",
        "color": "goldenrod",
        "desc": "Bures distance between the channel's Choi matrix and the input Bell state. Bures distance between A and B is defined as: sqrt( 2 - 2 * sqrt( F ) ), where F is the Uhlmann-Josza fidelity between A and B.",
        "func": "lambda J, kwargs: BuresDistance(J, kwargs)",
    },
    "uhl": {
        "phys": "Uhlmann Infidelity of the physical channel",
        "log": "Uhlmann Infidelity of the logical channel",
        "latex": "$1 - F_{U}(\\mathcal{J})$",
        "marker": u"h",
        "color": "midnightblue",
        "desc": "1 - Uhlmann-Jozsa fidelity between the channel's Choi matrix and the input Bell state. The Uhlmann-Jozsa fidelity between A and B is given by: ( Trace( sqrt( sqrt(A) B sqrt(A) ) ) )^2.",
        "func": "lambda J, kwargs: UhlmanFidelity(J, kwargs)",
    },
    "unitarity": {
        "phys": "Non-unitarity of the physical channel",
        "log": "Non-unitarity of the logical channel",
        "latex": "$1-\\mathcal{u}(\\mathcal{E})$",
        "marker": u"^",
        "color": "fuchsia",
        "desc": "In the Pauli-Liouville representation of the channel, P, the unitarity is given by: ( sum_(i,j; i not equal to j) |P_ij|^2 ).",
        "func": "lambda J, kwargs: NonUnitarity(J, kwargs)",
    },
    "uncorr": {
        "phys": "Uncorrectable error probability",
        "log": "Uncorrectable error probability of the logical channel",
        "latex": "$p_{u}$",
        "marker": u">",
        "color": "blue",
        "desc": "The total probability of uncorrectable (Pauli) errors.",
        "func": "lambda P, kwargs: UncorrectableProb(P, kwargs)",
    },
    "anisotropy": {
        "phys": "Anisotropy of the physical channel",
        "log": "Anisotropy of the logical channel",
        "latex": "$p_{Y}/p_{X} + p_{Z}/p_{X}$",
        "marker": u"d",
        "color": "goldenrod",
        "desc": "Anisotropy between Y errors and X, Z errors.",
        "func": "lambda J, kwargs: Anisotropy(J, kwargs)",
    },
}


def Identity():
    """
    Kraus operators for the identity channel.
    """
    return np.eye(2, dtype=np.complex128)[np.newaxis, :, :]


def Bitflip(params):
    """
    Kraus operators for the Bit flip channel
    """
    return PauliChannel(1 - params[0], params[0], 0, 0)


def AmplitudeDamping(params):
    """
    Kraus operators for the Amplitude damping channel
    """
    krauss = np.zeros((2, 2, 2), dtype=np.complex128)
    krauss[0, :, :] = np.array([[1, 0], [0, np.sqrt(1 - params[0])]])
    krauss[1, :, :] = np.array([[0, np.sqrt(params[0])], [0, 0]])
    return krauss


def Dephasing(params):
    """
    Kraus operators for the dephasing channel.
    """
    return PauliChannel(1 - params[0], 0, 0, params[0])


def GeneralizedDamping(params):
    """
    Kraus operators for the generalized damping channel
    """
    return gd.GeneralizedDamping(params[0], params[1])


def TimeDependentGeneralizedDamping(params):
    """
    Kraus operators for the time dependent generalized damping channel.
    """
    return gd.GDTimeScales(params[0], params[1])


def TimeDependentGeneralizedDampingExplicit(params):
    """
    Kraus operators for the time dependent generalized damping channel.
    """
    return gd.GDTimeScalesExplicit(params[0], params[1])


def BitPhaseFlip(params):
    """
    Kraus operators for the bit phase flip channel.
    """
    return PauliChannel(1 - params[0], 0, params[0], 0)


def Depolarizing(params):
    """
    Kraus operators for the Depolarizing channel.
    """
    return PauliChannel(1 - params[0], params[0] / 3, params[0] / 3, params[0] / 3)


def PauliChannel(params):
    """
    Kraus operators for the Pauli channel.
    """
    krauss = np.zeros((4, 2, 2), dtype=np.complex128)
    for i in range(4):
        krauss[i, :, :] = np.sqrt(params[i]) * gv.Pauli[i, :, :]
    return kraus


def ArbitraryRotation(params):
    """
    Asymmetric rotation to n qubits.
    """
    names = [
        "N",
        "mean_delta",
        "std_delta",
        "mean_theta",
        "std_theta",
        "mean_phi",
        "std_phi",
    ]
    specs = [1, 0.01, 0.01, -1, -1]
    for i in range(len(params)):
        specs[i] = params[i]

    nqubits = int(specs[0])

    mean_delta = specs[1]
    std_delta = specs[2]
    if std_delta < 0:
        std_delta = mean_delta / (-1 * std_delta)

    mean_theta = np.random.uniform(0, np.pi)
    std_theta = specs[3]
    if std_theta < 0:
        std_theta = mean_theta / (-1 * std_theta)

    mean_phi = np.random.uniform(0, 2 * np.pi)
    std_phi = specs[4]
    if std_phi < 0:
        std_phi = mean_phi / (-1 * std_phi)

    unitaries = np.zeros((nqubits, 2, 2), dtype=np.complex128)
    for i in range(nqubits):
        ####### Random setting
        theta = np.random.normal(mean_theta, std_theta)
        phi = np.random.normal(mean_phi, std_phi)
        delta = np.random.normal(mean_delta, std_delta)
        # Forcing theta to be between 0 and pi
        if theta < 0:
            theta = -1 * theta
        if theta > np.pi:
            theta = theta - int(theta / np.pi) * np.pi
        # print(
        #     "on qubit {}, delta = {}, theta = {}, phi = {}".format(
        #         i, delta, theta, phi
        #     )
        # )
        axis = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            dtype=np.double,
        )
        exponent = np.zeros((2, 2), dtype=np.complex128)
        for j in range(3):
            exponent = exponent + axis[j] * gv.Pauli[j + 1, :, :]
        unitaries[i, :, :] = linalg.expm(-1j * delta / 2 * np.pi * exponent)
    krauss = unitaries
    return


def FixedRotation(params):
    """
    Rotation about some axis of the Bloch sphere.

    The axis can be parameterized by:
    n = [sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)]

    Eg.,
    1. Rotation about the Z-axis: (0 0 1)
    theta = 0
    phi = 0
    2. Rotation about the X-axis: (1 0 0)
    theta = pi/2
    phi = 0
    3. Rotation about the Y-axis: (0 1 0)
    theta = pi/2
    phi = pi/2
    4. Rotations about the Hadamard axis: (1 0 1)/sqrt(2)
    theta = pi/4
    phi = 0
    """
    theta = params[0]
    phi = params[1]
    delta = params[1]
    axis = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        dtype=np.longdouble,
    )
    exponent = np.zeros((2, 2), dtype=np.complex128)
    for i in range(3):
        exponent = exponent + axis[i] * gv.Pauli[i + 1, :, :]
    krauss = np.zeros((1, 2, 2), dtype=np.complex128)
    krauss[0, :, :] = linalg.expm(-1j * delta * np.pi * exponent)
    return krauss


def ZRotation(params):
    """
    Kraus operators for rotation about the Z-axis.
    """
    return FixedRotation(0, 0, params[0])


def XRotation(params):
    """
    Kraus operators for rotation about the X-axis.
    """
    return FixedRotation(np.pi / 2, 0, params[0])


def YRotation(params):
    """
    Kraus operators for rotation about the Y-axis.
    """
    return FixedRotation(np.pi / 2, np.pi / 2, params[0])


def HRotation(params):
    """
    Kraus operators for rotation about the Hadamard (X + Z) axis.
    """
    return FixedRotation(0, np.pi / 4, params[0])


def PhotonLoss(params):
    """
    Kraus operators for the photon loss channel.
    """
    return pl.PLKraussOld(params[0], params[1])


def RandomStochastic(params):
    """
    Kraus operators for a random CPTP channel.
    """
    if len(params) < 2:
        krauss = RandomCPTP(params[0], 0)
    else:
        krauss = RandomCPTP(params[0], int(params[1]) - 1)
    return kraus


def RandomHamiltonian(params):
    """
    Kraus operators for the action of a random Hamiltonian on the qubit.
    """
    availmethods = ["exp", "qr", "haar", "hyps", "pauli"]
    if len(params) < 2:
        krauss = RandomUnitary(params[0], 2, "exp", None)[np.newaxis, :, :]
    else:
        krauss = RandomUnitary(params[0], 2, availmethods[int(params[1]) - 1], None)[
            np.newaxis, :, :
        ]
    return krauss


def CorrelatedPauli(params):
    """
    Kraus operators for a random fully correlated channel.
    The output in this case is the list of probabilities of n-qubit Pauli errors.
    See https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal.
    """
    sigma = 1
    mu = float(params[0])
    sigma = mu / 2
    lower = max(10e-3, mu - 0.1)
    upper = min(1 - 10e-3, mu + 0.1)
    # print("mu = {}, sigma = {}, upper = {}, lower = {}".format(mu, sigma, upper, lower))
    X = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    pauli_probs = RandomPauliChannel(X.rvs(), int(params[1]))
    return pauli_probs


def UserDefinedKraus(params):
    """
    Krauss operators from a file, input by the user.
    """
    return UserdefQC(chType, params)
