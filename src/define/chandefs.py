import os
import numpy as np
from scipy import linalg as linalg
from define import globalvars as gv
from define.randchans import (
    RandomCPTP,
    RandomPauliChannel,
    RandomUnitary,
    UncorrelatedRandomPauli,
)
from define import photonloss as pl
from define import gendamp as gd
from define import chanreps as crep
from define.QECCLfid.chans import GetProcessChi


def Identity():
    """
    Kraus operators for the identity channel.
    """
    return np.eye(2, dtype=np.complex128)[np.newaxis, :, :]


def Bitflip(params):
    """
    Kraus operators for the Bit flip channel
    """
    return PauliChannel([1 - params[0], params[0], 0, 0])


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
    return PauliChannel([1 - params[0], 0, 0, params[0]])


def BitPhaseFlip(params):
    """
    Kraus operators for the bit phase flip channel.
    """
    return PauliChannel([1 - params[0], 0, params[0], 0])


def Depolarizing(params):
    """
    Kraus operators for the Depolarizing channel.
    """
    return PauliChannel([1 - params[0], params[0] / 3, params[0] / 3, params[0] / 3])


def PauliChannel(params):
    """
    Kraus operators for the Pauli channel.
    """
    krauss = np.zeros((4, 2, 2), dtype=np.complex128)
    for i in range(4):
        krauss[i, :, :] = np.sqrt(params[i]) * gv.Pauli[i, :, :]
    return krauss


def WorstPauliChannel(params):
    """
    The worst n-qubit Pauli channel for a given infidelity.
    """
    qcode = params[0]
    nqubit_infid = params[1]
    pauliprobs = np.random.rand(4 ** qcode.N)
    pauliprobs[qcode.PauliCorrectableIndices] = 0
    pauliprobs[0] = 1 - nqubit_infid
    pauliprobs[1:] = nqubit_infid * pauliprobs[1:] / np.sum(pauliprobs[1:])
    return pauliprobs


def ArbitraryNormalRotation(params):
    """
    Asymmetric and uniformly random rotation to n qubits.
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
    specs = [1, 0.01, -1, -1, -1]
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
    krauss = unitaries[:, np.newaxis, :, :]
    return krauss


def ArbitraryUniformRotation(params):
    """
    Asymmetric and Gaussian random rotation to n qubits.
    """
    names = ["N", "mean_delta", "std_delta"]
    specs = [1, 0.01, -1]
    for i in range(len(params)):
        specs[i] = params[i]

    nqubits = int(specs[0])

    mean_delta = specs[1]
    std_delta = specs[2]
    if std_delta < 0:
        std_delta = mean_delta / (-1 * std_delta)

    unitaries = np.zeros((nqubits, 2, 2), dtype=np.complex128)
    for i in range(nqubits):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        delta = np.random.normal(mean_delta, std_delta)
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
    krauss = unitaries[:, np.newaxis, :, :]
    return krauss


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
    delta = params[2]
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
    return FixedRotation([0, 0, params[0]])


def XRotation(params):
    """
    Kraus operators for rotation about the X-axis.
    """
    return FixedRotation([np.pi / 2, 0, params[0]])


def YRotation(params):
    """
    Kraus operators for rotation about the Y-axis.
    """
    return FixedRotation([np.pi / 2, np.pi / 2, params[0]])


def HRotation(params):
    """
    Kraus operators for rotation about the Hadamard (X + Z) axis.
    """
    return FixedRotation([0, np.pi / 4, params[0]])


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
    return krauss


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


def CorrelatedNonPauli(params, method):
    """
    Return a correlated non-Pauli channel in the Pauli-Liouville respresentation.
    """
    (phychan, rawchan, interactions) = GetProcessChi(params[0], method, *params[1:])
    return (phychan, rawchan, interactions)


def CorrelatedPauli(params):
    """
    Kraus operators for a random fully correlated channel.
    The output in this case is the list of probabilities of n-qubit Pauli errors.
    See https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal.
    available methods = ["uniform", "poisson"]
    """
    kwargs = {
        "qcode": params[0],
        "infid": params[1],
        "method": int(params[2]) - 1,
        "iid_fraction": float(params[3]),
        "subset_fraction": float(params[4]),
    }
    kwargs["infid"] = np.abs(np.random.normal(params[1], 0.1 * params[1]))
    # print("args = {}".format(kwargs))
    return RandomPauliChannel(kwargs)


def UncorrelatedPauli(params):
    """
    Uncorrelated Pauli channel with controllable infidenlity.
    """
    infid = np.abs(np.random.normal(params[0], 0.1 * params[0]))
    return UncorrelatedRandomPauli(infid)


def GetKraussForChannel(chType, *params):
    # Return the Krauss operators of a few types of quantum channels
    # print("Getting Kraus for channel: {} with parameters {}".format(chType, params))
    if chType == "id":
        krauss = Identity()

    elif chType == "ad":
        # Amplitude damping channel
        krauss = AmplitudeDamping(params)

    elif chType == "bf":
        # Bit flip channel
        krauss = BitFlip(params)

    elif chType == "pd":
        # Phase flip channel
        krauss = Dephasing(params)

    elif chType == "gd":
        # Generalized damping channel
        krauss = gd.GeneralizedDamping(params[0], params[1])

    elif chType == "gdt":
        # Generalized damping channel
        krauss = gd.GDTimeScales(params[0], params[1])

    elif chType == "gdtx":
        # Generalized damping channel
        krauss = gd.GDTimeScalesExplicit(params[0], params[1], params[2])

    elif chType == "bpf":
        # Bit-Phase flip channel
        krauss = BitPhaseFlip(params)

    elif chType == "dp":
        # Depolarizing flip channel
        krauss = Depolarizing(params)

    elif chType == "pauli":
        # Generic Pauli channel
        krauss = PauliChannel(params)

    elif chType == "up":
        # Generic Pauli channel
        krauss = UncorrelatedPauli(params)

    elif chType == "rtz":
        # Rotation about the Z-axis
        krauss = ZRotation(params)
        # print("krauss\n%s" % (np.array_str(krauss)))

    elif chType == "rtx":
        # Rotation about the X-axis
        krauss = krauss = XRotation(params)

    elif chType == "rty":
        # Rotation about the Y-axis
        krauss = YRotation(params)

    elif chType == "rtas":
        # Asymmetric Gaussian random rotation to n qubits
        krauss = ArbitraryNormalRotation(params)

    elif chType == "rtasu":
        # Asymmetric uniformly random rotation to n qubits
        krauss = ArbitraryUniformRotation(params)

    elif chType == "rtnp":
        # This is the channel in a generic rotation channel about some axis of the Bloch sphere
        krauss = FixedRotation(params)

    elif chType == "pl":
        # This is the photon loss channel.
        krauss = PhotonLoss(params)

    elif chType == "rand":
        # Random CPTP channel.
        krauss = RandomStochastic(params)

    elif chType == "randunit":
        # Random Hamiltonian on the qubit.
        krauss = RandomHamiltonian(params)

    elif chType == "pcorr":
        # This is a correlated Pauli channel.
        krauss = CorrelatedPauli(params)

    elif chType == "usum":
        krauss = CorrelatedNonPauli(params, "sum_unitaries")

    elif chType == "ising":
        krauss = CorrelatedNonPauli(params, "ising")

    elif chType == "cptp":
        krauss = CorrelatedNonPauli(params, "sum_cptps")

    elif chType == "wpc":
        # Worst Pauli channel for a infidelity
        krauss = WorstPauliChannel(params)

    elif os.path.isfile(chType) == 1:
        krauss = UserdefQC(chType, params)

    else:
        print("\033[93mUnknown channel type, resetting to the identity channel.\033[0m")
        krauss = Identity()

    return krauss


def UserdefQC(fname, params):
    # Custom channel defined using a file.
    # The file will contain list of variables and a representation of the channel with numbers as well as symbolic entries.
    # We will replace the symbolic entries with values for the variables in the "params" array.
    # The i-th variable in the list of variables will be substituted with the i-th value in params.
    # At last, we have to convert the representation to Krauss.
    fail = 0
    process = np.zeros((4, 4), dtype=np.longdouble)
    variables = {}
    with open(fname, "r") as cfp:
        row = 0
        col = 0
        for (lno, line) in enumerate(cfp):
            if not (line[0] == "#"):
                contents = map(
                    lambda numstr: numstr.strip(" "), line[0].strip("\n").split(" ")
                )
                if contents[0] == "vars":
                    for i in range(1, len(contents)):
                        variables.update({contents[i]: params[i - 1]})
                else:
                    for j in range(4):
                        # If the matrix element is not a number, evaluate the expression to convert it to a number.
                        if IsNumber(contents[j] == 0):
                            contents[j] = Evaluate(contents[j], variables)
                            if contents[j] == "nan":
                                fail = 1
                                contents[j] = 0
                        process[row, col] = np.longdouble(contents[j])
                        col = col + 1
                row = row + 1
    if fail == 1:
        print(
            "\033[2mSome elements failed to load properly, their values have been replaced by 0.\033[0m"
        )
    krauss = crep.ConvertRepresentations(process, "process", "krauss")
    return krauss


def Evaluate(expr, values):
    # evaluate an expression by substituting the values of the variables provided.
    # Return "nan" is the result is not an expression
    for var in values:
        expr.replace(var, values[var])
    try:
        result = eval(expr)
    except:
        result = "nan"
    return result


def IsNumber(numstr):
    # Determine if a string contains a number of not.
    # Try to convert it to a float. If it fails, it is not a number. Else it is.
    try:
        np.float(numstr)
    except ValueError:
        return 0
    return 1
