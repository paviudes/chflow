import os

try:
    import numpy as np
    from scipy import linalg as linalg
    from scipy.stats import truncnorm
except:
    pass
from define import globalvars as gv
from define.randchans import RandomCPTP, RandomPauliChannel
from define import photonloss as pl
from define import gendamp as gd
from define import chanreps as crep


def GetKraussForChannel(chType, *params):
    # Return the Krauss operators of a few types of quantum channels
    root = 1 / np.longdouble(16)
    rootunity = np.array(
        [[np.exp(1j * np.pi * root), 0], [0, np.exp(-1j * np.pi * root)]],
        dtype=np.complex128,
    )

    if chType == "id":
        krauss = np.zeros((1, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = np.eye(2)

    elif chType == "ad":
        # Amplitude damping channel
        krauss = np.zeros((2, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = np.array([[1, 0], [0, np.sqrt(1 - params[0])]])
        krauss[1, :, :] = np.array([[0, np.sqrt(params[0])], [0, 0]])

    elif chType == "bf":
        # Bit flip channel
        krauss = np.zeros((2, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = np.sqrt(1 - params[0]) * np.eye(2)
        krauss[1, :, :] = np.sqrt(params[0]) * gv.Pauli[1, :, :]

    elif chType == "pd":
        # Phase flip channel
        krauss = np.zeros((2, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = np.sqrt(1 - params[0]) * np.eye(2)
        krauss[1, :, :] = np.sqrt(params[0]) * gv.Pauli[3, :, :]

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
        krauss = np.zeros((2, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = np.sqrt(1 - params[0]) * np.eye(2)
        krauss[1, :, :] = np.sqrt(params[0]) * gv.Pauli[2, :, :]

    elif chType == "dp":
        # Depolarizing flip channel
        krauss = np.zeros((4, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = np.sqrt(1 - params[0]) * np.eye(2)
        krauss[1, :, :] = np.sqrt(params[0] / 3) * gv.Pauli[1, :, :]
        krauss[2, :, :] = np.sqrt(params[0] / 3) * gv.Pauli[2, :, :]
        krauss[3, :, :] = np.sqrt(params[0] / 3) * gv.Pauli[3, :, :]

    elif chType == "pauli":
        # Generic Pauli channel
        krauss = np.zeros((4, 2, 2), dtype=np.complex128)
        for i in range(4):
            krauss[i, :, :] = np.sqrt(params[i]) * gv.Pauli[i, :, :]

    elif chType == "rtz":
        # Rotation about the Z-axis
        krauss = np.zeros((1, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = linalg.expm(-1j * np.pi * params[0] * gv.Pauli[3, :, :])
        # print("krauss\n%s" % (np.array_str(krauss)))

    elif chType == "rtzpert":
        # Inexact rotations about the Z-axis -- the rotation angle is a gaussian random number with mean = theta and variance = 1.
        krauss = np.zeros((1, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = linalg.expm(
            -1j
            * np.pi
            * (np.random.normal(loc=params[0], scale=params[0] / 2))
            * gv.Pauli[3, :, :]
        )

    elif chType == "rtx":
        # Rotation about the X-axis
        krauss = np.zeros((1, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = linalg.expm(-1j * np.pi * params[0] * gv.Pauli[1, :, :])

    elif chType == "rtxpert":
        # Random second order perturbations to a rotation about the Z-axis
        krauss = np.zeros((1, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = linalg.expm(
            -1j
            * np.pi
            * (np.random.normal(loc=params[0], scale=1.0))
            * gv.Pauli[1, :, :]
        )

    elif chType == "rty":
        # Rotation about the Y-axis
        krauss = np.zeros((1, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = linalg.expm(-1j * np.pi * params[0] * gv.Pauli[2, :, :])

    elif chType == "rtypert":
        # Random second order perturbations to a rotation about the Z-axis
        krauss = np.zeros((1, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = linalg.expm(
            -1j
            * np.pi
            * (np.random.normal(loc=params[0], scale=1.0))
            * gv.Pauli[2, :, :]
        )

    elif chType == "rtnp":
        # This is the channel in a generic rotation channel about some axis of the Bloch sphere
        # It represents a Phase damping along an axis in the X-Z plane of the Bloch sphere: n = [sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)]
        # delta represents the rotation angle
        # From https://arxiv.org/abs/1612.02830
        # For Steane code
        # theta = 0.8
        # phi = 1.3
        # For 5-qubit code
        # theta = pi/3
        # phi = pi/4
        # Rotations about the Hadamard axis: (1 1 0)/sqrt(2)
        # theta = pi/2
        # phi = pi/4
        phi = params[1]
        theta = params[2]
        axis = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            dtype=np.longdouble,
        )
        exponent = np.zeros((2, 2), dtype=np.complex128)
        for i in range(3):
            exponent = exponent + axis[i] * gv.Pauli[i + 1, :, :]
        krauss = np.zeros((1, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = linalg.expm(-1j * params[0] * np.pi * exponent)

    elif chType == "strtz":
        # Stochastic over-rotation about the Z-axis
        # E(rho) = (1-p) rho + p exp(i pi/4 Z) rho exp(-i pi/4 Z)
        krauss = np.zeros((3, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = np.sqrt(1 - params[0]) * np.eye(2)
        krauss[1, :, :] = np.sqrt(params[0]) * linalg.expm(
            1j * np.pi / np.longdouble(4) * gv.Pauli[3, :, :]
        )

    elif chType == "rtpd":
        # Rotation combined with phase damping. The channel is a stochastic mixture of a dephasing channel and a rotation channel.
        # There are two parameters for this channel. First the rotation angle and second, the dephasing.
        # We may set either of them to be fixed and scan over the others.
        dephase = params[1]
        krauss = np.zeros((3, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = np.sqrt(1 - dephase) * linalg.expm(
            -1j * np.pi * params[0] * gv.Pauli[3, :, :]
        )
        krauss[1, :, :] = np.sqrt(dephase) * np.dot(
            linalg.expm(-1j * np.pi * params[0] * gv.Pauli[3, :, :]), gv.Pauli[3, :, :]
        )

    elif chType == "shd":
        # Stochastic Hadamard channel
        krauss = np.zeros((2, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = np.sqrt(1 - params[0]) * np.eye(2)
        krauss[1, :, :] = np.sqrt(params[0]) * gv.hadamard

    elif chType == "sru":
        # Stochastic root of unity channel
        # Parameters: probability of rotation, angle of rotation as a fraction of pi, root of unity
        krauss = np.zeros((2, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = np.sqrt(1 - params[0]) * np.eye(2)
        krauss[1, :, :] = np.sqrt(params[0]) * linalg.expm(
            -1j * np.pi * params[1] * params[2]
        )

    elif chType == "rru":
        # Rotation about a root of unity axis
        # Parameters: angle of rotation as a fraction of pi, root of unity
        krauss = np.zeros((2, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = linalg.expm(-1j * np.pi * params[0] * params[1])

    elif chType == "simcorr":
        # This mimics a correlated error model.
        # Here we study an i.i.d noise model which does Depolarizing with probability p and Hadamard with probability p^(1/k)
        # Parameters: p, k
        # So that it represents a correlated noise model that does Depolarizing at the single qubit level and a Hadamard on k-qubits simultaneously.
        krauss = np.zeros((5, 2, 2), dtype=np.complex128)
        krauss[0, :, :] = np.sqrt(
            1 - params[0] - np.power(params[0], 1 / np.longdouble(params[1]))
        ) * np.eye(2)
        krauss[1, :, :] = np.sqrt(params[0] / 3) * gv.Pauli[1, :, :]
        krauss[2, :, :] = np.sqrt(params[0] / 3) * gv.Pauli[2, :, :]
        krauss[3, :, :] = np.sqrt(params[0] / 3) * gv.Pauli[3, :, :]
        krauss[4, :, :] = (
            np.power(params[0], 1 / np.longdouble(2 * params[1])) * hadamard
        )

    elif chType == "pl":
        # This is the photon loss channel.
        # krauss = pl.PLKrauss(params[0], params[1])
        krauss = pl.PLKraussOld(params[0], params[1])

    elif chType == "rand":
        if len(params) < 2:
            krauss = RandomCPTP(params[0], 0)
        else:
            krauss = RandomCPTP(params[0], int(params[1]) - 1)

    elif chType == "pcorr":
        # This is a correlated Pauli channel.
        # The output in this case is the list of probabilities of n-qubit Pauli errors.
        # See https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal.
        sigma = 1
        mu = float(params[0])
        sigma = mu / 2
        lower = max(10e-3, mu - 0.1)
        upper = min(1 - 10e-3, mu + 0.1)
        # print("mu = {}, sigma = {}, upper = {}, lower = {}".format(mu, sigma, upper, lower))
        X = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        krauss = RandomPauliChannel(X.rvs(), int(params[1]))

    elif os.path.isfile(chType) == 1:
        krauss = UserdefQC(chType, params)

    else:
        print("\033[93mUnknown channel type\033[0m")
        krauss = np.identity(2, dtype=np.complex128)[np.newaxis, :, :]

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
