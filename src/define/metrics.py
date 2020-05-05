import os
import sys
import time
import datetime as dt
import ctypes as ct

try:
    import numpy as np
    from scipy import linalg as linalg
    import matplotlib
    from matplotlib import colors, ticker, cm

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    import multiprocessing as mp
    import picos as pic
    import cvxopt as cvx
except:
    pass
from define import globalvars as gv
from define import chanreps as crep
from define import qchans as qc
from define import fnames as fn
from define.QECCLfid import uncorrectable as uc

##########################################
Metrics = {
    "dnorm": {
        "name": "Diamond norm",
        "phys": "Diamond distance of the physical channel",
        "log": "Diamond distance of the logical channel",
        "latex": "$|| \\mathcal{E} - \\mathsf{id} ||_{\\diamondsuit}$",
        "marker": u"+",
        "color": "crimson",
        "desc": "See Sec. 4 of DOI: 10.4086/toc.2009.v005a011.",
        "func": "lambda J, kwargs: DiamondNorm(J, kwargs)",
    },
    "errp": {
        "name": "Error probability",
        "phys": "Error probability of the physical channel",
        "log": "Error probability of the logical channel",
        "latex": "p_{err}(\\mathcal{E})$",
        "marker": u"*",
        "color": "darkblue",
        "desc": "1 - p* where p* is the maximum value between 0 and 1 such that J - (p*)*J_id is a valid choi matrix, up to normalization.",
        "func": "lambda J, kwargs: ErrorProbability(J, kwargs)",
    },
    "entropy": {
        "name": "Von Neumann Entropy",
        "phys": "Entropy of the physical channel",
        "log": "Entropy of the logical channel",
        "latex": "$S(\\mathcal{J})$",
        "marker": u"p",
        "color": "brown",
        "desc": "Von-Neumann entropy of the channel's Choi matrix.",
        "func": "lambda J, kwargs: Entropy(J, kwargs)",
    },
    "infid": {
        "name": "Infidelity",
        "phys": "Infidelity of the physical channel",
        "log": "Infidelity of the logical channel",
        "latex": "$1 - F$",
        "marker": u"s",
        "color": "forestgreen",
        "desc": "1 - Fidelity between the input Choi matrix and the Choi matrix corresponding to the identity state.",
        "func": "lambda J, kwargs: Infidelity(J, kwargs)",
    },
    "np1": {
        "name": "Non Pauliness",
        "phys": "Non-Pauliness of the physical channel",
        "log": "Non-Pauliness of the logical channel",
        "latex": "$\\mathcal{W}(\\mathcal{E})$",
        "marker": u"v",
        "color": "lavender",
        "desc": "L2 norm of the difference between the channel's Chi matrix and it's twirled approximation.",
        "func": "lambda J, kwargs: NonPaulinessChi(J, kwargs)",
    },
    "np2": {
        "name": "Non Pauliness",
        "phys": "Non-Pauliness of the physical channel",
        "log": "Non-Pauliness of the logical channel",
        "latex": "$np_{2}(\\mathcal{E})$",
        "marker": u"1",
        "color": "maroon",
        "desc": "Least fidelity between the channel's Choi matrix and a bell state.",
        "func": "lambda J, kwargs: NonPaulinessChoi(J, kwargs)",
    },
    "np4": {
        "name": "Non Pauliness",
        "phys": "Non-Pauliness of the physical channel",
        "log": "Non-Pauliness of the logical channel",
        "latex": "$np_{4}(\\mathcal{E})$",
        "marker": u"3",
        "color": "turquoise",
        "desc": 'Maximum "amount" of Pauli channel that can be subtracted from the input Pauli channel, such that what remains is still a valid quantum channel.',
        "func": "lambda J, kwargs: NonPaulinessRemoval(J, kwargs)",
    },
    "trn": {
        "name": "Trace norm",
        "phys": "Trace-distance of the physical channel",
        "log": "Trace-distance of the logical channel",
        "latex": "$\\left|\\left|\\mathcal{J} - \\mathsf{id}\\right|\\right|_{1}$",
        "marker": u"8",
        "color": "black",
        "desc": "Trace norm of the difference between the channel's Choi matrix and the input Bell state, Trace norm of A is defined as: Trace(Sqrt(A^\\dagger . A)).",
        "func": "lambda J, kwargs: TraceNorm(J, kwargs)",
    },
    "frb": {
        "name": "Frobenious norm",
        "phys": "Frobenious norm of the physical channel",
        "log": "Frobenious norm of the logical channel",
        "latex": "$\\left|\\left|\\mathcal{J} - \\mathsf{id}\\right|\\right|_{2}$",
        "marker": u"d",
        "color": "chocolate",
        "desc": "Frobenious norm of the difference between the channel's Choi matrix and the input Bell state, Frobenious norm of A is defined as: Sqrt(Trace(A^\\dagger . A)).",
        "func": "lambda J, kwargs: FrobeniousNorm(J, kwargs)",
    },
    "bd": {
        "name": "Bures distance",
        "phys": "Bures distance of the physical channel",
        "log": "Bures distance of the logical channel",
        "latex": "$\\Delta_{B}(\\mathcal{J})$",
        "marker": u"<",
        "color": "goldenrod",
        "desc": "Bures distance between the channel's Choi matrix and the input Bell state. Bures distance between A and B is defined as: sqrt( 2 - 2 * sqrt( F ) ), where F is the Uhlmann-Josza fidelity between A and B.",
        "func": "lambda J, kwargs: BuresDistance(J, kwargs)",
    },
    "uhl": {
        "name": "Uhlmann infidelity",
        "phys": "Uhlmann Infidelity of the physical channel",
        "log": "Uhlmann Infidelity of the logical channel",
        "latex": "$1 - F_{U}(\\mathcal{J})$",
        "marker": u"h",
        "color": "midnightblue",
        "desc": "1 - Uhlmann-Jozsa fidelity between the channel's Choi matrix and the input Bell state. The Uhlmann-Jozsa fidelity between A and B is given by: ( Trace( sqrt( sqrt(A) B sqrt(A) ) ) )^2.",
        "func": "lambda J, kwargs: UhlmanFidelity(J, kwargs)",
    },
    "unitarity": {
        "name": "Non-unitarity",
        "phys": "Non-unitarity of the physical channel",
        "log": "Non-unitarity of the logical channel",
        "latex": "$1-\\mathcal{u}(\\mathcal{E})$",
        "marker": u"^",
        "color": "fuchsia",
        "desc": "In the Pauli-Liouville representation of the channel, P, the unitarity is given by: ( sum_(i,j; i not equal to j) |P_ij|^2 ).",
        "func": "lambda J, kwargs: NonUnitarity(J, kwargs)",
    },
    "uncorr": {
        "name": "Uncorrectable error probability",
        "phys": "Uncorrectable error probability",
        "log": "Uncorrectable error probability of the logical channel",
        "latex": "$p_{u}$",
        "marker": u">",
        "color": "blue",
        "desc": "The total probability of uncorrectable (Pauli) errors.",
        "func": "lambda P, kwargs: UncorrectableProb(P, kwargs)",
    },
    "anisotropy": {
        "name": "Anisotropy",
        "phys": "Anisotropy of the physical channel",
        "log": "Anisotropy of the logical channel",
        "latex": "$p_{Y}/p_{X} + p_{Z}/p_{X}$",
        "marker": u"d",
        "color": "goldenrod",
        "desc": "Anisotropy between Y errors and X, Z errors.",
        "func": "lambda J, kwargs: Anisotropy(J, kwargs)",
    },
}
##########################################


def HermitianConjugate(mat):
    # Return the Hermitian conjugate of a matrix
    return np.conjugate(np.transpose(mat))


def IsDiagonal(mat):
    # Check if a matrix is diagonal
    atol = 10e-16
    if np.linalg.norm(np.diag(np.diag(mat)) - mat) <= atol:
        return 1
    return 0


def DiamondNormPhysical(choi, kwargs):
    # computes the diamond norm of the difference between an input Channel and another reference channel, which is by default, the identity channel
    # The semidefinite program outlined in Sec. 4 of DOI: 10.4086/toc.2009.v005a011 is used here.
    # See also: https://github.com/BBN-Q/matlab-diamond-norm/blob/master/src/dnorm.m
    # For some known types of channels, the Diamond norm can be computed efficiently
    # print("Function: dnorm")
    if IsDiagonal(crep.ConvertRepresentations(choi, "choi", "process")) == 1:
        dnorm = Infidelity(choi, kwargs)
    else:
        diff = (choi - gv.bell[0, :, :]).astype(complex)
        #### picos optimization problem
        prob = pic.Problem()
        # variables and parameters in the problem
        J = pic.new_param("J", cvx.matrix(diff))
        rho = prob.add_variable("rho", (2, 2), "hermitian")
        W = prob.add_variable("W", (4, 4), "hermitian")
        # objective function (maximize the hilbert schmidt inner product -- denoted by '|'. Here A|B means trace(A^\dagger * B))
        prob.set_objective("max", J | W)
        # adding the constraints
        prob.add_constraint(W >> 0)
        prob.add_constraint(rho >> 0)
        prob.add_constraint(("I" | rho) == 1)
        prob.add_constraint((W - ((rho & 0) // (0 & rho))) << 0)
        # solving the problem
        sol = prob.solve(verbose=0, maxit=500)
        dnorm = sol["obj"] * 2
        # print("SDP dnorm = %g" % (dnorm))
    return dnorm


def DiamondNorm(choi, kwargs):
    # Compute the Diamond norm of a physical channel or a set of logical channels.
    if qc.Channels[kwargs["channel"]]["Pauli"] == 1:
        return Infidelity(choi, kwargs)

    if kwargs["chtype"] == "physical":
        if kwargs["corr"] == 0:
            return DiamondNormPhysical(choi, kwargs)
        elif kwargs["corr"] == 2:
            chans_ptm = np.reshape(choi, [kwargs["qcode"].N, 4, 4])
            dnorm = 0
            for q in range(chans_ptm.shape[0]):
                dnorm = dnorm + DiamondNormPhysical(
                    crep.ConvertRepresentations(chans_ptm[q, :, :], "process", "choi"),
                    {"corr": 0, "chtype": kwargs["chtype"]},
                )
        else:
            print("Diamond form for fully correlated channels is not yet set up.")
    else:
        dnorm = np.zeros(kwargs["levels"], dtype=np.double)
        for l in range(kwargs["levels"]):
            dnorm[l] = DiamondNormPhysical(
                choi[l, :, :],
                dict(
                    [(key, kwargs[key]) for key in kwargs if not (key == "chtype")]
                    + [("chtype", "physical")]
                ),
            )
    return dnorm


def ErrorProbability(choi, kwargs):
    # compute the error probability associated with the Channel whose jamilowski form is given
    # The error probability is defined as: 1 - p* where p* is the maximum value between 0 and 1 such that J - (p*)*J_id is a valid choi matrix, up to normalization.
    #### picos optimization problem
    prob = pic.Problem()
    # variables and parameters in the problem
    Jid = pic.new_param("Jid", cvx.matrix(gv.bell[0, :, :]))
    J = pic.new_param("J", cvx.matrix(choi))
    p = prob.add_variable("p", 1)
    # adding the constraints
    prob.add_constraint(p <= 1)
    prob.add_constraint(p >= 0)
    prob.add_constraint(J - p * Jid >> 0)
    # objective function
    prob.set_objective("max", p)
    # solving the problem
    prob.solve(verbose=0, maxit=10)
    errp = 1 - prob.obj_value()
    return errp


def Entropy(choi, kwargs):
    # Compute the Von-Neumann entropy of the input Choi matrix.
    # The idea is that a pure state (which corresponds to unitary channels) will have zero entropy while any mixed state which corresponds to a channel that does not preserve the input state, has finiste entropy.
    sgvals = np.linalg.svd(choi.astype(np.complex), compute_uv=0)
    entropy = 0
    for i in range(sgvals.shape[0]):
        if abs(np.imag(sgvals[i])) < 10e-50:
            if sgvals[i] > 0:
                entropy = entropy - sgvals[i] * np.log(sgvals[i])
    return entropy


def InfidelityPhysical(choi, kwargs):
    # Compute the Fidelity between the input channel and the identity channel.
    # For independent errors, the single qubit channel must be in the Choi matrix representation.
    # For correlated Pauli channel, the channel is represented as a vector of diagonal elements of the respective Pauli trnsfer matrix.
    # For a tensor product of CPTP maps, the fidelity is the sum of the fidelities of the maps in the tensor product.
    if kwargs["corr"] == 0:
        fidelity = (1 / np.longdouble(2)) * np.longdouble(
            np.real(choi[0, 0] + choi[3, 0] + choi[0, 3] + choi[3, 3])
        )
    elif kwargs["corr"] == 1:
        fidelity = choi[0]
    else:
        fidelity = 1
        chans_ptm = np.reshape(choi, [kwargs["qcode"].N, 4, 4])
        for q in range(chans_ptm.shape[0]):
            fidelity = fidelity * (
                1
                - InfidelityPhysical(
                    crep.ConvertRepresentations(chans_ptm[q, :, :], "process", "choi"),
                    {"corr": 0},
                )
            )
    return 1 - fidelity


def Infidelity(choi, kwargs):
    # Compute the Infidelity for a physical channel or a set of logical channels.
    if kwargs["chtype"] == "physical":
        return InfidelityPhysical(choi, kwargs)
    else:
        infids = np.zeros(choi.shape[0], dtype=np.double)
        for l in range(choi.shape[0]):
            infids[l] = InfidelityPhysical(
                choi[l, :, :],
                dict(
                    [(key, kwargs[key]) for key in kwargs if not (key == "chtype")]
                    + [("chtype", "physical")]
                ),
            )
    return infids


def TraceNormPhysical(choi, kwargs):
    # Compute the trace norm of the difference between the input Choi matrix and the Choi matrix corresponding to the Identity channel
    # trace norm of A is defined as: Trace(Sqrt(A^\dagger . A))
    # https://quantiki.org/wiki/trace-norm
    if kwargs["corr"] == 0:
        trn = np.linalg.norm((choi - gv.bell[0, :, :]).astype(np.complex), ord="nuc")
    elif kwargs["corr"] == 2:
        chans_ptm = np.reshape(choi, [kwargs["qcode"].N, 4, 4])
        trn = 0
        for q in range(chans_ptm.shape[0]):
            trn = trn + TraceNormPhysical(
                crep.ConvertRepresentations(chans_ptm[q, :, :], "process", "choi"),
                {"corr": 0},
            )
    else:
        trn = 0
    return trn


def TraceNorm(choi, kwargs):
    # Compute the trace norm of a physical channel or a set of logical channels.
    if kwargs["chtype"] == "physical":
        return TraceNormPhysical(choi, kwargs)
    else:
        trnorm = np.zeros(choi.shape[0], dtype=np.double)
        for l in range(choi.shape[0]):
            trnorm[l] = TraceNormPhysical(
                choi[l, :, :],
                dict(
                    [(key, kwargs[key]) for key in kwargs if not (key == "chtype")]
                    + [("chtype", "physical")]
                ),
            )
    return trnorm


def FrobeniousNormPhysical(choi, kwargs):
    # Compute the Frobenious norm of the difference between the input Choi matrix and the Choi matrix corresponding to the Identity channel
    # Frobenious of A is defined as: sqrt(Trace(A^\dagger . A))
    # https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
    # Optimized in C
    if kwargs["corr"] == 0:
        frb = np.linalg.norm((choi - gv.bell[0, :, :]).astype(np.complex), ord="fro")
    elif kwargs["corr"] == 2:
        chans_ptm = np.reshape(choi, [kwargs["qcode"].N, 4, 4])
        frb = 0
        for q in range(chans_ptm.shape[0]):
            frb = frb + FrobeniousNormPhysical(
                crep.ConvertRepresentations(chans_ptm[q, :, :], "process", "choi"),
                {"corr": 0},
            )
    else:
        frb = 0
    return frb


def FrobeniousNorm(choi, kwargs):
    # Compute the Frobenious norm of a physical channel or a set of logical channels.
    if kwargs["chtype"] == "physical":
        return FrobeniousNormPhysical(choi, kwargs)
    else:
        frbnorm = np.zeros(choi.shape[0], dtype=np.double)
        for l in range(choi.shape[0]):
            frbnorm[l] = FrobeniousNormPhysical(
                choi[l, :, :],
                dict(
                    [(key, kwargs[key]) for key in kwargs if not (key == "chtype")]
                    + [("chtype", "physical")]
                ),
            )
    return frbnorm


def BuresDistance(choi, kwargs):
    # Compute the Bures distance between the input Choi matrix and the Choi matrix corresponding to the identity channel
    # The Bures distance between A and B is defined as: sqrt( 2 - 2 * sqrt( F ) ), where F is the Uhlmann-Josza fidelity between A and B
    # http://iopscience.iop.org/article/10.1088/1751-8113/40/37/010/meta
    # https://quantiki.org/wiki/fidelity
    fiduj = UhlmanFidelity(choi)
    if fiduj < -10e-50:
        distB = 0
    else:
        distB = np.sqrt(2 - 2 * np.sqrt(1 - fiduj))
    return distB


def UhlmanFidelity(choi, kwargs):
    # Compute the Uhlmann-Josza fidelity between the input Choi matrix and the Choi matrix corresponding to the identity channel
    # The Uhlmann fidelity between A and B is given by: ( Trace( sqrt( sqrt(A) B sqrt(A) ) ) )^2
    # https://www.icts.res.in/media/uploads/Old_Talks_Lectures/Slides/1266647057kzindia10.pdf
    # http://www.sciencedirect.com/science/article/pii/S0375960101006405
    (eigvals, eigvecs) = np.linalg.eigh(choi.astype(np.complex))
    overlap = np.zeros((4, 4), dtype=np.double)
    for i in range(eigvals.shape[0]):
        for j in range(eigvals.shape[0]):
            outer = np.dot(eigvecs[i, :, np.newaxis], eigvecs[np.newaxis, j, :])
            overlap = (
                overlap
                + np.sqrt(eigvals[i] * eigvals[j])
                * np.trace(np.dot(gv.bell[0, :, :], outer))
                * outer
            )
    eigvals = np.linalg.eigvals(np.real(overlap))
    fiduj = np.sum(np.sqrt(eigvals), dtype=np.longdouble) * np.sum(
        np.sqrt(eigvals), dtype=np.longdouble
    )
    return 1 - fiduj


def NonUnitarity(choi, kwargs):
    # Compute the unitarity metric for the input choi matrix of a channel
    # Convert from the Choi matrix to the process matrix
    # For a process matrix P, the unitarity is given by: ( sum_(i,j) |P_ij|^2 )
    # http://iopscience.iop.org/article/10.1088/1367-2630/17/11/113020
    process = crep.ConvertRepresentations(choi, "choi", "process")
    unitarity = np.sum(np.abs(process[1:, 1:]) * np.abs(process[1:, 1:])) / (
        process.shape[0] - 1
    )
    # print("unitarity = {}".format(unitarity))
    return unitarity


def NonPaulinessChi(choi, kwargs):
    # Quantify the behaviour of a quantum channel by its difference from a Pauli channel
    # Convert the input Choi matrix to it's Chi-representation
    # Compute the ration between the  sum of offdiagonal entries to the sum of disgonal entries.
    # While computing the sums, consider the absolution values of the entries.
    chi = crep.ConvertRepresentations(choi, "choi", "chi")
    # print("chi\n%s" % (np.array_str(chi, max_line_width=150)))
    atol = 10e-20
    nonpauli = 0.0
    for i in range(4):
        for j in range(4):
            if not (i == j):
                if np.abs(chi[i, i]) * np.abs(chi[j, j]) >= atol:
                    # print("contribution = %g" % (np.power(np.abs(chi[i, j]), 2.0)/(np.abs(chi[i, i]) * np.abs(chi[j, j]))))
                    nonpauli = nonpauli + np.power(np.abs(chi[i, j]), 2.0) / (
                        np.abs(chi[i, i]) * np.abs(chi[j, j])
                    )
    # print("nonpauli = %g." % (nonpauli))
    return nonpauli


def NonPaulinessChoi(choi, kwargs):
    # Quantify the behaviour of a quantum channel by its difference from a Pauli channel
    # Compute the least fidelity between the input Choi matrix and any of the bell states.
    overlaps = map(np.abs, map(np.trace, np.tensordot(gv.bell, choi, axes=[[1], [1]])))
    nonpauli = 1 - max(overlaps)
    return nonpauli


def NonPaulinessRemoval(choi, kwargs):
    # Quantify the behaviour of a quantum channel by its difference from a Pauli channel
    # Pauliness is defined as the maximum "amount" of Pauli channel that can be subtracted from the input Pauli channel, such that what remains is still a valid quantum channel.
    bellstates = np.zeros((4, 4, 4), dtype=np.complex128)
    # Bell state |00> + |11>
    # X -- Bell state |01> + |10>
    # Y -- Bell state i(|10> - |01>)
    # Z -- Bell state |00> - |11>

    ### picos optimization problem
    prob = pic.Problem()
    # parameters and variables
    J = pic.new_param("J", cvx.matrix(choi))
    JI = pic.new_param("JI", cvx.matrix(gv.bell[0, :, :]))
    JX = pic.new_param("JX", cvx.matrix(gv.bell[1, :, :]))
    JY = pic.new_param("JY", cvx.matrix(gv.bell[2, :, :]))
    JZ = pic.new_param("JZ", cvx.matrix(gv.bell[3, :, :]))
    pp = prob.add_variable("pp", 4)
    p = prob.add_variable("p")
    # specifying constraints
    # probabilities sum to 1. Each is bounded above by 1, below by 0.
    for i in range(4):
        prob.add_constraint(pp[i] >= 0)
        prob.add_constraint(pp[i] <= 1)
    prob.add_constraint(np.sum(pp) == 1)
    # Fraction of Pauli channel that can to be removed
    prob.add_constraint(p >= 0)
    prob.add_constraint(p <= 1)
    # What remains after subtracting a Pauli channel is a valid Choi matrix, up to normalization
    prob.add_constraint(JI * pp[0] + JX * pp[1] + JY * pp[2] + JZ * pp[3] - p * J >> 0)
    # objective function --- maximize the sum of Probabilities of I, X, Y and Z errors
    prob.set_objective("max", p)
    # Solve the problem
    sol = prob.solve(verbose=0, maxit=100)
    nonPauli = 1 - sol["obj"]
    return nonPauli


def UncorrectableProb(channel, kwargs):
    # Compute the probability of uncorrectable errors for a code.
    if kwargs["corr"] == 0:
        pauliProbs = np.tile(
            np.real(np.diag(crep.ConvertRepresentations(channel, "choi", "chi"))),
            [kwargs["qcode"].N, 1, 1],
        )
    elif kwargs["corr"] == 1:
        pauliProbs = channel
    else:
        chans_ptm = np.reshape(channel, [kwargs["qcode"].N, 4, 4])
        pauliProbs = np.zeros((kwargs["qcode"].N, 4), dtype=np.double)
        for q in range(kwargs["qcode"].N):
            pauliProbs[q, :] = np.real(
                np.diag(
                    crep.ConvertRepresentations(chans_ptm[q, :, :], "process", "chi")
                )
            )
    return uc.ComputeUnCorrProb(pauliProbs, kwargs["qcode"], kwargs["levels"])


def Anisotropy(channel, kwargs):
    # Compute the anisotropy in the noise process.
    # The anisotropy is simply the 2 * Prob(Y) - (Prob(X) + Prob(Z))
    atol = 10e-8
    if kwargs["chtype"] == "physical":
        if kwargs["rep"] == "choi":
            chichan = np.real(crep.ConvertRepresentations(channel, "choi", "chi"))
        else:
            chichan = channel
        probs = np.diag(chichan)
        anisotropy = 0
        if (probs[1] > atol) and (probs[3] > atol):
            anisotropy = probs[2] / probs[1] + probs[2] / probs[3]
        # np.dot(np.diag(chichan), np.array([0, -1, 2, -1], dtype=np.double))
    else:
        anisotropy = np.zeros(kwargs["levels"], dtype=np.double)
        for l in range(kwargs["levels"]):
            chichan = np.real(
                crep.ConvertRepresentations(channel[l, :, :], "choi", "chi")
            )
            probs = np.diag(chichan)
            if (probs[1] > atol) and (probs[3] > atol):
                anisotropy[l] = probs[2] / probs[1] + probs[2] / probs[3]
    return anisotropy


########################################################################################


def Filter(process, metric, lower, upper):
    # Test if a channel (described by the process matrix) passes across a filter or not.
    metVal = eval(Metrics[metric]["func"])(
        crep.ConvertRepresentations(process, "process", "choi"),
        {"rep": "process", "channel": "unknown"},
    )
    if (metVal >= lower) and (metVal <= upper):
        return 1
    return 0


def GenCalibrationData(chname, channels, noiserates, metrics):
    # Given a set of channels and metrics, compute each metric for every channel and save the result as a 2D array.
    # The input array "channels" has size (number of channels) x 4 x 4, where A[i, :, :] is generated from m free variables.
    # The output array is a 2D array of size (number of channels) x (1 + number of free variables in the channel + number of metrics).
    # output[0, 0] = number of free variables in every channel
    # {output[i + 1, 0], ..., output[i + 1, m - 1]} = values for the free variables used to specify the i-th channel
    # {output[i + 1, m], ..., output[i + 1, m + n - 1]} = values for the n metrics on the i-th channel.
    if not (os.path.exists("./../temp")):
        os.system("mkdir -p ./../temp")
    for m in range(len(metrics)):
        calibdata = np.zeros(
            (channels.shape[0], 1 + noiserates.shape[1]), dtype=np.longdouble
        )
        for i in range(channels.shape[0]):
            calibdata[i, : noiserates.shape[1]] = noiserates[i, :]
            calibdata[i, noiserates.shape[1]] = eval(Metrics[metrics[m]]["func"])(
                channels[i, :, :], chname
            )
        # Save the calibration data
        np.savetxt(fn.CalibrationData(chname, metrics[m]), calibdata)
    return None


def ComputeNorms(channel, metrics, kwargs):
    # Compute a set of metrics for a channel (in the choi matrix form) and return the metric values
    # print("Function ComputeNorms(\n%s,\n%s)" % (np.array_str(channel, max_line_width = 150, precision = 3), metrics))
    mets = np.zeros(len(metrics), dtype=np.longdouble)
    for m in range(len(metrics)):
        mets[m] = eval(Metrics[metrics[m]]["func"])(channel, kwargs)
    return mets


def ChannelMetrics(submit, metrics, start, end, results, rep, chtype):
    # Compute the various metrics for all channels with a given noise rate
    nlevels = submit.levels + 1
    for i in range(start, end):
        if chtype == "physical":
            (folder, fname) = os.path.split(
                fn.PhysicalChannel(submit, submit.available[i, :-1])
            )
            if submit.iscorr == 0:
                chan = np.load("%s/%s" % (folder, fname))[
                    int(submit.available[i, -1]), :
                ].reshape(4, 4)
                # print("Channel %d: Function ComputeNorms(\n%s,\n%s)" % (i, np.array_str(physical, max_line_width = 150, precision = 3), metrics))
                if not (rep == "choi"):
                    chan = crep.ConvertRepresentations(chan, "process", "choi")
            elif submit.iscorr == 1:
                chan = np.load("%s/raw_%s" % (folder, fname))[
                    int(submit.available[i, -1]), :
                ]
            else:
                chan = np.load("%s/%s" % (folder, fname))[
                    int(submit.available[i, -1]), :
                ]
        else:
            lchans = np.load(
                fn.LogicalChannel(
                    submit, submit.available[i, :-1], submit.available[i, -1]
                )
            )
            chan = np.zeros(
                (lchans.shape[0], lchans.shape[1], lchans.shape[2]), dtype=np.complex128
            )
            for l in range(chan.shape[0]):
                chan[l, :, :] = crep.ConvertRepresentations(
                    lchans[l, :, :], "process", "choi"
                )
        for m in range(len(metrics)):
            if chtype == "physical":
                results[i * len(metrics) + m] = eval(Metrics[metrics[m]]["func"])(
                    chan,
                    {
                        "qcode": submit.eccs[0],
                        "levels": nlevels,
                        "corr": submit.iscorr,
                        "channel": submit.channel,
                        "chtype": chtype,
                        "rep": "choi",
                    },
                )
            else:
                results[
                    (i * len(metrics) * nlevels + m * nlevels) : (
                        i * len(metrics) * nlevels + (m + 1) * nlevels
                    )
                ] = eval(Metrics[metrics[m]]["func"])(
                    chan,
                    {
                        "qcode": submit.eccs[0],
                        "levels": nlevels,
                        "corr": 0,
                        "channel": submit.channel,
                        "chtype": chtype,
                        "rep": "choi",
                    },
                )
            # print("%g" % (results[i * len(physmetrics) + m]))
    return None


def ComputeMetrics(submit, metrics, chtype="physical"):
    # Compute metrics for all physical channels in a submission.
    ncpu = 1
    nproc = min(ncpu, mp.cpu_count())
    chunk = int(np.ceil(submit.channels / np.float(nproc)))
    processes = []
    if chtype == "physical":
        results = mp.Array(ct.c_longdouble, submit.channels * len(metrics))
    else:
        results = mp.Array(
            ct.c_longdouble, submit.channels * len(metrics) * (submit.levels + 1)
        )
    for i in range(nproc):
        processes.append(
            mp.Process(
                target=ChannelMetrics,
                args=(
                    submit,
                    metrics,
                    i * chunk,
                    min(submit.channels, (i + 1) * chunk),
                    results,
                    "process",
                    chtype,
                ),
            )
        )
    for i in range(nproc):
        processes[i].start()
    for i in range(nproc):
        processes[i].join()

    if chtype == "physical":
        metvals = np.reshape(results, [submit.channels, len(metrics)], order="c")
        # print("Metric values for metric = %s\n%s" % (metrics, np.array_str(metvals)))
        # Write the physical metrics on to a file
        for m in range(len(metrics)):
            np.save(fn.PhysicalErrorRates(submit, metrics[m]), metvals[:, m])
    else:
        metvals = np.reshape(
            results, [submit.channels, len(metrics), submit.levels + 1], order="c"
        )
        # Write the logical metrics on to a file
        for i in range(submit.channels):
            for m in range(len(metrics)):
                fname = fn.LogicalErrorRate(
                    submit,
                    submit.available[i, :-1],
                    submit.available[i, -1],
                    metrics[m],
                    average=1,
                )
                np.save(fname, metvals[i, m, :])
    return None


def PlotCalibrationData1D(chname, metrics, xcol=0):
    # The calibration data for every metric is an array of size m that contains a metric value for every channel.
    # "noiserates" is a 2D array of size (number of channels) x m that contains the parameter combinations corresponding to every channel index.
    # "xcol" indicates the free variable of the noise model that distinguishes various channels in the plot. (This is the X-axis.)
    # All metrics will be in the same plot.
    fig = plt.figure(figsize=(gv.canvas_size[0] * 1.2, gv.canvas_size[1] * 1.2))
    plt.title("%s" % (qc.Channels[chname]["name"]), fontsize=gv.title_fontsize, y=1.01)
    for m in range(len(metrics)):
        calibdata = np.loadtxt(fn.CalibrationData(chname, metrics[m]))
        # print("calibdata\n%s" % (np.array_str(calibdata)))
        plt.plot(
            calibdata[:, xcol],
            calibdata[:, -1],
            label=Metrics[metrics[m]]["latex"],
            marker=Metrics[metrics[m]]["marker"],
            color=Metrics[metrics[m]]["color"],
            markersize=gv.marker_size + 5,
            linestyle="-",
            linewidth=7.0,
        )
    ax = plt.gca()
    ax.set_xlabel(
        qc.Channels[chname]["latex"][xcol], fontsize=gv.axes_labels_fontsize + 20
    )
    # ax.set_xscale('log')
    ax.set_ylabel("$\\mathcal{N}_{0}$", fontsize=gv.axes_labels_fontsize + 20)
    # ax.set_yscale('log')
    # separate the axes labels from the plot-frame
    ax.tick_params(
        axis="both",
        direction="inout",
        which="both",
        pad=gv.ticks_pad,
        labelsize=gv.ticks_fontsize + 10,
        length=gv.ticks_length,
        width=gv.ticks_width,
    )
    # Legend
    plt.legend(
        numpoints=1,
        loc=4,
        shadow=True,
        fontsize=gv.legend_fontsize,
        markerscale=gv.legend_marker_scale,
    )
    # Save the plot
    plt.savefig(fn.CalibrationPlot(chname, "_".join(metrics)))
    plt.close()
    return None


def PlotCalibrationData2D(chname, metrics, xcol=0, ycol=1):
    # Plot performance contours for various noise strength values, with repect to the physical noise parameters.
    plotfname = fn.CalibrationPlot(chname, "_".join(metrics))

    with PdfPages(plotfname) as pdf:
        for m in range(len(metrics)):
            calibdata = np.loadtxt(fn.CalibrationData(chname, metrics[m]))
            (meshX, meshY) = np.meshgrid(
                np.linspace(
                    calibdata[:, xcol].min(),
                    calibdata[:, xcol].max(),
                    max(10, calibdata.shape[0]),
                ),
                np.linspace(
                    calibdata[:, ycol].min(),
                    calibdata[:, ycol].max(),
                    max(10, calibdata.shape[0]),
                ),
            )
            meshZ = griddata(
                (calibdata[:, xcol], calibdata[:, ycol]),
                calibdata[:, -1],
                (meshX, meshY),
                method="cubic",
            )
            # Contour Plot
            fig = plt.figure(figsize=gv.canvas_size)
            # Data points
            cplot = plt.contourf(
                meshX,
                meshY,
                meshZ,
                cmap=cm.winter,
                locator=ticker.LogLocator(),
                linestyles=gv.contour_linestyle,
            )
            plt.scatter(calibdata[:, xcol], calibdata[:, ycol], marker="o", color="k")
            plt.title("%s channel" % (chname), fontsize=gv.title_fontsize, y=1.03)
            ax = plt.gca()
            ax.set_xlabel(
                qc.Channels[chname]["latex"][xcol], fontsize=gv.axes_labels_fontsize
            )
            ax.set_ylabel(
                qc.Channels[chname]["latex"][ycol], fontsize=gv.axes_labels_fontsize
            )
            ax.tick_params(
                axis="both",
                which="both",
                pad=gv.ticks_pad,
                direction="inout",
                length=gv.ticks_length,
                width=gv.ticks_width,
                labelsize=gv.ticks_fontsize,
            )
            # Legend
            cbar = plt.colorbar(
                cplot, extend="both", spacing="proportional", drawedges=False
            )
            cbar.ax.set_xlabel(
                Metrics[metrics[m]]["latex"], fontsize=gv.colorbar_fontsize
            )
            cbar.ax.tick_params(
                labelsize=gv.legend_fontsize,
                pad=gv.ticks_pad,
                length=gv.ticks_length,
                width=gv.ticks_width,
            )
            cbar.ax.xaxis.labelpad = gv.ticks_pad
            # Save the plot
            pdf.savefig(fig)
            plt.close()
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = "%s for %d %s channels." % (
            ",".join(metrics),
            calibdata.shape[0],
            chname,
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None
