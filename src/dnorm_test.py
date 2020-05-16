import time
import timeout_decorator
import math
from functools import reduce
import numpy as np
from qutip import (
    Qobj,
    rand_super,
    to_choi,
    to_super,
    composite,
    dnorm,
    rz,
    rand_super_bcsz,
)
import cvxpy as cvx
from define.chanreps import ConvertRepresentations
from define.chandefs import GetKraussForChannel


def Kron(*mats):
    """
    Kronecker product of a list of matrices.
    """
    if len(mats) <= 1:
        return mats[0]
    return np.kron(mats[0], Kron(*mats[1:]))


############### Function to be tested.
def DiamondNorm(choi, idnchan):
    """
    Compute the Diamond norm using functions in CVXOPT.

    The Diamond distance is defined as follows.
    Maximize:
        1/2 * (J | X) + 1/2 * (J' | X'),
    where M' is the Hermitian conjugate of M.
    Subject to the constraints: M =
        [I o R1      X       ]
        [X'         I o R2   ]
        >>
        0
    where
        R1 and R2 are density matrices, i.e., R1 >> 0, R2 >> 0 and trace(R1) == 1, trace(R2) == 1.
        X is a linear operator whose dimension is that of J.
    We will also use the property that for a complex matrix X = Xr + i Xi:
    X >> 0 if and only if
    [Xr    Xi]
    [-Xi   Xr]
    >>
    0
    """
    # single_qubit_identity = np.zeros((4, 4), dtype=np.double)
    # single_qubit_identity[0, 0] = 0.5
    # single_qubit_identity[0, 3] = 0.5
    # single_qubit_identity[3, 0] = 0.5
    # single_qubit_identity[3, 3] = 0.5
    # idnchan = Kron(*[single_qubit_identity for __ in range(nqubits)])

    nqubits = int(math.log(choi.shape[0], 4))
    #### Constants and Variables
    I = cvx.Constant(np.eye(2 ** nqubits))
    Z = cvx.Constant(np.zeros((2 ** nqubits, 2 ** nqubits)))

    Jr = cvx.Constant(value=np.real(choi - idnchan))
    Ji = cvx.Constant(value=np.imag(choi - idnchan))

    Xr = cvx.Variable(name="Xr", shape=(4 ** nqubits, 4 ** nqubits))
    Xi = cvx.Variable(name="Xi", shape=(4 ** nqubits, 4 ** nqubits))

    R1r = cvx.Variable(name="R1r", shape=(2 ** nqubits, 2 ** nqubits), symmetric=True)
    R1i = cvx.Variable(name="R1i", shape=(2 ** nqubits, 2 ** nqubits))

    R2r = cvx.Variable(name="R2r", shape=(2 ** nqubits, 2 ** nqubits), symmetric=True)
    R2i = cvx.Variable(name="R2i", shape=(2 ** nqubits, 2 ** nqubits))

    #### Constraints
    constraints = []

    # R1 is a density matrix
    constraints.append(cvx.bmat([[R1r, -1 * R1i], [R1i, R1r]]) >> 0)
    constraints.append(cvx.trace(R1r) == 1)
    constraints.append(R1i == -1 * R1i.T)
    constraints.append(cvx.trace(R1i) == 0)

    # R1 is a density matrix
    constraints.append(cvx.bmat([[R2r, -1 * R2i], [R2i, R2r]]) >> 0)
    constraints.append(cvx.trace(R2r) == 1)
    constraints.append(R2i == -1 * R2i.T)
    constraints.append(cvx.trace(R2i) == 0)

    constraints.append(
        cvx.bmat(
            [
                [cvx.kron(I, R1r), Xr, -1 * cvx.kron(I, R1i), -1 * Xi],
                [Xr.T, cvx.kron(I, R2r), Xi.T, -1 * cvx.kron(I, R2i)],
                [cvx.kron(I, R1i), Xi, cvx.kron(I, R1r), Xr],
                [-1 * Xi.T, cvx.kron(I, R2i), Xr.T, cvx.kron(I, R1r)],
            ]
        )
        >> 0
    )

    #### Objective
    obj = cvx.Maximize(cvx.trace(Jr @ Xr) + cvx.trace(Ji @ Xi))

    #### Setting up the problem
    prob = cvx.Problem(obj, constraints=constraints)
    # print("Problem\n{}".format(prob))
    #### Solve and print the solution
    prob.solve(solver="SCS", parallel=True, verbose=False)
    dnorm = obj.value
    # print("Diamond norm from CVXPY = {}.".format(dnorm))
    return dnorm


def ColumnVecToRowVec(colvec):
    """
    Change from a column vectorization to a row vectorization
    """
    n = int(math.log(colvec.shape[0], 2))
    tensor_form = colvec.reshape([2] * (2 * n))
    transpose_map = (
        list(range(n // 2, n))
        + list(range(0, n // 2))
        + list(range(3 * n // 2, 2 * n))
        + list(range(n, 3 * n // 2))
    )
    # print("Transpose map = {}".format(transpose_map))
    rowvec = np.reshape(np.transpose(tensor_form, transpose_map), colvec.shape)
    return rowvec


def MultiQubitChoi(*single_qubit_choi):
    """
    Compute the Choi matrix of a multi-qubit channel.
    Re-ordering of qubits = [(i, n+i) for i in range(n)]
    Re-ordering of rows = [(i, n+i) for i in range(n)]
    Re-ordering of columns = 2*n + [(i, n+i) for i in range(n)]
    Total ordering = row ordering + column ordering
    """
    n = len(single_qubit_choi)
    unordered = Kron(*single_qubit_choi).reshape([2] * 4 * n)
    ordering = np.zeros((2 * n, 2), dtype=np.int)
    # Row ordering
    ordering[:n, 0] = np.arange(n, dtype=np.int)
    ordering[:n, 1] = ordering[:n, 0] + n
    # Column ordering
    ordering[n:, 0] = 2 * n + np.arange(n, dtype=np.int)
    ordering[n:, 1] = ordering[n:, 0] + n
    # Transpose according to the ordering
    # print(
    #     "Shape of the unordered Choi: {}\nTranspose map: {}.".format(
    #         unordered.shape, ordering.ravel()
    #     )
    # )
    ordered = np.transpose(unordered, np.ravel(ordering))
    choi = np.reshape(ordered, [4 ** n, 4 ** n])
    return choi


def VerifyChan(*chans_choi):
    """
    Verify properties of a Choi matrix.
    """
    for q in range(len(chans_choi)):
        print("Channel {}\n{}".format(q + 1, np.round(chans_choi[q], 3)))
        isherm = np.round(np.linalg.norm(chans_choi[q] - chans_choi[q].T.conj()), 2)
        isuntr = np.round(np.trace(chans_choi[q]), 2)
        eigvals = np.round(np.linalg.eigvals(chans_choi[q]), 2)
        print(
            "||R - R^\dag|| = {}\nTrace(R) = {}\nEigenvalues = {}".format(
                isherm, isuntr, eigvals
            )
        )
        print("")
    return None


@timeout_decorator.timeout(60, timeout_exception=StopIteration)
def DNorm_qutip(inpchans, refchans, quiet=0):
    """
    Compute the Diamond distance between a pair of channels, using qutip.
    Each of the channels in the pair are specified as a tensor product of a set of channels.
    """
    nqubits = len(inpchans)
    if quiet == 0:
        print("Computing using qutip ...")
    start = time.time()
    chan = reduce(composite, inpchans)
    idchan = reduce(composite, refchans)
    qutip_dist = dnorm(chan, idchan) / 2 ** nqubits
    runtime = time.time() - start
    if quiet == 0:
        print(
            "Time = {} seconds, dnorm = {}.".format(
                round(runtime, 2), round(qutip_dist, 3)
            )
        )
    return (qutip_dist, runtime)


@timeout_decorator.timeout(600, timeout_exception=StopIteration)
def DNorm_chflow(inpchans, refchans, quiet=0, chan_from="chflow"):
    """
    Compute the Diamond distance between a pair of channels, using chflow.
    Each of the channels in the pair are specified as a tensor product of a set of channels.
    """
    if quiet == 0:
        print("Computing using chflow ...")
    start = time.time()
    if chan_from == "qutip":
        chan_choi_mats = [to_choi(rc).full() for rc in inpchans]
        inpchans_rvec = [ColumnVecToRowVec(J / np.trace(J)) for J in chan_choi_mats]
    else:
        inpchans_rvec = inpchans
    ref_choi_mats = [to_choi(rc).full() for rc in refchans]
    refchans_rvec = [ColumnVecToRowVec(J / np.trace(J)) for J in ref_choi_mats]

    inpchan = MultiQubitChoi(*inpchans_rvec)
    refchan = MultiQubitChoi(*refchans_rvec)
    chflow_dist = DiamondNorm(inpchan, refchan)
    runtime = time.time() - start
    if quiet == 0:
        print(
            "Time = {} seconds, dnorm = {}.".format(
                round(runtime, 2), round(chflow_dist, 3)
            )
        )
    return (chflow_dist, runtime)


def VerifySubAdditivity(inpchans, refchans, method="chflow", chan_from="chflow"):
    """
    Verify the sub-additivity property of Diamond distance.
    We want to know by how much is the sum of diamond norms of channels greater than the diamond norm of its tensor product.
    """
    nchans = len(inpchans)

    # Computing the diamond norms of the individual maps.
    print(
        "\033[2mComputing individual diamond distances using {} ...\033[0m".format(
            method
        )
    )
    if method == "chflow":
        individual_info = np.array(
            [
                DNorm_chflow([inpchans[i]], [refchans[i]], quiet=1, chan_from=chan_from)
                for i in range(nchans)
            ],
            dtype=np.double,
        )
    elif method == "qutip":
        individual_info = np.array(
            [DNorm_qutip([inpchans[i]], [refchans[i]], quiet=1) for i in range(nchans)],
            dtype=np.double,
        )
    else:
        pass
    runtime_individuals = np.sum([individual_info[i][1] for i in range(nchans)])
    print("\033[2mDone in {} seconds.\033[0m".format(round(runtime_individuals, 2)))
    dist_sum = np.sum([individual_info[i][0] for i in range(nchans)])
    print("\033[2mSum of diamond distances = {}.\033[0m".format(round(dist_sum, 3)))

    # Computing the diamond norms of the composite map.
    print(
        "\033[2mComputing collective diamond distance using {} ...\033[0m".format(
            method
        )
    )
    if method == "chflow":
        (composite_dnorm, runtime_composite) = DNorm_chflow(
            inpchans, refchans, quiet=1, chan_from=chan_from
        )
    elif method == "qutip":
        (composite_dnorm, runtime_composite) = DNorm_qutip(inpchans, refchans, quiet=1)
    else:
        pass
    print("\033[2mDone in {} seconds.\033[0m".format(round(runtime_composite, 2)))
    print(
        "\033[2mDiamond distance of the tensor product map = {}.\033[0m".format(
            round(composite_dnorm, 3)
        )
    )

    # Computing the difference between the sum of diamond norms and the diamond norm of the composite.
    difference = dist_sum - composite_dnorm
    print("====")
    print(
        "Difference in the diamond distances: sum - tensor_product = {}. ({}%)\nSpeedup in computing the sum = {}.".format(
            round(difference, 3),
            round(difference / composite_dnorm, 3) * 100,
            round(runtime_composite / runtime_individuals, 3),
        )
    )
    return difference


def CompareDNormMethods(inpchans, refchans):
    """
    Compare different methods of computing diamond norm: using qutip and using chflow.
    """
    # qutip diamond norm
    try:
        (qutip_dist, runtime_qutip) = DNorm_qutip(inpchans, refchans)
        qutip_dist = qutip_dist
    except StopIteration:
        (qutip_dist, runtime_qutip) = (-1, 60)
    # chflow diamond norm
    start = time.time()
    try:
        (chflow_dist, runtime_chflow) = DNorm_chflow(inpchans, refchans)
    except StopIteration:
        (chflow_dist, runtime_chflow) = (-1, 60)
    # Summary
    print("====")
    print(
        "Difference in the diamond distances: {}, Speed-up in chflow >= {}.".format(
            round(qutip_dist - chflow_dist, 5), round(runtime_qutip / runtime_chflow, 2)
        )
    )
    return None


###############################
if __name__ == "__main__":
    # Matching the outputs of dnorm in chflow and qutip.
    ntrials = 10
    nqubits = 5
    for t in range(1, 1 + ntrials):
        print("Trial: {}".format(t))
        rand_chans = GetKraussForChannel("rtasu", nqubits, t * 0.01)
        rand_chans = [ConvertRepresentations(ch, "krauss", "choi") for ch in rand_chans]
        # VerifyChan(*rand_chans)
        iden_chans = [
            to_super(Qobj(np.eye(2, dtype=np.float))) for __ in range(nqubits)
        ]
        # CompareDNormMethods(rand_chans, iden_chans)
        VerifySubAdditivity(rand_chans, iden_chans, method="chflow", chan_from="chflow")
        print("xxxxxxxxxxx")
