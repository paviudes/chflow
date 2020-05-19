import time

# import timeout_decorator
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


############### Diamond norm using .
def DiamondNorm(choi, idnchan):
    """
    Compute the Diamond norm using functions in CVXOPT.

    The Diamond distance is defined as follows.
    Maximize:
        Tr(J' . W) + Tr(W' . J),
    where X' is the conjugate transpose of X.
    Subject to the constraints:
        W >= 0
        I o R - W >= 0
        R >= 0
        Tr(R) = 1
    where
        R is a density matrices.
        W is a linear operator whose dimension is that of J.
    We will also use the property that for a complex matrix X = Xr + i Xi:
    X >> 0 if and only if
    [Xr    Xi]
    [-Xi   Xr]
    >>
    0
    """
    sdim = choi.shape[0]
    nqubits = int(math.log(sdim, 4))
    qdim = 2 ** nqubits
    #### Constants and Variables
    I = cvx.Constant(np.eye(qdim))

    Jr = cvx.Constant(value=np.real(choi - idnchan))
    Ji = cvx.Constant(value=np.imag(choi - idnchan))

    Rr = cvx.Variable(name="Rr", shape=(qdim, qdim), symmetric=True)
    Ri = cvx.Variable(name="Ri", shape=(qdim, qdim))

    Wr = cvx.Variable(name="Wr", shape=(sdim, sdim))
    Wi = cvx.Variable(name="Wi", shape=(sdim, sdim))

    #### Constraints
    constraints = []

    # R is a density matrix
    constraints.append(cvx.bmat([[Rr, -1 * Ri], [Ri, Rr]]) >> 0)
    constraints.append(cvx.trace(Rr) == 1)
    constraints.append(Ri == -1 * Ri.T)
    constraints.append(cvx.trace(Ri) == 0)

    # W >> 0
    constraints.append(cvx.bmat([[Wr, -Wi], [Wi, Wr]]) >> 0)

    # I o R - W >= 0
    constraints.append(
        cvx.bmat(
            [
                [cvx.kron(I, Rr) - Wr, -cvx.kron(I, Ri) + Wi],
                [cvx.kron(I, Ri) - Wi, cvx.kron(I, Rr) - Wr],
            ]
        )
        >> 0
    )

    #### Objective
    obj = cvx.Maximize(
        cvx.trace(Jr @ Wr)
        - cvx.trace(Ji @ Wi)
        + cvx.trace(Wr.T @ Jr)
        + cvx.trace(Wi.T @ Ji)
    )

    #### Setting up the problem
    prob = cvx.Problem(obj, constraints=constraints)
    # print("Problem\n{}".format(prob))

    #### Solve and print the solution
    prob.solve(solver="SCS", parallel=True, verbose=False)
    dnorm = obj.value
    # print("Diamond norm from CVXPY = {}.".format(dnorm))
    return dnorm


############### Diamond norm using .
def DiamondNormSimpler(choi, idnchan):
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
    obj = cvx.Maximize(cvx.trace(Jr.T @ Xr) + cvx.trace(Ji.T @ Xi))

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


def KrausToChoi(kraus):
    r"""
    Convert from the Kraus representation of a quantum channel to its Choi matrix.

    The Choi matrix of a quantum channel E, denoted by J(E), is
    J(E) = \sum_{ij}E(B_ij) o B_ij
    where
        {B_ij} form a computational basis for n x n matrices.
        B_ij = |b(i)><b(j)| in the Drac notation, where
            b(n) is the binary encoding of the integer n.
        Note: B_ij is a matrix, all of whose entries are 0, except for the i,j entry, which is 1.
    Note that E(B_ij) can be simplified as
    E(B_ij) = \sum_k K_k B_ij (K_k)^\dag
    where {K_k} are the Kraus operators. Using orthogonality of the basis vectors, we find
    E(B_ij) = \sum_k \sum_lm \sum_pq (K_k)_lm ((K_k)_qp)* |l><m| |i><j| |p><q|
            = \sum_k \sum_lq (K_k)_li ((K_k)_qj)* |l><q|
            = \sum_k [\sum_l (K_k)_li |l>] [\sum_l <q|(K_k)*_qj]
            = \sum_k [\sum_l (K_k)_li |l>] [\sum_l (K_k)_jq |q>]^\dag
            = \sum_k [(K_k)_(:,i)] . [(K_k)_(j,:)]^\dag
    where (:,i) denotes the i-th column and (j,:), the j-the row.
    """
    nqubits = int(math.log(kraus.shape[1], 2))
    hdim = 2 ** nqubits
    # Constructing the computational basis
    compbasis = np.zeros((2 ** (nqubits * 2), hdim, hdim), dtype=np.double)
    for i in range(hdim):
        for j in range(hdim):  # This for loop can be converted to numpy broadcasting.
            compbasis[i * hdim + j, i, j] = 1
    # print("computational basis\n{}".format(compbasis))
    # Applying the channel on the basis vector.
    choi = np.zeros((4 ** nqubits, 4 ** nqubits), dtype=np.complex)
    for i in range(hdim):
        for j in range(hdim):
            for k in range(kraus.shape[0]):
                col_i = kraus[k, :, i]
                col_j = kraus[k, :, j]
                choi = choi + np.kron(
                    np.dot(col_i[:, np.newaxis], col_j[np.newaxis, :].conj()),
                    compbasis[i * hdim + j],
                )
    choi = choi / np.trace(choi)
    # print("choi: {}".format(np.trace(choi)))
    return choi


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
    # print("Choi matrix of the composite channel\n{}".format(np.trace(choi)))
    return choi


def VerifyChan(inpchans, chan_from="chflow", quiet=0):
    """
    Verify properties of a Choi matrix.
    """
    # print("inpchans = {}".format(inpchans))
    atol = 10e-10
    if chan_from == "qutip":
        chans_choi = [J.full() for J in inpchans]
    else:
        chans_choi = [J for J in inpchans]

    # Write voilations on to a file
    if quiet == 1:
        chanlog = "./../../temp/bad_chans.txt"
        bc = open(chanlog, "a")
        isbad = 0

    for q in range(len(chans_choi)):
        if quiet == 0:
            print("Channel {}\n{}".format(q + 1, np.round(chans_choi[q], 3)))
        isherm = np.linalg.norm(chans_choi[q] - chans_choi[q].T.conj())
        isuntr = np.trace(chans_choi[q])
        eigvals = np.linalg.eigvals(chans_choi[q])
        if quiet == 0:
            print(
                "||R - R^\dag|| = {}\nTrace(R) = {}\n(Negative) Eigenvalues = {}".format(
                    np.round(isherm, 3),
                    np.round(isuntr, 3),
                    np.round(eigvals[np.abs(eigvals) <= -atol], 5),
                )
            )
            print("")
        if quiet == 1:
            if (
                (isherm >= atol)
                or (np.abs(isuntr - 1) >= atol)
                or (np.any(eigvals <= -atol))
            ):
                isbad = 1
                print(
                    "\033[2mFound a bad channel, saving to {}.\033[0m".format(chanlog)
                )
                bc.write("Channel {}\n{}\n".format(q + 1, np.round(chans_choi[q], 5)))
                bc.write("||R - R^\\dag|| = {}\n".format(np.round(isherm, 5)))
                bc.write("Trace(R) = {}\n".format(np.round(isuntr, 5)))
                bc.write(
                    "(Negative) Eigenvalues = {}\n".format(
                        np.round(eigvals[np.abs(eigvals) < -atol], 5)
                    )
                )
                bc.write("-------\n")
    if quiet == 1:
        if isbad == 1:
            bc.write("xxxxxxxxxxxxx\n\n")
        bc.close()
    return None


# @timeout_decorator.timeout(60, timeout_exception=StopIteration)
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
    qutip_dist = dnorm(chan, idchan)
    runtime = time.time() - start
    if quiet == 0:
        print(
            "Time = {} seconds, dnorm = {}.".format(
                round(runtime, 2), round(qutip_dist, 3)
            )
        )
    return (qutip_dist, runtime)


# @timeout_decorator.timeout(600, timeout_exception=StopIteration)
def DNorm_chflow(inpchans, refchans, quiet=0, chan_from="chflow"):
    """
    Compute the Diamond distance between a pair of channels, using chflow.
    Each of the channels in the pair are specified as a tensor product of a set of channels.
    """
    # nqubits = len(inpchans)
    # print("chanel in dnorm chflow is from {}".format(chan_from))
    if quiet == 0:
        print("Computing using chflow ...")
    start = time.time()
    if chan_from == "qutip":
        chan_choi_mats = [to_choi(rc).full() for rc in inpchans]
        inpchans_rvec = [ColumnVecToRowVec(J) for J in chan_choi_mats]
        ref_choi_mats = [to_choi(rc).full() for rc in refchans]
        refchans_rvec = [ColumnVecToRowVec(J) for J in ref_choi_mats]
    else:
        inpchans_rvec = [J for J in inpchans]
        refchans_rvec = [J for J in refchans]

    # print("chan_from = {}\nChannels:\n{}".format(chan_from, inpchans_rvec))

    inpchan = MultiQubitChoi(*inpchans_rvec)
    refchan = MultiQubitChoi(*refchans_rvec)

    # print("~~~~~~~~~~~~~~~~~~~")
    # print("Verifying the composite channel:")
    VerifyChan([inpchan], chan_from=chan_from, quiet=1)
    # print("~~~~~~~~~~~~~~~~~~~")

    chflow_dist = DiamondNormSimpler(inpchan, refchan)
    # chflow_dist = 0 ### Only for debudding, temporarily turning off dnorm computation.
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
    # print("Trace of channels: {}".format([np.trace(J) for J in inpchans]))

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
    print(
        "\033[2mSum of diamond distances = {}\n{}.\033[0m".format(
            round(dist_sum, 3), [round(individual_info[i][0], 3) for i in range(nchans)]
        )
    )

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
        "Difference in the diamond distances: sum - tensor_product = {}. ({}%)\n\033[2mSpeedup in computing the sum = {}.\033[0m".format(
            round(difference, 3),
            round(difference * 100 / composite_dnorm, 3),
            round(runtime_composite / runtime_individuals, 3),
        )
    )
    return difference


def CompareDNormMethods(inpchans, refchans, chan_from="chflow"):
    """
    Compare different methods of computing diamond norm: using qutip and using chflow.
    """
    # qutip diamond norm
    # print("inpchans = {}, chan_from={}".format(inpchans, chan_from))
    try:
        (qutip_dist, runtime_qutip) = DNorm_qutip(inpchans, refchans, quiet=0)
        qutip_dist = qutip_dist
    except StopIteration:
        (qutip_dist, runtime_qutip) = (-1, 60)
    # chflow diamond norm
    start = time.time()
    try:
        (chflow_dist, runtime_chflow) = DNorm_chflow(
            inpchans, refchans, quiet=0, chan_from=chan_from
        )
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


def GetChansFromChflow(chname, nqubits, *params):
    """
    Generate CPTP maps using chflow.
    """

    if chname == "rtasu":
        rand_kraus = GetKraussForChannel(chname, nqubits, *params)
    else:
        rand_kraus = np.tile(GetKraussForChannel(chname, *params), [nqubits, 1, 1, 1])
    rand_chois = [KrausToChoi(ch) for ch in rand_kraus]
    iden_kraus = np.tile(GetKraussForChannel("id"), [nqubits, 1, 1, 1])
    iden_chois = [KrausToChoi(ch) for ch in iden_kraus]
    VerifyChan(rand_chois, chan_from="chflow", quiet=0)
    return (rand_chois, iden_chois)


def GetChansFromQutip(chname, nqubits, *params):
    """
    Generate CPTP maps using qutip.
    """
    rand_kraus = [rand_super(2) for __ in range(nqubits)]
    rand_chois = [to_choi(ch) for ch in rand_kraus]
    iden_kraus = [to_super(Qobj(np.eye(2))) for __ in range(nqubits)]
    VerifyChan(rand_chois, chan_from="qutip", quiet=0)
    return (rand_kraus, iden_kraus)


###############################
if __name__ == "__main__":
    # Matching the outputs of dnorm in chflow and qutip.
    chanlog = "./../../temp/bad_chans.txt"
    with open(chanlog, "a") as bc:
        bc.write("\n\nLogging run at %s.\n\n" % (time.strftime("%d/%m/%Y %H:%M:%S")))

    test_options = ["subadd", "compare"]
    test_mode = "subadd"
    ntrials = 10
    nqubits = 4
    for t in range(1, 1 + ntrials):
        print("Trial: {}".format(t))
        (rand_chans, iden_chans) = GetChansFromChflow("rand", nqubits, t * 0.01)
        if test_mode == "compare":
            CompareDNormMethods(rand_chans, iden_chans, chan_from="qutip")
        elif test_mode == "subadd":
            VerifySubAdditivity(
                rand_chans, iden_chans, method="chflow", chan_from="chflow"
            )
        else:
            pass
        print("xxxxxxxxxxx")
    with open(chanlog, "a") as bc:
        bc.write("%%%%%%%%%%%%%% End of log.")
