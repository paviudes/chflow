import sys
import time

try:
    import numpy as np
    from scipy import optimize as opt
    import picos as pic
    import cvxopt as cvx
except:
    pass

from define import chanreps as crep


def HermitianConjugate(mat):
    # Return the Hermitian conjugate of a matrix
    return np.conjugate(np.transpose(mat))


def Twirl(channel, rep="chi"):
    # Return the process matrix of the Twirled channel.
    # Convert to the chi matrix and return its diagonal entries.
    chi = np.copy(channel)
    if not (rep == "chi"):
        chi = crep.ConvertRepresentations(channel, rep, "chi")
    proba = np.real(np.diag(chi))
    return proba


def MatrixPositivityConstraints(proba, inchan):
    ## Impose the semidefinite constraint: A - B > 0, where A - B are related to the reference and the input (arbitrary) quanum channels, respectively.
    # B = (1-M).T * (1-M) where M is the bloch rotation matrix corresponding to the input channel
    # A = (1 - M)^2 where M is the bloch rotation matrix for the Pauli channel
    # pauliKrauss = np.transpose(np.transpose(pauli, (2, 1, 0)) * proba, (2, 1, 0))
    pI = 1 - np.sum(proba, dtype=np.float)
    pauli = np.array(
        [
            [np.power(1 - pI - proba[0] + proba[1] + proba[2], 2.0), 0.0, 0.0],
            [0.0, np.power(1 - pI + proba[0] - proba[1] + proba[2], 2.0), 0.0],
            [0.0, 0.0, np.power(1 - pI + proba[0] + proba[1] - proba[2], 2.0)],
        ],
        dtype=np.float,
    )
    # The difference between the bloch error matrices must be positive semi-definite
    positiveMatrix = pauli - inchan
    evals = np.linalg.eigvals(positiveMatrix)
    return evals


def DiamondDistanceFromPauli(proba, refchan):
    ## Diamond distance between a channel and a Pauli channel with a given probability distribution
    # If the Pauli channel distribution has only three elements, the first one is calculated as 1 - (sum of all other probabilties)
    pI = 1 - np.sum(proba, dtype=np.float)
    pchoi = (
        np.array(
            [
                [pI + proba[2], 0, 0, pI - proba[2]],
                [0, proba[0] + proba[1], proba[0] - proba[1], 0],
                [0, proba[0] - proba[1], proba[0] + proba[1], 0],
                [pI - proba[2], 0, 0, pI + proba[2]],
            ],
            dtype=np.float,
        )
        * 0.5
    )
    diff = np.real(
        ((refchan - pchoi) + HermitianConjugate(refchan - pchoi)) / np.float(2)
    )
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
    return dnorm


def HonestPauliApproximation(channel, rep="process"):
    ## Find the Honest Pauli Approximation of an input channel, which is presented in the choi matrix representation
    # http://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.012324
    if not (rep == "choi"):
        choi = crep.ConvertRepresentations(channel, rep, "choi").astype(complex)
    else:
        choi = np.copy(channel).astype(complex)
    if not (rep == "process"):
        process = np.real(crep.ConvertRepresentations(channel, rep, "process")).astype(
            float
        )
    else:
        process = np.copy(channel).astype(float)

    (translation, rotation) = (process[0, 1:, np.newaxis], process[1:4, 1:4])
    # non-unital channels have an additional contribution C where
    # C = ||t||^2 + 2 * ||v|| where v = (1 - M)^T * t and t is the translation of the Bloch sphere (non-unital)
    nonunital = np.linalg.norm(translation, ord="fro") + 2 * np.linalg.norm(
        np.dot(np.transpose(np.eye(3, dtype=np.float) - rotation), translation),
        ord="fro",
    )
    blocherr = np.dot(
        np.transpose(np.eye(3, dtype=np.float) - rotation),
        (np.eye(3, dtype=np.float) - rotation),
    ) + nonunital * np.eye(3, dtype=np.float)
    # Minimize the diamond distance between the Pauli channel and the input channel
    objective = lambda pp: DiamondDistanceFromPauli(pp, choi)
    cons = [{"type": "ineq", "fun": (lambda pp: 1 - pp[0] - pp[1] - pp[2])}]
    cons.append(
        {"type": "ineq", "fun": (lambda pp: MatrixPositivityConstraints(pp, blocherr))}
    )
    initGuess = np.array([0.25, 0.25, 0.25], dtype=float)
    start = time.time()
    print("\033[2m"),
    result = opt.minimize(
        objective,
        initGuess,
        jac=None,
        constraints=cons,
        bounds=[(0, 1), (0, 1), (0, 1)],
        method="SLSQP",
        options={"disp": True, "maxiter": 5000},
    )
    print("\033[0m"),
    runtime = time.time() - start
    # Extract the solution
    proba = result.x
    proxim = result.fun
    if result.success == True:
        print(
            "\t\033[2mOptimization completed successfully in %d seconds.\033[0m"
            % (runtime)
        )
    else:
        print(
            "\t\033[2mOptimization terminated because of\n%s\nin %d seconds.\033[0m"
            % (result.message, runtime)
        )
    result.clear()
    return (proba, proxim)
