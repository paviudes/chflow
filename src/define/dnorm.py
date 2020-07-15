import math
import cvxpy as cvx


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
