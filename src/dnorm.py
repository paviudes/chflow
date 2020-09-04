import numpy as np
import cvxpy as cvx

def DiamondNorm(choi, idnchan=None):
	"""
	Compute the Diamond distance using SDP solver from the CVXPY package.
	
	The inputs are:
		1. choi -- the Choi-Jamilowski matrix of the input channel, J. Note, Tr(J) should be 1.
		2. idnchan -- the Choi-Jamilowski matrix of a reference channel from which the Diamond distance needs to be computed.

	The program follows the precription in https://arxiv.org/abs/1207.5726.

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

	if idnchan == None:
		idnchan = np.zeros_like(choi)

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


if __name__ == '__main__':
	# Testing the Diamond norm function.
	choi = np.array([[1/2,0,0,1/2],[0,0,0,0],[0,0,0,0],[1/2,0,0,1/2]], dtype=np.complex128)
	dnorm = DiamondNorm(choi)
	print("The Diamond norm of\n{}\nis {}.".format(choi, dnorm))