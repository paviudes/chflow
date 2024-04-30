import numpy as np
from define.QECCLfid.utils import GetNQubitPauli, PauliTensor
from define.QECCLfid.contract import ContractTensorNetwork

if __name__ == '__main__':
	# We want to test a code to compute Tr[ K . P ] where K is a Kraus operator and P is a Pauli matrix.
	PauliMats = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=np.complex128)

	nqubits = 5
	dim = np.power(2, nqubits, dtype = int)
	
	kraus = np.random.rand(dim, dim)
	kraus_support = tuple((range(nqubits)))

	pauli_index = 10
	nqubit_pauli_op = GetNQubitPauli(pauli_index, nqubits)
	# pauli_op = PauliMats[nqubit_pauli_op, :, :]
	
	network = [(kraus_support, kraus.reshape([2, 2] * nqubits))] + [((q,), PauliMats[nqubit_pauli_op[q], :, :]) for q in range(nqubits)]

	(__, result_tcon) = ContractTensorNetwork(network, end_trace=1, use_einsum=1)
	print("Chi element from tensor contraction: {}".format(result_tcon))

	pauli_mat = PauliTensor(nqubit_pauli_op).reshape(dim, dim)
	result_matmul = np.trace(np.dot(kraus, pauli_mat))
	print("Chi element from matrix multiplication: {}".format(result_matmul))