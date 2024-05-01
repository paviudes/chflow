import numpy as np
from define.QECCLfid.utils import extend_operator

if __name__ == '__main__':
	M = np.random.rand(4, 4) + 1j
	H = M * M.conj().T
	support_qubits = np.array([0, 2], dtype = int)
	nqubits = 4
	print("Operator\n{}\nsupported on {}".format(H, support_qubits))
	extended_operator = extend_operator(support_qubits, H, nqubits)
	print("Extended Operator on {} qubits is\n{}".format(nqubits, extended_operator))