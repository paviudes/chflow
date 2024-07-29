import numpy as np
from define.QECCLfid.utils import extend_operator, check_hermiticity

if __name__ == '__main__':

	M = np.random.randint(0, high=15, size=(4,4))
	H = (M + M.T.conj()) / 2

	support_qubits = np.array([0, 2], dtype = int)
	nqubits = 4

	print("H\n{}\nsupported on {}".format(H, support_qubits))
	check_hermiticity(H, "|H - H^dag|")

	extended_H = extend_operator(support_qubits, H, nqubits)
	
	print("Extended Operator on {} qubits is\n{}".format(nqubits, extended_H))

	check_hermiticity(extended_H, "|H - H^dag|")