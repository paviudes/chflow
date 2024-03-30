import numpy as np
from define import qcode as qc
from define.heuristic import AssignErrorProbs

if __name__ == '__main__':
	r'''
	Code to test the splitting method.
	We will test it using the 5 qubit code.
	Let us assume that NR data comprises of the following errors and their probabilities.
	errors: I, X_5, X_3, X_1, Y_2, Y_4, Y_2 Y_5, Z_2 Z_4, Z_1, Z_3.
	probabilities: 0.9, 0.00264285, 0.02541568, 0.04758009, 0.0161835 , 0.00522989, 0.04834212, 0.02560945, 0.01376481, 0.03177657, 0.02972115
	'''
	
	qecc = qc.QuantumErrorCorrectingCode("5qc")
	qc.Load(qecc)
	qc.PrepareSyndromeLookUp(qecc)

	known_pauli_errors = np.array([

			[0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1],
			[0, 0, 1, 0, 0],
			[1, 0, 0, 0, 0],
			[0, 2, 0, 0, 0],
			[0, 0, 0, 2, 0],
			[0, 2, 0, 0, 2],
			[0, 3, 0, 3, 0],
			[3, 0, 0, 0, 0],
			[0, 0, 3, 0, 0]

		], dtype = int)
	known_pauli_indices = np.array([qecc.GetPositionInLST(known_pauli_errors[p, :]) for p in range(known_pauli_errors.shape[0])], dtype = int)
	known_probs = np.array([0.9, 0.00264285, 0.02541568, 0.04758009, 0.0161835 , 0.00522989, 0.04834212, 0.02560945, 0.01376481, 0.03177657, 0.02972115], dtype = np.double)

	pauli_errors = qecc.PauliOperatorsLST

	pauli_probs = AssignErrorProbs(known_pauli_indices, known_probs, pauli_errors)

	print("Errors and their probabilities")
	for w in qecc.group_by_weight:
		print("===========\nWeight {} errors".format(w))
		for p in range(qecc.group_by_weight[w].size):
			error_index = qecc.group_by_weight[w][p]
			print("Prob( {} ) = {}".format(pauli_errors[error_index], pauli_probs[error_index]))