import numpy as np
from define import qcode as qc
from define.heuristic import AssignErrorProbs, BuildNRHash, prob_splitting_method
from define.randchans import CreateIIDPauli

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
			[0, 0, 0, 0, 3]

		], dtype = np.uint8)
	known_pauli_indices = np.array([qecc.GetPositionInLST(known_pauli_errors[p, :]) for p in range(known_pauli_errors.shape[0])], dtype = np.uint64)
	known_probs = np.array([0.7,
							0.00264285,
							0.02541568,
							0.04758009,
							0.0161835,
							0.00522989,
							0.04834212,
							0.02560945,
							0.01376481,
							0.03177657], dtype = np.float64)
	
	mask = np.ones(qecc.PauliOperatorsLST.shape[0], dtype=bool)
	mask[known_pauli_indices] = False
	
	total_unknown = 1 - np.sum(known_probs)
	print("Total probability of known errors: {}".format(1 - total_unknown))
	pauli_errors = qecc.PauliOperatorsLST.astype(np.uint8)

	r'''
	r = 1 - 0.9^1/5
	Testing individual errors
	1. P( Y2 Y4 Y5 ) = 0.000252823	
	'''
	nr_hash = BuildNRHash(known_pauli_indices, known_probs, pauli_errors)
	single_qubit_infid = 1 - np.power(nr_hash[0], 1/qecc.N)
	
	pauli_error = np.array([1, 2, 3, 2, 4, 2], dtype=np.uint8)
	prob = prob_splitting_method(pauli_error, nr_hash, 5, single_qubit_infid)
	print("P ( {} ) = {}".format(pauli_error, prob))
	
	# pauli_probs = AssignErrorProbs(known_pauli_indices, known_probs, pauli_errors, single_qubit_infid)
	# pauli_probs[mask] = total_unknown / np.sum(pauli_probs[mask]) * pauli_probs[mask]

	# pauli_probs_dp = CreateIIDPauli(single_qubit_infid, qecc)
	# pauli_probs_dp[known_pauli_indices] = known_probs
	# pauli_probs_dp[mask] = total_unknown / np.sum(pauli_probs_dp[mask]) * pauli_probs_dp[mask]

	# print("Errors and their probabilities")
	# for w in qecc.group_by_weight:
	# 	print("===========\nWeight {} errors".format(w))
	# 	for p in range(qecc.group_by_weight[w].size):
	# 		error_index = qecc.group_by_weight[w][p]
	# 		print("Prob( {} ): Heuristic = {}, Depolarizing = {}, Relative factor = {}".format(pauli_errors[error_index], pauli_probs[error_index], pauli_probs_dp[error_index], pauli_probs[error_index] / pauli_probs_dp[error_index]))
	
	# print("Sum of all error probabilities: Heuristic = {}, Depolarizing = {}.".format(np.sum(pauli_probs), np.sum(pauli_probs_dp)))