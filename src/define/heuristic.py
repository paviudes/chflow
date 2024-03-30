import numpy as np

def PartitionError(pauli, split_at):
	# Given an n-qubit error E, find partitions of it into a k-qubit error and an n-k qubit error, for a given k.
	# The Pauli error is provided as a list of tuples of the form (x, P) where x is a qubit index and P is a single qubit Pauli error P supported on x.
	error_left = pauli[:split_at]
	error_right = pauli[split_at:]
	return (error_left, error_right)

def FormatPauliError(pauli_error_op):
	# Format a Pauli operator as a list of tuples: [(x_1, P_1), ..., (x_k, P_k)] where x_i are integers and P_i are Pauli operators indexed from 0 to 3.
	# The Pauli error is given to us as a list of single qubit operators in the tensor product form.
	pauli_format = [(q, pauli_error_op[q]) for q in range(len(pauli_error_op)) if pauli_error_op[q] > 0]
	return pauli_format

def HashPauliError(pauli_error):
	# Convert a Pauli error formatted as [(x_1, P_1), ..., (x_k, P_k)] into a string: x_1P_1_..._x_kP_k.
	if (len(pauli_error) == 0):
		hash_encoding = "I"
	else:
		hash_encoding = "_".join(["%d%d" % (tup[0], tup[1]) for tup in pauli_error])
	return hash_encoding

def BuildNRHash(known_paulis, known_probs, operators):
	# We want to store the NR data as a hash table.
	# For each error in the NR data, we want to use the string encoding of its format:
	# [(x_1, P_1), ..., (x_k, P_k)]
	# as a index for a dictionary to store the probability of the error retrieved from NR.
	nr_hash = {}
	# print("Building the hash table")
	for p in range(known_paulis.size):
		error = operators[known_paulis[p], :]
		# print("E = {}\nF_E = {}".format(error, FormatPauliError(error)))
		# print("and hash = {}, prob = {}".format(HashPauliError(FormatPauliError(error)), known_probs[p]))
		nr_hash[HashPauliError(FormatPauliError(error))] = known_probs[p]
	return nr_hash

def prob_splitting_method(pauli_error, nr_hash, nqubits = 7, single_qubit_infid = None):
	# Assign the probability of an error given the error probabilities extracted from NR.
	# We assume that the Pauli error is specified in the format {(q,P) : where P is the single qubit error from X, Y or Z supported on q}
	# Refer to the handwritten notes on Slack for a detailed explaination of this algorithm.
	# If the error exists in the NR data itself:
		# # then retirve it probability form nr_hash.
		# prob = nr_hash[HashPauliError(pauli_error)]
	# Else:
		# Let this error be P.
		# # if P is supported one only one qubit, then its probability is r/3 where r is the single qubit infidelity.
		# if len(P) == 1:
		# prob = r/3 * (1-r)^6
		# Else: For every partition j of the non-trivial support of P := P1 . P2, do
					# set prob_P1 = prob_splitting_method(P1, nr_hash)
					# set prob_P2 = prob_splitting_method(P2, nr_hash)
					# set prob_j = prob_P1 . prob_P2
				# set prob = max {prob_1, prob_2, ..., prob_M} where M is the number of partitions. # Compute the maximum probability.
	# return prob
	prob = 0
	pauli_key = HashPauliError(pauli_error)
	
	if (pauli_key in nr_hash):
		prob = nr_hash[pauli_key]
	
	else:
		support_size = len(pauli_error)

		if (support_size == 1):
			if (single_qubit_infid is None):
				single_qubit_infid = 1 - np.power(1 - nr_hash["I"], 1/nqubits)
			prob = single_qubit_infid / 3 * np.power(1 - single_qubit_infid, nqubits - 1)
		
		else:
			max_prob = 0

			for p in range(1, support_size):
				
				left_partition = pauli_error[:p]
				right_partition = pauli_error[p:]

				prob_left = prob_splitting_method(left_partition, nr_hash)
				prob_right = prob_splitting_method(right_partition, nr_hash)

				if (max_prob < prob_left * prob_right):
					max_prob = prob_left * prob_right

			prob = max_prob

	return prob


def AssignErrorProbs(known_paulis, known_probs, pauli_errors, single_qubit_infid):
	# Assign the probability of errors using the splitting method described in prob_splitting_method(...).
	
	print("Using the splitting method to assign probabilities of {} errors excluded in the NR data.".format(pauli_errors.shape[0] - known_paulis.size))

	(npauli, nqubits) = pauli_errors.shape

	nr_hash = BuildNRHash(known_paulis, known_probs, pauli_errors)
	pauli_probs = np.zeros(npauli, dtype = np.double)
	
	for p in range(npauli):
		pauli_probs[p] = prob_splitting_method(FormatPauliError(pauli_errors[p, :]), nr_hash, nqubits, single_qubit_infid)

	return pauli_probs