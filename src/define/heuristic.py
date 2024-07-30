from numba import jit, njit, prange
from numba.core import types
import numpy as np

@njit("uint8[:](uint8[:])")
def FormatPauliError(pauli_error_op):
	# Format a Pauli operator as a list of tuples: [(x_1, P_1), ..., (x_k, P_k)] where x_i are integers and P_i are Pauli operators indexed from 0 to 3.
	# The Pauli error is given to us as a list of single qubit operators in the tensor product form.
	support_size = np.count_nonzero(pauli_error_op)
	pauli_format = np.empty(2 * support_size, dtype = np.uint8)
	n_nontrivial = 0
	for k in range(pauli_error_op.size):
		if (pauli_error_op[k] > 0):
			pauli_format[2 * n_nontrivial] = k
			pauli_format[2 * n_nontrivial + 1] = pauli_error_op[k]
			n_nontrivial = n_nontrivial + 1
	# print("pauli error ", pauli_error_op, " is formatted as ", pauli_format)
	return pauli_format

@njit("uint64(uint8[:])")
def HashPauliError(pauli_error):
	# Convert a Pauli error formatted as [(x_1, P_1), ..., (x_k, P_k)] into a string: x_1P_1_..._x_kP_k.
	# We will assign the error its index in an lexicographic ordering of n-qubit Pauli errors
	# O ( [(x_1, P_1), ..., (x_k, P_k)] ) = 4^x_1 * P_1 + 4^x_2 * P_2 + ... + 4^x_k * P_k
	# O ( [0, 3, 1, 2, 2, 3, 4, 2] ) = 3 * 4^0 + 2 * 4^1 + 3 * 4^2 + 2 * 4^4 = 571
	hash_encoding = 0
	support_size = len(pauli_error) // 2
	if (len(pauli_error) > 0):
		for k in range(support_size):
			(x_k, P_k) = (pauli_error[2 * k], pauli_error[2 * k + 1])
			hash_encoding = hash_encoding + np.power(4, x_k) * P_k
	# print("Error ", pauli_error, " is logged with hash ",  hash_encoding, ".")
	return types.uint64(hash_encoding)

@njit("float64[:](uint64[:], float64[:], uint8[:, :])")
def BuildNRHash(known_paulis, known_probs, operators):
	# We want to store the NR data as a hash table.
	# For each error in the NR data, we want to use the string encoding of its format:
	# [(x_1, P_1), ..., (x_k, P_k)]
	# as a index for a dictionary to store the probability of the error retrieved from NR.
	(nerrors, __) = operators.shape
	nr_hash = -1 * np.ones(nerrors, dtype = np.float64)
	# print("Building the hash table")
	for p in range(known_paulis.size):
		error = operators[known_paulis[p], :].astype(np.uint8)
		nr_hash[HashPauliError(FormatPauliError(error))] = types.float64(known_probs[p])
	return nr_hash

def get_partitions(arr):
	# Compute all bi-partitions of an array.
	# We will compute all subsets of an array. Each subset would correspond to a bi-partition, along with its complement.
	# We will exclude the trivial cases: the empty set and the complete set.
	# Assume that the set has n elements.
	# Each non-trivial subset corresponds to a unique binary sequence of n bits that encodes a number from 1 to 2^n - 2.
	# The position of 1's in the binary sequence determines the elements selected in the set.
	n = len(arr)
	partitions = []
	for s in range(1, np.power(2, n, dtype = int) - 1):
		partition_encoding = np.array(list(map(int, np.binary_repr(s, width=n))), dtype = int)
		selected_elements, = np.nonzero(partition_encoding)
		unselected_elements, = np.nonzero(1 - partition_encoding)
		left_partition = [arr[j] for j in selected_elements]
		right_partition = [arr[j] for j in unselected_elements]
		partitions.append((left_partition, right_partition))
	return partitions

@njit("uint8[:](uint64, uint64)")
def dec2bin(dec_num, nbits):
	# Convert from the decimal to binary
	bin_num = np.zeros(nbits, dtype = np.uint8)
	for b in range(nbits):
		bin_num[nbits - b - 1] = types.uint8(dec_num % 2)
		dec_num = types.uint8(dec_num // 2)
	return bin_num


@jit("float64(uint8[:], float64[:], uint8, float64)", fastmath=True)
def prob_splitting_method(pauli_error, nr_hash, nqubits, single_qubit_infid):
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

	# print("Error = ", pauli_error)
	
	# This base case will never happen because we will directly deal with a single qubit error.
	# We have to state this case explicity for the compiler. 
	if (len(pauli_error) == 0):
		return nr_hash[0]
	
	else:
		pauli_key = HashPauliError(pauli_error)
		
		if (nr_hash[pauli_key] >= 0):
			prob = nr_hash[pauli_key]
			# print("Found ", pauli_error, " in nr_hash and its probability is ", prob)
		
		else:
			# print("Guessing the probability of ", pauli_error, ".")

			support_size = len(pauli_error) // 2

			if (support_size == 1):
				prob = single_qubit_infid / 3 * np.power(1 - single_qubit_infid, nqubits - 1)
				# print("Depolarizing channel assumption invoked, with probability = ", prob)
				
			else:
				
				# We associate each partition to a binary string of length equal to the size of the support.
				# The location of 0's in the binary string denotes the left partition and the location of ones denotes the right partition.

				n_partitions = np.power(2, support_size - 1)
				
				# print("Computing ", n_partitions - 2, " partitions of the error ", pauli_error)
				
				sum_prob = 0
				max_prob = 0
				for j in range(1, n_partitions):
					binary_repr_j = dec2bin(j, support_size)

					left_partition_size = np.count_nonzero(binary_repr_j)
					left_partition = np.empty(2 * left_partition_size, dtype = np.uint8)
					right_partition_size = support_size - left_partition_size
					right_partition = np.empty(2 * right_partition_size, dtype = np.uint8)
					left_count = 0
					right_count = 0
					# print("Partition ", j, "of ", pauli_error, " corresponding to ", binary_repr_j)
					for k in range(support_size):
						if (binary_repr_j[k] == 1):
							left_partition[2 * left_count] = pauli_error[2 * k]
							left_partition[2 * left_count + 1] = pauli_error[2 * k + 1]
							left_count = left_count + 1
						else:
							right_partition[2 * right_count] = pauli_error[2 * k]
							right_partition[2 * right_count + 1] = pauli_error[2 * k + 1]
							right_count = right_count + 1

					# print("left = ", left_partition, " and right = ", right_partition)

					prob_left = prob_splitting_method(left_partition, nr_hash, nqubits, single_qubit_infid)
					prob_right = prob_splitting_method(right_partition, nr_hash, nqubits, single_qubit_infid)

					# print("Probability of left partition = ", prob_left, "\nProbability of right partition = ", prob_right)

					error_prob = prob_left * prob_right
					# Normalization: divide the error probability by (1-p)^n to compensate for Identity terms.
					norm = (1 - single_qubit_infid) ** nqubits
					error_prob = error_prob / norm
					
					# sum_prob = sum_prob + error_prob

					if (max_prob < error_prob):
						max_prob = error_prob

				# prob = sum_prob
				prob = max_prob
	
		# print("Prob( ", pauli_error, " ) = ", prob)
		nr_hash[pauli_key] = prob
	return prob


@njit("float64[:](uint64[:], float64[:], uint8[:,:], float64)", fastmath=True)
def AssignErrorProbs(known_paulis, known_probs, pauli_errors, single_qubit_infid):
	# Assign the probability of errors using the splitting method described in prob_splitting_method(...).
	
	print("Using the splitting method to assign probabilities of ", pauli_errors.shape[0] - known_paulis.size, " errors excluded in the NR data.")
	
	(npauli, nqubits) = pauli_errors.shape
	nr_hash = BuildNRHash(known_paulis, known_probs, pauli_errors)
	
	pauli_probs = np.zeros(npauli, dtype = np.float64)
	for p in prange(npauli):
		weight = np.count_nonzero(pauli_errors[p, :])
		pauli_probs[p] = prob_splitting_method(FormatPauliError(pauli_errors[p, :]), nr_hash, nqubits, single_qubit_infid)

	return pauli_probs

def FilterUnrealInferences(known_paulis, known_probs, inferred_probs):
	# We want to eliminate instances where the heuristic guessed a probability for a Pauli error that is higher than the chosen errors in the NR dataset.
	npauli = inferred_probs.size
	adjusted_probs = np.zeros(npauli, dtype = np.float64)
	
	# Identify the Pauli errors that are not in the NR data.
	mask = np.ones(npauli, dtype=bool)
	mask[known_paulis] = False

	# If the error is excluded in the NR dataset, check if its probability is higher than the lowest probability of an error in the NR dataset.
	# If yes, then set it to the probability of the lowest known error in the NR dataset.
	min_nr_prob = np.min(known_probs)
	n_unreal_probs = 0
	for p in prange(npauli):
		if (mask[p]):
			if (inferred_probs[p] > min_nr_prob):
				adjusted_probs[p] = min_nr_prob
				n_unreal_probs = n_unreal_probs + 1
			else:
				adjusted_probs[p] = inferred_probs[p]
		else:
			adjusted_probs[p] = inferred_probs[p]
	
	print("Filtered {} unreal probabilities where the heuristic inferrence is higher than the NR data.".format(n_unreal_probs))

	return adjusted_probs