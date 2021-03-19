import numpy as np
from define.QECCLfid import utils as ut
from define.randchans import CreateIIDPauli
from define.chanreps import PauliConvertToTransfer
from define import fnames as fn
from scipy.special import comb
from define.qcode import PrepareSyndromeLookUp
from define.QECCLfid.utils import SamplePoisson

def TailorDecoder(qecc, channel, bias=None):
	# Tailor a decoder to an error model by exploiting simple structure.
	# At the moment, this only works differently from MWD for a biased Pauli error model "bpauli".
	# We need to design the relative importance that should be given to I, X, Y and Z errors.
	# print("TailorDecoder({}, {})".format(submit.channel, noise))
	if channel == "bpauli":
		cX = int(bias)
		cZ = 1
		cY = int(bias)
		qecc.weight_convention = {"method": "bias", "weights": {"X": cX, "Y": cY, "Z": cZ}}
		PrepareSyndromeLookUp(qecc)
		# print("Lookup table for {} code with bias {}.".format(qecc.name, bias))
	else:
		for l in range(submit.levels):
			qecc.weight_convention = {"method": "Hamming"}
			PrepareSyndromeLookUp(qecc)
	return None

def GetTotalErrorBudget(dbs, noise, sample):
	# Compute the total number of distinct Pauli error rates included in the NR dataset.
	nrw = np.loadtxt(fn.NRWeightsFile(dbs, noise))[sample, :]
	max_weight = 1 + dbs.eccs[0].N//2
	(weight_count_alpha, __) = ComputeNRBudget(nrw, [dbs.decoder_fraction], dbs.eccs[0].N, max_weight=max_weight)
	budget = np.sum(weight_count_alpha)
	# print("alpha = {}, budget = {}".format(dbs.decoder_fraction, budget))
	return budget


def ComputeNRBudget(nr_weights_all, alphas, nq, max_weight=None):
	# Compute the relative budget of weight-w error rates in the NR dataset.
	n_paulis = nr_weights_all.size
	if max_weight is None:
		max_weight = nr_weights_all.max()
	n_alphas = len(alphas)
	relative_budget = np.zeros((n_alphas, max_weight + 1), dtype=np.float)
	
	# Count the frequency of each weight.
	weight_counts = np.zeros(max_weight + 1, dtype=np.int)
	min_weight_counts = np.zeros(max_weight + 1, dtype=np.int)
	for w in range(max_weight + 1):
		weight_counts[w] = np.count_nonzero(nr_weights_all == w)
	min_weight_counts[0] = 1

	for (alpha_count, alpha) in enumerate(alphas):
		weight_count_alpha = np.maximum((alpha * weight_counts).astype(np.int), min_weight_counts)
		budget_pauli_count = np.sum(weight_count_alpha)
		# print("alpha = {}, budget_pauli_count = {}\nweight_count_alpha\n{}".format(alpha, budget_pauli_count, weight_count_alpha))
		
		# Resetting the number of errors of each weight to the theoretical maximum
		nerrors_weight = np.zeros(max_weight + 1, dtype = np.int)
		excess_budget = 0
		for w in range(weight_counts.size):
			count = weight_count_alpha[w]
			if(count > comb(nq, w) * (3 ** w)):
				nerrors_weight[w] = comb(nq, w) * (3 ** w)
				excess_budget += count - nerrors_weight[w]
			else:
				nerrors_weight[w] = count
		
		# Redistributing excess errors generated by Poisson distribution
		# Lower weights get priority over higher ones
		for w in range(max_weight + 1):
			need_weight_w = comb(nq, w)*(3 ** w) - nerrors_weight[w]
			add_to_weight_w = min(excess_budget, need_weight_w)
			nerrors_weight[w] += add_to_weight_w
			excess_budget -= add_to_weight_w
		
		relative_budget[alpha_count, :] = [nerrors_weight[w]*100/budget_pauli_count for w in range(max_weight+1)]
	
	return (nerrors_weight, relative_budget)


def GetLeadingPaulis(lead_frac, qcode, chan_probs, option, nr_weights_all = None, max_weight = None):
	# Get the leading Pauli probabilities in the iid model.
	# To get the indices of the k-largest elements: https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
	# If the option is "full", it supplies top alpha fraction of entire chi diagonal
	# If the option is "weight", it supplies errors with weights sampled from Poisson distribution having mean 1 and excess budget redistributed 
	if chan_probs.ndim == 1:
		iid_chan_probs = chan_probs
	else:
		iid_chan_probs = ut.GetErrorProbabilities(
			qcode.PauliOperatorsLST, chan_probs, 0
		)

	if option == "full":
		nPaulis = max(1, int(lead_frac * (4 ** qcode.N)))
		leading_paulis = np.argsort(iid_chan_probs)[-nPaulis:]


	elif option == "weight":

		if max_weight is None:
			max_weight = qcode.N//2 + 1

		if qcode.group_by_weight is None:
			PrepareSyndromeLookUp(qcode)

		(nerrors_weight, __) = ComputeNRBudget(nr_weights_all, [lead_frac], qcode.N, max_weight=max_weight)

		leading_paulis = np.zeros(np.sum(nerrors_weight, dtype = np.int), dtype=np.int)
		start = 0
		for w in range(max_weight + 1):
			if (nerrors_weight[w] > 0):
				stop = start + nerrors_weight[w]
				errors_wtw = qcode.group_by_weight[w]
				indices_picked = errors_wtw[np.argsort(iid_chan_probs[errors_wtw])[-nerrors_weight[w]:]]
				leading_paulis[start:stop] = indices_picked[:]
				start = stop
	return (1 - iid_chan_probs[0], leading_paulis, iid_chan_probs[leading_paulis])


def CompleteDecoderKnowledge(leading_fraction, chan_probs, qcode, option = "full", nr_weights = None):
	# Complete the probabilities given to a ML decoder.
	(infid, known_paulis, known_probs) = GetLeadingPaulis(
		leading_fraction, qcode, chan_probs, option, nr_weights
	)
	"""
	Create a function similar to GetLeadingPaulis.
	This function will simply identify the Pauli indices (in LST) we need to keep from NR.
	"""
	# print(
	#     "Number of known paulis in decoder knowledge = {}".format(known_paulis.shape[0])
	# )
	# print("Total known probability = {}".format(np.sum(known_probs)))
	infid_qubit = 1 - np.power(1 - infid, 1 / qcode.N)
	decoder_probs = CreateIIDPauli(infid_qubit, qcode)
	decoder_probs[known_paulis] = known_probs
	total_unknown = 1 - np.sum(known_probs)
	# print("Total unknown probability = {}".format(total_unknown))
	# Normalize the unknown Paulis
	# https://stackoverflow.com/questions/27824075/accessing-numpy-array-elements-not-in-a-given-index-list
	mask = np.ones(decoder_probs.shape[0], dtype=bool)
	mask[known_paulis] = False
	decoder_probs[mask] = (
		total_unknown * decoder_probs[mask] / np.sum(decoder_probs[mask])
	)
	# print("Sum of decoder probs = {}".format(np.sum(decoder_probs)))
	# print("Sorted decoder probs = {}".format(np.sort(decoder_probs)))
	return decoder_probs


def PrepareNRWeights(submit):
	# Prepare the weights of Pauli errors that will be supplied to the decoder: nr_weights.
	# Use properties of submit to retrieve the mean and cutoff of \the Poisson distribution: submit.noiserates[i, :] = (__, cutoff, __, mean)
	# Save the nr_weights to a file.
	qcode = submit.eccs[0]
	max_weight = qcode.N//2 + 1
	submit.nr_weights = np.zeros((submit.noiserates.shape[0], submit.samps, 4 ** qcode.N), dtype = np.int)
	for r in range(submit.noiserates.shape[0]):
		if (submit.channel == "cptp"):
			(__, cutoff, __, mean) = submit.noiserates[r, :]
		else:
			mean = 1
			cutoff = max_weight
		for s in range(submit.samps):
			submit.nr_weights[r, s, :] = [SamplePoisson(mean, cutoff=max_weight) for __ in range(4 ** qcode.N)]
	return None


def PrepareChannelDecoder(submit, noise, sample):
	# Generate the reference channel for partial ML decoder
	# Reference channel at present is only created for level 1
	l = 0
	qcode = submit.eccs[l]
	(nrows, ncols) = (4 ** submit.eccs[l].K, 4 ** submit.eccs[l].K)
	diagmask = [
		q * ncols * nrows + ncols * j + j for q in range(qcode.N) for j in range(nrows)
	]
	rawchan = np.load(fn.RawPhysicalChannel(submit, noise))[sample, :]
	if submit.iscorr == 0:
		chan_probs = np.reshape(np.tile(rawchan, qcode.N)[diagmask], [qcode.N, nrows])
	elif submit.iscorr == 2:
		chan_probs = np.reshape(rawchan[diagmask], [qcode.N, nrows])
	else:
		chan_probs = rawchan
	decoder_probs = CompleteDecoderKnowledge(submit.decoder_fraction, chan_probs, qcode)
	decoder_knowledge = PauliConvertToTransfer(decoder_probs, qcode)
	return decoder_knowledge
