import sys
import os
try:
	from tqdm import tqdm
	import numpy as np
	from scipy import linalg as linalg
except:
	pass
from define import metrics as ml
from define import randchans as rchan
from define import chandefs as chdef
from define import chanreps as crep
from define import fnames as fn

def ChannelPair(chtype, rates, dim, method = "qr"):
	# Generate process matrices for two channels that can be associated to the same "family".
	channels = np.zeros((2, 4, 4), dtype = np.longdouble)
	if (chtype == "rand"):
		# Generate process matrices for two random channels that originate from the same Hamiltonian.
		# If the rate already has a source, use it. When the source is fixed, there is no randomness in the channels.
		sourcefname = ("physical/rand_source_%s.npy" % (method))
		if (not os.path.isfile(sourcefname)):
			print("\033[2mCreating new random source with method = %s.\033[0m" % (method))
			randH = rchan.RandomHermitian(dim, method = method)
			np.save(sourcefname, randH)
		randH = np.load(sourcefname)
		for i in range(2):
			randU = rchan.RandomUnitary(rates[i], dim, method = method, randH = randH)
			krauss = crep.ConvertRepresentations(randU, 'stine', 'krauss')
			channels[i, :, :] = crep.ConvertRepresentations(krauss, 'krauss', 'process')
	else:
		for i in range(2):
			krauss = chdef.GetKraussForChannel(chtype, rates[i])
			channels[i, :, :] = crep.ConvertRepresentations(krauss, 'krauss', 'process')
	return channels



def PreparePhysicalChannels(submit):
	# Prepare a file for each noise rate, that contains all single qubit channels, one for each sample.
	os.system("mkdir -p %s/physical" % (submit.outdir))
	# Create quantum channels for various noise parameters and store them in the process matrix formalism.
	submit.phychans = np.zeros((submit.noiserates.shape[0], submit.samps, 4, 4), dtype = np.longdouble)
	noise = np.zeros(submit.noiserates.shape[1], dtype = np.longdouble)
	for i in tqdm(range(submit.noiserates.shape[0]), ascii=True, desc = "\033[2mPreparing physical channels:"):
		for j in range(submit.noiserates.shape[1]):
			if (submit.scales[j] == 1):
				noise[j] = submit.noiserates[i, j]
			else:
				noise[j] = np.power(submit.scales[j], submit.noiserates[i, j])
		for j in range(submit.samps):
			# To create a random quantum channel, construct a random unitary operator by exponentiating a hertimitan operator and do a Krauss decomposition.
			submit.phychans[i, j, :, :] = crep.ConvertRepresentations(chdef.GetKraussForChannel(submit.channel, *noise), 'krauss', 'process')
	print("\033[0m")
	if (submit.nodes > 0):
		submit.cores[0] = int(np.ceil(submit.noiserates.shape[0] * submit.samps/np.longdouble(submit.nodes)))
	submit.nodes = int(np.ceil(submit.noiserates.shape[0] * submit.samps/np.longdouble(submit.cores[0])))
	return None
