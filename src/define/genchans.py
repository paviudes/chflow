import sys
import os
import numpy as np
from scipy import linalg as linalg
import metrics as ml
import randchans as rchan
import chandefs as chdef
import chanreps as crep
import fnames as fn

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
	completed = 1
	if (not (os.path.exists("./../physical/"))):
		os.mkdir("./../physical/")
	# Create quantum channels for various noise parameters and store them in the process matrix formalism.
	channels = np.zeros((submit.samps, 4, 4), dtype = np.longdouble)
	submit.params = np.zeros((submit.noiserates.shape[0] * submit.samps, submit.noiserates.shape[1] + 1), dtype = np.longdouble)
	for i in range(submit.noiserates.shape[0]):
		if (submit.scale == 1):
			noise = submit.noiserates[i, :]
		else:
			noise = np.power(submit.scale, submit.noiserates[i, :])
		for j in range(submit.samps):
			submit.params[i * submit.samps + j, :-1] = submit.noiserates[i, :]
			submit.params[i * submit.samps + j, -1] = j
			# To create a random quantum channel, construct a random unitary operator by exponentiating a hertimitan operator and do a Krauss decomposition.
			channels[j, :, :] = crep.ConvertRepresentations(chdef.GetKraussForChannel(submit.channel, *noise), 'krauss', 'process')
			print("\r\033[2mPreparing physical channels... %d (%d%%) done.\033[0m" % (completed, 100*completed/float(submit.noiserates.shape[0]  * submit.samps))),
			sys.stdout.flush()
			completed = completed + 1
		# Write all the process matrices corresponding to all samples
		chanfile = fn.PhysicalChannel(submit, submit.noiserates[i], loc = "local")
		np.save(chanfile, channels)
		submit.chfiles.append(chanfile)
	print("")
	if (submit.nodes > 0):
		submit.cores[0] = int(np.ceil(len(submit.params)/np.longdouble(submit.nodes)))
	# submit.cores[0] = min(submit.cores[0], len(submit.params))
	submit.nodes = int(np.ceil(len(submit.params)/np.longdouble(submit.cores[0])))
	return None
