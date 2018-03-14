import os
import numpy as np
import itertools as it
import chanreps as crep
import randchans as rchan
import chandefs as chdef
import chanapprox as capp
import metrics as ml

DEBUG = 0

########################################## GLOBAL VARIABLES ##########################################
Channels = {'idn':["Identity channel", "No parameters.", "white"],
			'ad':["Amplitude damping channel", "Damping rate", "$\lambda$", "forestgreen"],
			'bf':["Bit flip channel", "Bit flip probability", "$p_{\\textsf{bit flip}}$", "sienna"],
			'pd':["Phase damping channel", "Damping rate", "$p_{\\textsf{damp}}$", "deeppink"],
			'bpf':["Bit-phase flip channel", "Bit-phase flip probability", "$p_{\\textsf{bit-phase}}$", "darkkhaki"],
			'dp':["Depolarizing channel", "Depolarizing rate", "$p$", "red"],
			'rtz':["Rotation about Z-axis", "Rotation angle as a fraction of pi", "$\\frac{\\theta}{2\\pi}$", "black"],
			'rtx':["Rotation about X-axis", "Rotation angle as a fraction of pi", "$\\frac{\\theta}{2\\pi}$", "darkviolet"],
			'rty':["Rotation about Y-axis", "Rotation angle as a fraction of pi", "$\\frac{\\theta}{2\\pi}$", "darkorchid"],
			'rtpd':["Z-Rotation with Phase dampling", "Damping rate, Rotation angle as a fraction of pi.", "$p_{\\textsf{damp}}$", "midnightblue"],
			'rtnp':["Rotation about an arbitrary axis of the Bloch sphere", "phi, theta, rotation angle as a fraction of pi (the rotation axis is decribed by (phi, theta)).", "$\\frac{\\theta}{2\\pi}$", "indigo"],
			'rand':["Random channel", "Proximity to identity channel, method for generating random unitary.", "$U = \\exp(i\\mathsf{H}\\alpha)$", "peru"],
			'shd':["Stochastic Hadamard channel", "Hadamard rate", "$p_{\\textsf{H}}$", "maroon"],
			'sru':["Stochastic T channel", "T rate", "$p_{\\textsf{T}}$", "chocolate"],
			'rru':["Rotation about a T axis", "$\\theta$", "$\\frac{\\theta}{2\\pi}$", "purple"],
			'corr':["p Depolarizing + sqrt(p) HH correlations", "$p$", "$p$", "limegreen"],
			'pl':["Photon Loss channel", "number of photons (alpha), decoherence rate (gamma)", "$\\lambda$", "steelblue"],
			'gd':["Generalized damping", "Dephasing rate (p), Relaxation rate (lambda)", "$\\lambda$", "green"],
			'gdt':["Generalized time dependent damping", "Relaxation (t/T1), ratio (T2/T1)", "$\\frac{T_{2}}{T_{1}}$", "darkgreen"],
			'gdtx':["Eplicit generalized time dependent damping", "T1, T2, t", "$T_{2}$", "olive"]
}

def GenChannelsForCalibration(chname, rangeinfo):
	# Load a set of channels from thier symbolic description as: <channel type> l1,h1,n1;l2,h2,n2;...
	# where li,hi,ni specify the range of the variable i in the channel.
	# Store the channels as a 3D array of size (number of channels) x 4 x 4.
	noiserange = np.array(map(lambda intv: map(np.float, intv.split(",")), rangeinfo.split(";")), dtype = np.float)
	pvalues = np.array(list(it.product(*map(lambda intv: np.linspace(intv[0], intv[1], int(intv[2]), dtype = np.float), noiserange))), dtype = np.float)
	channels = np.zeros((np.prod(noiserange[:, 2], dtype = np.int), 4, 4), dtype = np.complex128)
	for i in range(channels.shape[0]):
		channels[i, :, :] = crep.ConvertRepresentations(chdef.GetKraussForChannel(chname, *pvalues[i, :]), "krauss", "choi")
	return (pvalues, channels)


def Calibrate(chname, rangeinfo, metinfo, xcol = 0):
	# Plot various metrics for channels corresponding to a noise model with a range of different parameter values.
	metrics = map(lambda met: met.strip(" "), metinfo.split(","))
	(noiserates, channels) = GenChannelsForCalibration(chname, rangeinfo)
	ml.GenCalibrationData(chname, channels, noiserates, metrics)
	ml.PlotCalibrationData1D(chname, metrics, xcol)
	return None


def Save(fname, channel, rep = "Unknwon"):
	# Save a channel into a file.
	if (fname.endswith(".txt")):
		np.savetxt(fname, channel, fmt='%.18e', delimiter=" ", newline = '\n', comments = ("# Representation: %s" % (rep)))
	elif (fname.endswith(".npy")):
		np.save(fname, channel)
	else:
		pass
	return channel	


def Print(channel, rep = "unknown"):
	# Print a channel in its current representation
	if (rep == "krauss"):
		print("Krauss representation")
		for i in range(channel.shape[0]):
			print("E_%d\n%s" % (i + 1, np.array_str(channel[i, :, :], max_line_width = 150, precision = 3, suppress_small = True)))
	elif (rep == "choi"):
		print("Choi representation")
		print("%s" % (np.array_str(channel, max_line_width = 150, precision = 3, suppress_small = True)))
	elif (rep == "chi"):
		print("Chi representation")
		print("%s" % (np.array_str(channel, max_line_width = 150, precision = 3, suppress_small = True)))
	elif (rep == "process"):
		print("Pauli Liouville representation")
		print("%s" % (np.array_str(channel, max_line_width = 150, precision = 3, suppress_small = True)))
	elif (rep == "stine"):
		print("Stinespring dialation representation")
		print("%s" % (np.array_str(channel, max_line_width = 150, precision = 3, suppress_small = True)))
	else:
		print("Unknwon representation \"%s\"." % (rep))
		print("%s" % (np.array_str(channel, max_line_width = 150, precision = 3, suppress_small = True)))
	return None
