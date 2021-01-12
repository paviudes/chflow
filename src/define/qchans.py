import os
import numpy as np
import itertools as it
from define import chanreps as crep
from define import randchans as rchan
from define import chandefs as chdef
from define import chanapprox as capp
from define import metrics as ml

Channels = {
	"idn": {
		"name": "Identity channel",
		"params": [],
		"latex": [],
		"color": "white",
		"Pauli": 1,
		"corr": 0,
	},
	"ad": {
		"name": "Amplitude damping",
		"params": ["Damping rate"],
		"latex": ["$\\lambda$"],
		"color": "forestgreen",
		"Pauli": 0,
		"corr": 0,
	},
	"bf": {
		"name": "Bit flip channel",
		"params": ["X error rate"],
		"latex": ["p_{X}"],
		"color": "sienna",
		"Pauli": 1,
		"corr": 0,
	},
	"pd": {
		"name": "Dephasing channel",
		"params": ["Dephasing rate"],
		"latex": ["$p$"],
		"color": "deeppink",
		"Pauli": 1,
		"corr": 0,
	},
	"bpf": {
		"name": "Bit phase flip",
		"params": ["Y error rate"],
		"latex": ["$p_{Y}$"],
		"color": "darkkhaki",
		"Pauli": 1,
		"corr": 0,
	},
	"dp": {
		"name": "Depolarizing channel",
		"params": ["Depolarizing rate"],
		"latex": ["$p$"],
		"color": "red",
		"Pauli": 1,
		"corr": 0,
	},
	"pauli": {
		"name": "Pauli channel",
		"params": ["Prob I", "Prob X", "Prob Y", "Prob Z"],
		"latex": ["$p_{I}$", "$p_{X}$", "$p_{Y}$", "$p_{Z}$"],
		"color": "red",
		"Pauli": 1,
		"corr": 0,
	},
	"up": {
		"name": "Uncorrelated Pauli channel",
		"params": ["Infidelity"],
		"latex": ["$1 - p_{I}$"],
		"color": "red",
		"Pauli": 1,
		"corr": 0,
	},
	"rtx": {
		"name": "Rotation about the X-axis",
		"params": ["Rotation angle"],
		"latex": ["\\delta/2\\pi"],
		"color": "darkviolet",
		"Pauli": 0,
		"corr": 0,
	},
	"rty": {
		"name": "Rotation about Y-axis",
		"params": ["Rotation angle"],
		"latex": ["\\delta/2\\pi"],
		"color": "darkorchid",
		"Pauli": 0,
		"corr": 0,
	},
	"rtz": {
		"name": "Rotation about the Z-axis",
		"params": ["Rotation angle"],
		"latex": ["\\delta/2\\pi"],
		"color": "black",
		"Pauli": 0,
		"corr": 0,
	},
	"rtnp": {
		"name": "Rotation about an arbitrary axis of the Bloch sphere",
		"params": [
			"phi (azimuthal angle)",
			"theta (angle with Z-axis)",
			"rotation angle as a fraction of pi (the rotation axis is decribed by (phi, theta)).",
		],
		"latex": ["$\\theta/2\\pi$", "\\phi/2\\pi", "\\delta"],
		"color": "indigo",
		"Pauli": 0,
		"corr": 0,
	},
	"rtas": {
		"name": "Asymmetric Gaussian random rotations about arbitrary axes of the Bloch sphere",
		"params": [
			"Number of qubits",
			"Average angle of rotation",
			"Standard deviation for rotation angle",
			"Standard deviation for angle with Z-axis",
			"Standard deviation for azimuthal angle",
		],
		"latex": [
			"N",
			"$\\mu_{\\delta}$",
			"\\Delta_{\\delta}",
			"\\Delta_{\\theta}",
			"\\Delta_{\\phi}",
		],
		"color": "midnightblue",
		"Pauli": 0,
		"corr": 2,
	},
	"rtasu": {
		"name": "Asymmetric uniformly random rotations about arbitrary axes of the Bloch sphere",
		"params": [
			"Number of qubits",
			"Average angle of rotation",
			"Standard deviation for rotation angle",
		],
		"latex": ["N", "$\\mu_{\\delta}$", "\\Delta_{\\delta}"],
		"color": "midnightblue",
		"Pauli": 0,
		"corr": 2,
	},
	"rand": {
		"name": "Random CPTP map",
		"params": ["Interaction time of Hamiltonian on system and environment"],
		"latex": ["$t$"],
		"color": "peru",
		"Pauli": 0,
		"corr": 0,
	},
	"randunit": {
		"name": "Random unitary channel",
		"params": ["Interaction time"],
		"latex": ["$t$"],
		"color": "goldenrod",
		"Pauli": 0,
		"corr": 0,
	},
	"pcorr": {
		"name": "Random correlated Pauli channel",
		"params": "Infidelity",
		"latex": ["$1 - p_{I}$"],
		"color": "limegreen",
		"Pauli": 1,
		"corr": 1,
	},
	"usum": {
		"name": "Correlated non Pauli channel as sum of unitaries",
		"params": ["Probability", "Angle"],
		"latex": ["$p$", "$\\theta$"],
		"color": "limegreen",
		"Pauli": 0,
		"corr": 3,
	},
	"cptp": {
		"name": "Correlated non Pauli channel as sum of CPTP maps",
		"params": ["Angle", "Cutoff", "Number of maps", "Mean"],
		"latex": ["$\\theta$", "K", "$n_{\\mathsf{maps}}$", "\\mu"],
		"color": "limegreen",
		"Pauli": 0,
		"corr": 3,
	},
	"ising": {
		"name": "Non Pauli channel implementing Ising type interactions",
		"params": ["J", "mu", "time"],
		"latex": ["$J$", "$\\mu$", "$t$"],
		"color": "limegreen",
		"Pauli": 0,
		"corr": 3,
	},
	"wpc": {
		"name": "Worst Pauli channel",
		"params": "Infidelity",
		"latex": ["$1 - p_{I}$"],
		"color": "limegreen",
		"Pauli": 1,
		"corr": 1,
	},
	"pl": {
		"name": "Photon loss channel",
		"params": ["number of photons (alpha)", "decoherence rate (gamma)"],
		"latex": ["$\\alpha$", "$\\gamma$"],
		"color": "steelblue",
		"Pauli": 0,
		"corr": 0,
	},
	"gd": {
		"name": "Generalized damping",
		"params": ["Relaxation", "dephasing"],
		"latex": ["$\\lambda$", "$p$"],
		"color": "green",
		"Pauli": 0,
		"corr": 0,
	},
	"gdt": {
		"name": "Generalized time dependent damping",
		"params": ["Relaxation (t/T1)", "ratio (T2/T1)"],
		"latex": ["$\\frac{t}{T_{1}}$", "$\\frac{T_{2}}{T_{1}}$"],
		"color": "darkgreen",
		"Pauli": 0,
		"corr": 0,
	},
	"gdtx": {
		"name": "Explicit generalized time dependent damping",
		"params": ["T1", "T2", "t"],
		"latex": ["$t$", "$T_{1}$", "$T_{2}$"],
		"color": "olive",
		"Pauli": 0,
		"corr": 0,
	},
}


def GenChannelsForCalibration(chname, rangeinfo):
	# Load a set of channels from thier symbolic description as: <channel type> l1,h1,n1;l2,h2,n2;...
	# where li,hi,ni specify the range of the variable i in the channel.
	# Store the channels as a 3D array of size (number of channels) x 4 x 4.
	noiserange = np.array(
		list(
			map(lambda intv: list(map(np.float, intv.split(","))), rangeinfo.split(";"))
		),
		dtype=np.float,
	)
	pvalues = np.array(
		list(
			it.product(
				*list(
					map(
						lambda intv: np.linspace(intv[0], intv[1], int(intv[2])).astype(
							np.float
						),
						noiserange,
					)
				)
			)
		),
		dtype=np.float,
	)
	channels = np.zeros(
		(np.prod(noiserange[:, 2], dtype=np.int), 4, 4), dtype=np.complex128
	)
	for i in range(channels.shape[0]):
		channels[i, :, :] = crep.ConvertRepresentations(
			chdef.GetKraussForChannel(chname, *pvalues[i, :]), "krauss", "choi"
		)
	return (pvalues, channels)


def Calibrate(chname, rangeinfo, metinfo, xcol=0, ycol=-1):
	# Plot various metrics for channels corresponding to a noise model with a range of different parameter values.
	metrics = list(map(lambda met: met.strip(" "), metinfo.split(",")))
	(noiserates, channels) = GenChannelsForCalibration(chname, rangeinfo)
	ml.GenCalibrationData(chname, channels, noiserates, metrics)
	if ycol == -1:
		ml.PlotCalibrationData1D(chname, metrics, xcol)
	else:
		ml.PlotCalibrationData2D(chname, metrics, xcol, ycol)
	return None


def SaveChan(fname, channel, rep="Unknwon"):
	# Save a channel into a file.
	if fname.endswith(".txt"):
		np.savetxt(
			fname,
			channel,
			fmt="%.18e",
			delimiter=" ",
			newline="\n",
			comments=("# Representation: %s" % (rep)),
		)
	elif fname.endswith(".npy"):
		np.save(fname, channel)
	else:
		pass
	return channel


def Print(channel, rep="unknown"):
	# Print a channel in its current representation
	if rep == "krauss":
		print("Krauss representation")
		for i in range(channel.shape[0]):
			print(
				"E_%d\n%s"
				% (
					i + 1,
					np.array_str(
						channel[i, :, :],
						max_line_width=150,
						precision=3,
						suppress_small=True,
					),
				)
			)
	elif rep == "choi":
		print("Choi representation")
		print(
			"%s"
			% (
				np.array_str(
					channel, max_line_width=150, precision=3, suppress_small=True
				)
			)
		)
	elif rep == "chi":
		print("Chi representation")
		print(
			"%s"
			% (
				np.array_str(
					channel, max_line_width=150, precision=3, suppress_small=True
				)
			)
		)
	elif rep == "process":
		print("Pauli Liouville representation")
		print(
			"%s"
			% (
				np.array_str(
					channel, max_line_width=150, precision=3, suppress_small=True
				)
			)
		)
	elif rep == "stine":
		print("Stinespring dialation representation")
		print(
			"%s"
			% (
				np.array_str(
					channel, max_line_width=150, precision=3, suppress_small=True
				)
			)
		)
	else:
		print('Unknwon representation "%s".' % (rep))
		print(
			"%s"
			% (
				np.array_str(
					channel, max_line_width=150, precision=3, suppress_small=True
				)
			)
		)
	return None
