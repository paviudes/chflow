import os
import sys
import time
import datetime as dt
import ctypes as ct
try:
	import numpy as np
	from scipy import linalg as linalg
	import matplotlib
	from matplotlib import colors, ticker, cm
	matplotlib.use("Agg")
	from matplotlib.backends.backend_pdf import PdfPages
	import matplotlib.pyplot as plt
	from scipy.interpolate import griddata
	import multiprocessing as mp
	import picos as pic
	import cvxopt as cvx
except:
	pass
from define import globalvars as gv
from define import chanreps as crep
from define import qchans as qc
from define import fnames as fn

########################################## GLOBAL VARIABLES ##########################################

# Metrics, thier names and the associated functions
# Usage: Metrics[<metname>][0]: Metric full name
#						   [1]: Metric label in tex format
#						   [2]: Marker assigned for the metric in the plots
#						   [3]: Name of the function function that will compute the metric
Metrics = {'dnorm':["Diamond norm",
					"$|| \\mathcal{E} - \\mathsf{id} ||_{\\diamondsuit}$",
					u'+',
					'crimson',
					"See Sec. 4 of DOI: 10.4086/toc.2009.v005a011.",
					"lambda J, ch: DiamondNorm(J, ch)"],
			'errp':["Error probability",
					"$p_{err}(\\mathcal{J})$",
					u'*', 'darkblue',
					"1 - p* where p* is the maximum value between 0 and 1 such that J - (p*)*J_id is a valid choi matrix, up to normalization.",
					"lambda J, ch: ErrorProbability(J, ch)"],
			'entropy':["Entropy",
					   "$S(\\mathcal{J})$",
					   u'p',
					   'brown',
					   "Von-Neumann entropy of the channel\'s Choi matrix.",
					   "lambda J, ch: Entropy(J, ch)"],
			'infid':["Infidelity",
						"$1 - F(\\mathcal{J})$",
						u's',
						'forestgreen',
						"1 - Fidelity between the input Choi matrix and the Choi matrix corresponding to the identity state.",
						"lambda J, ch: Infidelity(J, ch)"],
			'np1':["Non Pauliness (Chi)",
				   "$\\mathcal{W}(\\mathcal{E})$",
				   u'v',
				   'lavender',
				   "L2 norm of the difference between the channel\'s Chi matrix and it's twirled approximation.",
				   "lambda J, ch: NonPaulinessChi(J, ch)"],
			'np2':["Non Pauliness (max Fidelity)",
				   "$np_2(\\mathcal{J})$",
				   u'1',
				   'maroon',
				   "least fidelity between the channel\'s Choi matrix and a bell state.",
				   "lambda J, ch: NonPaulinessChoi(J, ch)"],
			'np4':["Non Pauliness (closest Pauli)",
				   "$np_4(\\mathcal{J})$",
				   u'3',
				   'turquoise',
				   "Maximum \"amount\" of Pauli channel that can be subtracted from the input Pauli channel, such that what remains is still a valid quantum channel.",
				   "lambda J, ch: NonPaulinessRemoval(J, ch)"],
			'trn':["Trace norm",
				   "$\\left|\\left|\\mathcal{J} - \\mathsf{id}\\right|\\right|_{1}$",
				   u'8',
				   'black',
				   "Trace norm of the difference between the channel\'s Choi matrix and the input Bell state, Trace norm of A is defined as: Trace(Sqrt(A^\\dagger . A)).",
				   "lambda J, ch: TraceNorm(J, ch)"],
			'frb':["Frobenious norm",
				   "$\\left|\\left|\\mathcal{J} - \\mathsf{id}\\right|\\right|_{2}$",
				   u'd',
				   'chocolate',
				   "Frobenious norm of the difference between the channel\'s Choi matrix and the input Bell state, Frobenious norm of A is defined as: Sqrt(Trace(A^\\dagger . A)).",
				   "lambda J, ch: FrobeniousNorm(J, ch)"],
			'bd':["Bures distance",
				  "$\\Delta_{B}(\\mathcal{J})$",
				  u'<',
				  'goldenrod',
				  "Bures distance between the channel\'s Choi matrix and the input Bell state. Bures distance between A and B is defined as: sqrt( 2 - 2 * sqrt( F ) ), where F is the Uhlmann-Josza fidelity between A and B.",
				  "lambda J, ch: BuresDistance(J, ch)"],
			'uhl':["Uhlmann Fidelity",
				   "$1 - F_{U}(\\mathcal{J})$",
				   u'h',
				   'midnightblue',
				   "1 - Uhlmann-Jozsa fidelity between the channel\'s Choi matrix and the input Bell state. The Uhlmann-Jozsa fidelity between A and B is given by: ( Trace( sqrt( sqrt(A) B sqrt(A) ) ) )^2.",
				   "lambda J, ch: UhlmanFidelity(J, ch)"],
			'unitarity':["NonUnitarity",
						 "$1-\\mathcal{u}(\\mathcal{E})$",
						 u'^',
						 'fuchsia',
						 "In the Pauli-Liouville representation of the channel, P, the unitarity is given by: ( sum_(i,j; i not equal to j) |P_ij|^2 ).",
						 "lambda J, ch: NonUnitarity(J, ch)"]
			}
######################################################################################################

def HermitianConjugate(mat):
	# Return the Hermitian conjugate of a matrix
	return np.conjugate(np.transpose(mat))

def DiamondNorm(choi, channel = "unknown"):
	# computes the diamond norm of the difference between an input Channel and another reference channel, which is by default, the identity channel
	# The semidefinite program outlined in Sec. 4 of DOI: 10.4086/toc.2009.v005a011 is used here.
	# See also: https://github.com/BBN-Q/matlab-diamond-norm/blob/master/src/dnorm.m
	# For some known types of channels, the Diamond norm can be computed efficiently
	# 1. Depolarizing channel
	if (channel == "dp"):
		# The choi matrix of the channel is in the form
		# 1/2-p/3	0		0	1/2-(2 p)/3
		# 0			p/3		0	0
		# 0			0		p/3	0
		# 1/2-(2 p)/3	0		0	1/2-p/3
		# and it's Diamond norm is p, in other words, it is 3 * Choi[1,1]/3
		dnorm = 3 * np.real(choi[1,1])
	# 2. Rotation about the Z axis
	elif (channel == "broken"):
		# The Choi matrix of the Rotation channel is in the form
		# 1/2 							0 	0 	(cos(2 t) - i sin(2 t))/2
		# 0 							0 	0 	0
		# 0 							0 	0 	0
		# (cos(2 t) + i sin(2 t))/2 	0 	0 	1/2
		# and its diamond norm is sin(t).
		if (np.real(choi[3, 0]) <= 0.0):
			angle = np.pi - np.arcsin(2 * np.imag(choi[3, 0]))
		else:
			angle = np.arcsin(2 * np.imag(choi[3, 0]))
		dnorm = np.abs(np.sin(angle/np.float(2)))
		# print("channel\n%s\n2 t = %g and dnorm = %g" % (np.array_str(choi, max_line_width = 150), (np.pi - np.arcsin(2 * np.imag(choi[3, 0]))), dnorm))
	else:
		# print("Function: dnorm")
		diff = (choi - gv.bell[0, :, :]).astype(complex)
		#### picos optimization problem
		prob = pic.Problem()
		# variables and parameters in the problem
		J = pic.new_param('J', cvx.matrix(diff))
		rho = prob.add_variable('rho', (2, 2), 'hermitian')
		W = prob.add_variable('W', (4, 4), 'hermitian')
		# objective function (maximize the hilbert schmidt inner product -- denoted by '|'. Here A|B means trace(A^\dagger * B))
		prob.set_objective('max', J | W)
		# adding the constraints
		prob.add_constraint(W >> 0)
		prob.add_constraint(rho >> 0)
		prob.add_constraint(('I' | rho) == 1)
		prob.add_constraint((W - ((rho & 0) // (0 & rho))) << 0)
		# solving the problem
		sol = prob.solve(verbose = 0, maxit = 500)
		dnorm = sol['obj']*2
		# print("SDP dnorm = %g" % (dnorm))
	return dnorm


def ErrorProbability(choi, channel = "unknown"):
	# compute the error probability associated with the Channel whose jamilowski form is given
	# The error probability is defined as: 1 - p* where p* is the maximum value between 0 and 1 such that J - (p*)*J_id is a valid choi matrix, up to normalization.
	#### picos optimization problem
	prob = pic.Problem()
	# variables and parameters in the problem
	Jid = pic.new_param('Jid', cvx.matrix(gv.bell[0, :, :]))
	J = pic.new_param('J', cvx.matrix(choi))
	p = prob.add_variable('p', 1)
	# adding the constraints
	prob.add_constraint(p <= 1)
	prob.add_constraint(p >= 0)
	prob.add_constraint(J - p*Jid >> 0)
	# objective function
	prob.set_objective('max', p)
	# solving the problem
	prob.solve(verbose = 0, maxit = 10)
	errp = 1 - prob.obj_value()
	return errp


def Entropy(choi, channel = "unknown"):
	# Compute the Von-Neumann entropy of the input Choi matrix.
	# The idea is that a pure state (which corresponds to unitary channels) will have zero entropy while any mixed state which corresponds to a channel that does not preserve the input state, has finiste entropy.
	sgvals = np.linalg.svd(choi.astype(np.complex), compute_uv = 0)
	entropy = 0
	for i in range(sgvals.shape[0]):
		if (abs(np.imag(sgvals[i])) < 10E-50):
			if (sgvals[i] > 0):
				entropy = entropy - sgvals[i] * np.log(sgvals[i])
	return entropy


def Infidelity(choi, channel = "unknown"):
	# Compute the Fidelity between the input Choi matrix and the Choi matrix corresponding to the identity state.
	fidelity = (1/np.longdouble(2)) * np.longdouble(np.real(choi[0, 0] + choi[3, 0] + choi[0, 3] + choi[3, 3]))
	# print("Infidelity for\n%s\n is %g." % (np.array_str(choi), 1 - fidelity))
	return (1 - fidelity)


def TraceNorm(choi, channel = "unknown"):
	# Compute the trace norm of the difference between the input Choi matrix and the Choi matrix corresponding to the Identity channel
	# trace norm of A is defined as: Trace(Sqrt(A^\dagger . A))
	# https://quantiki.org/wiki/trace-norm
	trnorm = np.linalg.norm((choi - gv.bell[0, :, :]).astype(np.complex), ord = "nuc")
	return trnorm


def FrobeniousNorm(choi, channel = "unknown"):
	# Compute the Frobenious norm of the difference between the input Choi matrix and the Choi matrix corresponding to the Identity channel
	# Frobenious of A is defined as: sqrt(Trace(A^\dagger . A))
	# https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
	# Optimized in C
	frob = np.linalg.norm((choi - gv.bell[0, :, :]).astype(np.complex), ord = "fro")
	return frob


def BuresDistance(choi, channel = "unknown"):
	# Compute the Bures distance between the input Choi matrix and the Choi matrix corresponding to the identity channel
	# The Bures distance between A and B is defined as: sqrt( 2 - 2 * sqrt( F ) ), where F is the Uhlmann-Josza fidelity between A and B
	# http://iopscience.iop.org/article/10.1088/1751-8113/40/37/010/meta
	# https://quantiki.org/wiki/fidelity
	fiduj = UhlmanFidelity(choi)
	if (fiduj < -10E-50):
		distB = 0
	else:
		distB = np.sqrt(2 - 2 * np.sqrt(1 - fiduj))
	return distB


def UhlmanFidelity(choi, channel = "unknown"):
	# Compute the Uhlmann-Josza fidelity between the input Choi matrix and the Choi matrix corresponding to the identity channel
	# The Uhlmann fidelity between A and B is given by: ( Trace( sqrt( sqrt(A) B sqrt(A) ) ) )^2
	# https://www.icts.res.in/media/uploads/Old_Talks_Lectures/Slides/1266647057kzindia10.pdf
	# http://www.sciencedirect.com/science/article/pii/S0375960101006405
	(eigvals, eigvecs) = np.linalg.eigh(choi.astype(np.complex))
	overlap = np.zeros((4, 4), dtype = np.double)
	for i in range(eigvals.shape[0]):
		for j in range(eigvals.shape[0]):
			outer = np.dot(eigvecs[i, :, np.newaxis], eigvecs[np.newaxis, j, :])
			overlap = overlap + np.sqrt(eigvals[i] * eigvals[j]) * np.trace(np.dot(gv.bell[0, :, :], outer)) * outer
	eigvals = np.linalg.eigvals(np.real(overlap))
	fiduj = np.sum(np.sqrt(eigvals), dtype = np.longdouble) * np.sum(np.sqrt(eigvals), dtype = np.longdouble)
	return (1 - fiduj)


def NonUnitarity(choi, channel = "unknown"):
	# Compute the unitarity metric for the input choi matrix of a channel
	# Convert from the Choi matrix to the process matrix
	# For a process matrix P, the unitarity is given by: ( sum_(i,j; i not equal to j) |P_ij|^2 )
	# http://iopscience.iop.org/article/10.1088/1367-2630/17/11/113020
	process = crep.ConvertRepresentations(choi, "choi", "process")
	unitarity = np.sum(np.abs(process[1:, 1:]) * np.abs(process[1:, 1:]))/np.longdouble(3)
	return (1 - unitarity)


def NonPaulinessChi(choi, channel = "unknown"):
	# Quantify the behaviour of a quantum channel by its difference from a Pauli channel
	# Convert the input Choi matrix to it's Chi-representation
	# Compute the ration between the  sum of offdiagonal entries to the sum of disgonal entries.
	# While computing the sums, consider the absolution values of the entries.
	chi = crep.ConvertRepresentations(choi, "choi", "chi")
	# print("chi\n%s" % (np.array_str(chi, max_line_width=150)))
	atol = 10E-20
	nonpauli = 0.0
	for i in range(4):
		for j in range(4):
			if (not (i == j)):
				if (np.abs(chi[i, i]) * np.abs(chi[j, j]) >= atol):
					# print("contribution = %g" % (np.power(np.abs(chi[i, j]), 2.0)/(np.abs(chi[i, i]) * np.abs(chi[j, j]))))
					nonpauli = nonpauli + np.power(np.abs(chi[i, j]), 2.0)/(np.abs(chi[i, i]) * np.abs(chi[j, j]))
	# print("nonpauli = %g." % (nonpauli))
	return nonpauli


def NonPaulinessChoi(choi, channel = "unknown"):
	# Quantify the behaviour of a quantum channel by its difference from a Pauli channel
	# Compute the least fidelity between the input Choi matrix and any of the bell states.
	overlaps = map(np.abs, map(np.trace, np.tensordot(gv.bell, choi, axes = [[1],[1]])))
	nonpauli = 1 - max(overlaps)
	return nonpauli


def NonPaulinessRemoval(choi, channel = "unknown"):
	# Quantify the behaviour of a quantum channel by its difference from a Pauli channel
	# Pauliness is defined as the maximum "amount" of Pauli channel that can be subtracted from the input Pauli channel, such that what remains is still a valid quantum channel.
	bellstates = np.zeros((4, 4, 4), dtype = np.complex128)
	# Bell state |00> + |11>
	# X -- Bell state |01> + |10>
	# Y -- Bell state i(|10> - |01>)
	# Z -- Bell state |00> - |11>
	
	### picos optimization problem
	prob = pic.Problem()
	# parameters and variables
	J = pic.new_param('J', cvx.matrix(choi))
	JI = pic.new_param('JI', cvx.matrix(gv.bell[0, :, :]))
	JX = pic.new_param('JX', cvx.matrix(gv.bell[1, :, :]))
	JY = pic.new_param('JY', cvx.matrix(gv.bell[2, :, :]))
	JZ = pic.new_param('JZ', cvx.matrix(gv.bell[3, :, :]))
	pp = prob.add_variable('pp', 4)
	p = prob.add_variable('p')
	# specifying constraints
	# probabilities sum to 1. Each is bounded above by 1, below by 0.
	for i in range(4):
		prob.add_constraint(pp[i] >= 0)
		prob.add_constraint(pp[i] <= 1)
	prob.add_constraint(np.sum(pp) == 1)
	# Fraction of Pauli channel that can to be removed
	prob.add_constraint(p >= 0)
	prob.add_constraint(p <= 1)
	# What remains after subtracting a Pauli channel is a valid Choi matrix, up to normalization
	prob.add_constraint(JI*pp[0] + JX*pp[1] + JY*pp[2] + JZ*pp[3] - p*J >> 0)
	# objective function --- maximize the sum of Probabilities of I, X, Y and Z errors
	prob.set_objective('max', p)
	# Solve the problem
	sol = prob.solve(verbose = 0, maxit = 100)
	nonPauli = 1 - sol['obj']
	return nonPauli


def Filter(process, channelType, metric, lower, upper):
	# Test if a channel (described by the process matrix) passes across a filter or not.
	metVal = eval(Metrics[metric][-1])(crep.ConvertRepresentations(process, 'process', 'choi'), channelType)
	if ((metVal >= lower) and (metVal <= upper)):
		return 1
	return 0


def GenCalibrationData(chname, channels, noiserates, metrics):
	# Given a set of channels and metrics, compute each metric for every channel and save the result as a 2D array.
	# The input array "channels" has size (number of channels) x 4 x 4, where A[i, :, :] is generated from m free variables.
	# The output array is a 2D array of size (number of channels) x (1 + number of free variables in the channel + number of metrics).
	# output[0, 0] = number of free variables in every channel
	# {output[i + 1, 0], ..., output[i + 1, m - 1]} = values for the free variables used to specify the i-th channel
	# {output[i + 1, m], ..., output[i + 1, m + n - 1]} = values for the n metrics on the i-th channel.
	if (not (os.path.exists("./../temp"))):
		os.system("mkdir -p ./../temp")
	for m in range(len(metrics)):
		calibdata = np.zeros((channels.shape[0], 1 + noiserates.shape[1]), dtype = np.longdouble)
		for i in range(channels.shape[0]):
			calibdata[i, :noiserates.shape[1]] = noiserates[i, :]
			calibdata[i, noiserates.shape[1]] = eval(Metrics[metrics[m]][-1])(channels[i, :, :], chname)
		# Save the calibration data
		np.savetxt(fn.CalibrationData(chname, metrics[m]), calibdata)
	return None

def ComputeNorms(channel, metrics, name = "unknown"):
	# Compute a set of metrics for a channel (in the choi matrix form) and return the metric values
	# print("Function ComputeNorms(\n%s,\n%s)" % (np.array_str(channel, max_line_width = 150, precision = 3), metrics))
	mets = np.zeros(len(metrics), dtype = np.longdouble)
	for m in range(len(metrics)):
		mets[m] = eval(Metrics[metrics[m]][-1])(channel, name)
	return mets

def ChannelMetrics(submit, physmetrics, start, end, results, rep):
	# Compute the various metrics for all channels with a given noise rate
	for i in range(start, end):
		physical = np.load(fn.PhysicalChannel(submit, submit.available[i, :-1]))[int(submit.available[i, -1]), :, :]
		# print("Channel %d: Function ComputeNorms(\n%s,\n%s)" % (i, np.array_str(physical, max_line_width = 150, precision = 3), physmetrics))
		if (not (rep == "choi")):
			physical = crep.ConvertRepresentations(physical, "process", "choi")
		for m in range(len(physmetrics)):
			results[i * len(physmetrics) + m] = eval(Metrics[physmetrics[m]][-1])(physical, submit.channel)
			# print("%g" % (results[i * len(physmetrics) + m]))
	return None

def ComputePhysicalMetrics(submit, physmetrics, ncpu = 4, loc = "local"):
	# Compute metrics for all physical channels in a submission.
	ncpu = 1
	nproc = min(ncpu, mp.cpu_count())
	chunk = int(np.ceil(submit.channels/np.float(nproc)))
	processes = []
	results = mp.Array(ct.c_longdouble, submit.channels * len(physmetrics))
	for i in range(nproc):
		processes.append(mp.Process(target = ChannelMetrics, args = (submit, physmetrics, i * chunk, min(submit.channels, (i + 1) * chunk), results, "process")))
	for i in range(nproc):
		processes[i].start()
	for i in range(nproc):
		processes[i].join()
	
	metvals = np.reshape(results, [len(physmetrics), submit.channels], order = 'c')
	# print("Metric values for metric = %s\n%s" % (physmetrics, np.array_str(metvals)))
	# Write the physical metrics on to a file
	for m in range(len(physmetrics)):
		np.save(fn.PhysicalErrorRates(submit, physmetrics[m]), metvals[m, :])
	return None


def PlotCalibrationData1D(chname, metrics, xcol = 0):
	# The calibration data for every metric is an array of size m that contains a metric value for every channel.
	# "noiserates" is a 2D array of size (number of channels) x m that contains the parameter combinations corresponding to every channel index.
	# "xcol" indicates the free variable of the noise model that distinguishes various channels in the plot. (This is the X-axis.)
	# All metrics will be in the same plot.
	fig = plt.figure(figsize = (gv.canvas_size[0] * 1.2, gv.canvas_size[1] * 1.2))
	plt.title("%s" % (qc.Channels[chname][0]), fontsize = gv.title_fontsize, y = 1.01)
	for m in range(len(metrics)):
		calibdata = np.loadtxt(fn.CalibrationData(chname, metrics[m]))
		# print("calibdata\n%s" % (np.array_str(calibdata)))
		plt.plot(calibdata[:, xcol], calibdata[:, -1], label = Metrics[metrics[m]][1], marker = Metrics[metrics[m]][2], color = Metrics[metrics[m]][3], markersize = gv.marker_size + 5, linestyle = "-", linewidth = 7.0)
	ax = plt.gca()
	ax.set_xlabel(qc.Channels[chname][2][xcol], fontsize = gv.axes_labels_fontsize + 20)
	# ax.set_xscale('log')
	ax.set_ylabel("$\\mathcal{N}_{0}$", fontsize = gv.axes_labels_fontsize + 20)
	# ax.set_yscale('log')
	# separate the axes labels from the plot-frame
	ax.tick_params(axis = 'both', direction = "inout", which = 'both', pad = gv.ticks_pad, labelsize = gv.ticks_fontsize + 10, length = gv.ticks_length, width = gv.ticks_width)
	# Legend
	plt.legend(numpoints = 1, loc = 4, shadow = True, fontsize = gv.legend_fontsize, markerscale = gv.legend_marker_scale)
	# Save the plot
	plt.savefig(fn.CalibrationPlot(chname, "_".join(metrics)))
	plt.close()
	return None


def PlotCalibrationData2D(chname, metrics, xcol = 0, ycol = 1):
	# Plot performance contours for various noise strength values, with repect to the physical noise parameters.
	plotfname = fn.CalibrationPlot(chname, "_".join(metrics))
	
	with PdfPages(plotfname) as pdf:
		for m in range(len(metrics)):
			calibdata = np.loadtxt(fn.CalibrationData(chname, metrics[m]))
			(meshX, meshY) = np.meshgrid(np.linspace(calibdata[:, xcol].min(), calibdata[:, xcol].max(), max(10, calibdata.shape[0])), np.linspace(calibdata[:, ycol].min(), calibdata[:, ycol].max(), max(10, calibdata.shape[0])))
			meshZ = griddata((calibdata[:, xcol], calibdata[:, ycol]), calibdata[:, -1], (meshX, meshY), method = "cubic")
			# Contour Plot
			fig = plt.figure(figsize = gv.canvas_size)
			# Data points
			cplot = plt.contourf(meshX, meshY, meshZ, cmap = cm.winter, locator = ticker.LogLocator(), linestyles = gv.contour_linestyle)
			plt.scatter(calibdata[:, xcol], calibdata[:, ycol], marker = 'o', color = 'k')
			plt.title("%s channel" % (chname), fontsize = gv.title_fontsize, y = 1.03)
			ax = plt.gca()
			ax.set_xlabel(qc.Channels[chname][2][xcol], fontsize = gv.axes_labels_fontsize)
			ax.set_ylabel(qc.Channels[chname][2][ycol], fontsize = gv.axes_labels_fontsize)
			ax.tick_params(axis = 'both', which = 'both', pad = gv.ticks_pad, direction = 'inout', length = gv.ticks_length, width = gv.ticks_width, labelsize = gv.ticks_fontsize)
			# Legend
			cbar = plt.colorbar(cplot, extend = "both", spacing = "proportional", drawedges = False)
			cbar.ax.set_xlabel(Metrics[metrics[m]][1], fontsize = gv.colorbar_fontsize)
			cbar.ax.tick_params(labelsize = gv.legend_fontsize, pad = gv.ticks_pad, length = gv.ticks_length, width = gv.ticks_width)
			cbar.ax.xaxis.labelpad = gv.ticks_pad
			# Save the plot
			pdf.savefig(fig)
			plt.close()
		#Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo['Title'] = ("%s for %d %s channels." % (",".join(metrics), calibdata.shape[0], chname))
		pdfInfo['Author'] = "Pavithran Iyer"
		pdfInfo['ModDate'] = dt.datetime.today()
	return None