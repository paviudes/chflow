import os
import sys
import datetime as dt
import numpy as np
try:
	import matplotlib
	matplotlib.use("Agg")
	from matplotlib.backends.backend_pdf import PdfPages
	import matplotlib.pyplot as plt
except Exception:
	sys.stderr.write("\033[91m\033[2mMATPLOTLIB does not exist, cannot make plots.\n\033[0m")
# Force the module scripts to run locally -- https://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path
import inspect as ins
current = os.path.realpath(os.path.abspath(os.path.dirname(ins.getfile(ins.currentframe()))))
if (not (current in sys.path)):
	sys.path.insert(0, current)

from define import fnames as fn
from define import globalvars as gv
from define import metrics as ml


def ThresholdPlot(submit, phymet, logmet, maxlevel = 3):
	# Plot the logical error rate data that has already been collected.
	logErrorRates = np.load(fn.LogicalErrorRates(submit, logmet, fmt = "npy"))
	physErrorRates = np.load(fn.PhysicalErrorRates(submit, phymet))
	# print("Logical error rates\n%s" % (np.array_str(logErrorRates)))
	# print("Physical error rates\n%s" % (np.array_str(physErrorRates)))
	plotfname = fn.ThreshPlot(submit, phymet, logmet)
	with PdfPages(plotfname) as pdf:
		fig = plt.figure(figsize = gv.canvas_size)
		nqubits = 1
		dist = 1
		for l in range(submit.levels):
			nqubits = nqubits * submit.eccs[l].N
			dist = dist * submit.eccs[l].D
			plt.plot(physErrorRates, logErrorRates[:, l + 1], label = ("N = %d, d = %d" % (nqubits, dist)), color = ml.Metrics[ml.Metrics.keys()[l]][3], marker = ml.Metrics[ml.Metrics.keys()[l]][2], markersize = gv.marker_size, linestyle = '--', linewidth = gv.line_width)
		# Legend
		plt.legend(numpoints = 1, loc = 4, shadow = True, fontsize = gv.legend_fontsize, markerscale = gv.legend_marker_scale)
		# Axes labels
		ax = plt.gca()
		ax.set_xlabel(ml.Metrics[phymet][1], fontsize = gv.axes_labels_fontsize)
		ax.set_xscale('log')
		ax.set_ylabel(ml.Metrics[logmet][1], fontsize = gv.axes_labels_fontsize)
		ax.set_yscale('log')
		ax.tick_params(axis = 'both', which = 'major', pad = gv.ticks_pad, direction = 'inout', length = gv.ticks_length, width = gv.ticks_width, labelsize = gv.ticks_fontsize)
		# Save the plot
		pdf.savefig(fig)
		plt.close()
		#Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo['Title'] = ("%s at levels %s, with physical %s for %d channels." % (ml.Metrics[logmet][0], ", ".join(map(str, range(1, 1 + submit.levels))), ml.Metrics[phymet][0], submit.channels))
		pdfInfo['Author'] = "Pavithran Iyer"
		pdfInfo['ModDate'] = dt.datetime.today()
	return None


def LevelWisePlot(submit, refs, phymet, logmet, maxlevel = 3):
	# Plot the logical error rate data that has already been collected.
	logErrorRates = np.load(fn.LogicalErrorRates(submit, logmet, fmt = "npy"))
	physErrorRates = np.load(fn.PhysicalErrorRates(submit, phymet))
	# print("Logical error rates\n%s" % (np.array_str(logErrorRates)))
	# print("Physical error rates\n%s" % (np.array_str(physErrorRates)))
	refLogErrRates = []
	refPhysErrRates = []
	for i in range(len(refs)):
		refLogErrRates.append(np.load(refs[i].LogicalErrorRates(submit, logmet, fmt = "npy")))
		refPhysErrRates.append(np.load(refs[i].PhysicalErrorRates(submit, phymet)))
	plotfname = fn.LevelWise(submit, phymet, logmet)
	with PdfPages(plotfname) as pdf:
		for l in range(submit.levels):
			fig = plt.figure(figsize = gv.canvas_size)
			plt.plot(physErrorRates, logErrorRates[:, l + 1], label = ("%s" % (submit.channel)), color = ml.Metrics[phymet][3], marker = ml.Metrics[phymet][2], markersize = gv.marker_size, linestyle = 'None')
			for i in range(len(refs)):
				plt.plot(refPhysErrRates[i], refLogErrRates[i][:, l + 1], color = qch.Channels[refs[i].channel][2], label = ("%s" % (refs[i].channel)), marker = ml.Metrics[phymet][2], markersize = gv.marker_size, linestyle = "-", linewidth = gv.line_width)
			# Legend
			plt.legend(numpoints = 1, loc = 4, shadow = True, fontsize = gv.legend_fontsize, markerscale = gv.legend_marker_scale)
			# Axes labels
			ax = plt.gca()
			ax.set_xlabel(("$\\mathcal{N}_{0}$  $\\left(%s\\right)$" % (ml.Metrics[phymet][1].replace("$", ""))), fontsize = gv.axes_labels_fontsize)
			ax.set_xscale('log')
			ax.set_ylabel(("$\\mathcal{N}_{%d}$  $\\left(%s\\right)$" % (l + 1, ml.Metrics[logmet][1].replace("$", ""))), fontsize = gv.axes_labels_fontsize)
			ax.set_yscale('log')
			ax.tick_params(axis = 'both', which = 'major', pad = gv.ticks_pad, direction = 'inout', length = gv.ticks_length, width = gv.ticks_width, labelsize = gv.ticks_fontsize)
			# Save the plot
			pdf.savefig(fig)
			plt.close()
		#Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo['Title'] = ("%s at levels %s, with physical %s for %d channels." % (ml.Metrics[logmet][0], ", ".join(map(str, range(1, 1 + submit.levels))), ml.Metrics[phymet][0], submit.channels))
		pdfInfo['Author'] = "Pavithran Iyer"
		pdfInfo['ModDate'] = dt.datetime.today()
	return None


def CompareSubs(dbses, logmet):
	# Compare the Logical error rates from two submissions.
	# The comparision only makes sense when the logical error rates are measured for two submissions that have the same physical channels.
	compdepth = min([dbses[i].levels for i in range(2)])
	logErrorRates = np.zeros((2, dbs1.channels, compdepth), dtype = np.longdouble)
	for i in range(2):
		logErrorRates[i, :, :] = np.load(fn.LogicalErrorRates(dbses[i], logmet))
	plotfname = fn.CompareLogErrRates(dbses, logmet)
	with PdfPages(plotfname) as pdf:
		fig = plt.figure(figsize = gv.canvas_size)
		for l in range(compdepth):
			plt.plot(logErrorRates[0, :, l + 1], logErrorRates[1, :, l + 1], color = ml.Metrics[logmet][3], marker = ml.Metrics[logmet][2], markersize = gv.marker_size, linestyle = 'None')
		# Axes labels
		ax = plt.gca()
		ax.set_xlabel(("$\\mathcal{N}_{%d}$  $\\left(%s\\right)$ for %s" % (l + 1, ml.Metrics[logmet][1].replace("$", "")), dbses[0].ecc), fontsize = gv.axes_labels_fontsize)
		ax.set_xscale('log')
		ax.set_xlabel(("$\\mathcal{N}_{%d}$  $\\left(%s\\right)$ for %s" % (l + 1, ml.Metrics[logmet][1].replace("$", "")), dbses[1].ecc), fontsize = gv.axes_labels_fontsize)
		ax.set_yscale('log')
		ax.tick_params(axis = 'both', which = 'major', pad = gv.ticks_pad, direction = 'inout', length = gv.ticks_length, width = gv.ticks_width, labelsize = gv.ticks_fontsize)
		# Save the plot
		pdf.savefig(fig)
		plt.close()
		#Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo['Title'] = ("Comparison of logical error rates (%s) for %s and %s at levels %s for %d channels." % (ml.Metrics[logmet][0], dbses[0].ecc, dbses[1].ecc, ", ".join(map(str, range(1, 1 + compdepth))), dbses[0].channels))
		pdfInfo['Author'] = "Pavithran Iyer"
		pdfInfo['ModDate'] = dt.datetime.today()
	return None