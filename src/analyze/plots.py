import os
import sys
import datetime as dt
try:
	import numpy as np
	import matplotlib
	from matplotlib import colors, ticker, cm
	matplotlib.use("Agg")
	from matplotlib.backends.backend_pdf import PdfPages
	import matplotlib.pyplot as plt
	from scipy.interpolate import griddata
except:
	pass
# Force the module scripts to run locally -- https://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path
# import inspect as ins
# current = os.path.realpath(os.path.abspath(os.path.dirname(ins.getfile(ins.currentframe()))))
# if (not (current in sys.path)):
# 	sys.path.insert(0, current)

from define import qchans as qc
from define import fnames as fn
from define import globalvars as gv
from define import metrics as ml


def IsNumber(numorstr):
	# test if the input is a number.
	try:
		float(numorstr)
		return 1
	except:
		return 0


def latex_float(f):
	# Function taken from: https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python
	float_str = "{0:.2g}".format(f)
	if "e" in float_str:
		base, exponent = float_str.split("e")
		return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
	else:
		return float_str


def DisplayForm(number, base):
	# Express a float number in a scientific notation with two digits after the decimal.
	# N = A b^x where A < b. Then, log N/(log b) = log A/(log b) + x.
	# So, x = floor(log N/(log b)) and log A/(log b) = log N/(log b) - x.
	# then, A = b^(log N/(log b) - x)
	# print("number = %g, base = %g" % (number, base))
	numstr = "0"
	if (number == 0):
		return numstr	
	exponent = np.floor(np.log(number)/np.log(base))
	factor = np.int(np.power(base, np.log(number)/np.log(base) - exponent) * 100)/np.float(100)
	numstr = ("$%g \\times %g^{%d}$" % (factor, base, exponent))
	if (number < 0):
		numstr = ("-%s" % (numstr))
	return numstr


def ThresholdPlot(phymets, logmet, dbs):
	# For each physical noise rate, plot the logical error rate vs. the levels of concatenation.
	# The set of curves should have a bifurcation at the threshold.
	phylist = map(lambda phy: phy.strip(" "), phymets.split(","))
	sampreps = np.hstack((np.nonzero(dbs.available[:, -1] == 0)[0], [dbs.channels]))
	logErrs = np.load(fn.LogicalErrorRates(dbs, logmet, fmt = "npy"))
	logErr = np.zeros((sampreps.shape[0], dbs.levels + 1), dtype = np.longdouble)
	for i in range(sampreps.shape[0] - 1):
		for l in range(dbs.levels + 1):
			logErr[i, l] = np.sum(logErrs[sampreps[i]:sampreps[i + 1], l], dtype = np.longdouble)/np.longdouble(sampreps[i + 1] - sampreps[i])
	# print("sampreps\n%s" % (np.array_str(sampreps)))
	phyerrs = np.zeros((sampreps.shape[0], len(phylist)), dtype = np.longdouble)
	phyparams = []
	for m in range(len(phylist)):
		if (IsNumber(phylist[m])):
			# If phylist[m] is a number, then it indicates an independent parameter of the channel to serve as a measure of the physical noise strength
			for i in range(sampreps.shape[0] - 1):
				phyerrs[i, m] = np.sum(dbs.available[sampreps[i]:sampreps[i + 1], np.int8(phylist[m])], dtype = np.longdouble)/np.longdouble(sampreps[i + 1] - sampreps[i])
			phyparams.append(qc.Channels[dbs.channel][2][np.int8(phylist[m])])
		else:
			# print("loading: %s" % (fn.PhysicalErrorRates(dbs, phylist[m])))
			phyrates = np.load(fn.PhysicalErrorRates(dbs, phylist[m]))
			# print("metric = %s, phyrates\n%s" % (phylist[m], np.array_str(phyrates)))
			for i in range(sampreps.shape[0] - 1):
				# print("phyrates[%d:%d, np.int8(phylist[m])]\n%s" % (sampreps[i], sampreps[i + 1], np.array_str(phyrates[sampreps[i]:sampreps[i + 1]])))
				phyerrs[i, m] = np.sum(phyrates[sampreps[i]:sampreps[i + 1]], dtype = np.longdouble)/np.longdouble(sampreps[i + 1] - sampreps[i])
			phyparams.append(ml.Metrics[phylist[m]][1])
	plotfname = fn.ThreshPlot(dbs, "_".join(phylist), logmet)
	with PdfPages(plotfname) as pdf:
		for m in range(len(phylist)):
			fig = plt.figure(figsize = gv.canvas_size)
			for p in range(phyerrs.shape[0]):
				fmtidx = ml.Metrics.keys()[p % len(ml.Metrics)]
				plt.plot(np.arange(dbs.levels + 1), logErr[p, :], label = ("%s = %s" % (phyparams[m], DisplayForm(phyerrs[p, m], 10))), color = ml.Metrics[fmtidx][3], marker = ml.Metrics[fmtidx][2], markersize = gv.marker_size, linestyle = '--', linewidth = gv.line_width)
			# Legend
			plt.legend(numpoints = 1, loc = 4, shadow = True, fontsize = gv.legend_fontsize, markerscale = gv.legend_marker_scale)
			# Title
			plt.title("Threshold of %s channel in %s." % (qc.Channels[dbs.channel][0], phyparams[m]), fontsize = gv.title_fontsize, y = 1.03)
			# Axes labels
			ax = plt.gca()
			xlabels = np.cumprod([dbs.eccs[i].D for i in range(dbs.levels)])
			ax.set_xlabel("Concatenation levels", fontsize = gv.axes_labels_fontsize)
			ax.set_ylabel(ml.Metrics[logmet][1], fontsize = gv.axes_labels_fontsize)
			ax.set_yscale('log')
			ax.tick_params(axis = 'both', which = 'major', pad = gv.ticks_pad, direction = 'inout', length = gv.ticks_length, width = gv.ticks_width, labelsize = gv.ticks_fontsize)
			# Save the plot
			pdf.savefig(fig)
			plt.close()
		#Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo['Title'] = ("%s at levels %s, with physical %s for %d channels." % (ml.Metrics[logmet][0], ", ".join(map(str, range(1, 1 + dbs.levels))), ", ".join(phyparams), dbs.channels))
		pdfInfo['Author'] = "Pavithran Iyer"
		pdfInfo['ModDate'] = dt.datetime.today()
	return None


def LevelWisePlot(phymets, logmet, dbses):
	# Plot logical error rates vs. physical error rates.
	# Use a new figure for every new concatenated level.
	# In each figure, each curve will represent a new physical metric.
	phylist = map(lambda phy: phy.strip(" "), phymets.split(","))
	ndb = len(dbses)
	maxlevel = max([dbses[i].levels for i in range(ndb)])
	plotfname = fn.LevelWise(dbses[0], "_".join(phylist), logmet)
	with PdfPages(plotfname) as pdf:
		for l in range(maxlevel):
			phlines = []
			phynames = []
			dblines = []
			dbnames = []
			fig = plt.figure(figsize = gv.canvas_size)
			for p in range(len(phylist)):
				for d in range(ndb):
					logErrs = np.load(fn.LogicalErrorRates(dbses[d], logmet, fmt = "npy"))[:, l + 1]
					if (phylist[p] in ml.Metrics):
						phymet = ml.Metrics[phylist[i]][1]
						phyerrs = np.load(fn.PhysicalErrorRates(dbses[d], phylist[p]))
						plotset = [ml.Metrics[phylist[p]][3], ml.Metrics[phylist[p]][2], ['None', '--'][dbses[d].samps == 1]]
					else:
						phymet = qc.Channels[dbses[d].channel][2][np.int8(phylist[p])]
						phyerrs = dbses[d].available[:, np.int8(phylist[p])]
						plotset = [ml.Metrics[ml.Metrics.keys()[p % len(ml.Metrics)]][3], ml.Metrics[ml.Metrics.keys()[p % len(ml.Metrics)]][2], ['None', '--'][dbses[d].samps == 1]]
					if (not (dbses[d].scales[p] == 1)):
						phyerrs = np.power(dbses[d].scales[p], phyerrs)
					# Plotting
					if (d == 0):
						plotobj = plt.plot(phyerrs, logErrs, color = plotset[0], marker = plotset[1], markersize = gv.marker_size, linestyle = plotset[2], linewidth = gv.line_width)
					else:
						plotobj = plt.plot(phyerrs, logErrs, color = dbses[d].plotsettings[0], marker = dbses[d].plotsettings[1], markersize = gv.marker_size, linestyle = dbses[d].plotsettings[2], linewidth = gv.line_width)
					# if we find a new physical metric, we must add it to metric legend labels
					if (not (phymet in phynames)):
						phlines.append(plotobj[0])
						phynames.append(phymet)
					if (not (dbses[d].timestamp in [name[0] for name in dbnames])):
						dblines.append(plotobj[0])
						dbnames.append([dbses[d].timestamp, (("N = %d, D = %d, %s") % (np.prod([dbses[d].eccs[j].N for j in range(l + 1)]), np.prod([dbses[d].eccs[j].D for j in range(l + 1)]), qc.Channels[dbses[d].channel][0]))])

			# Title
			# plt.title(("%s vs. physical error metrics for the %s channel." % (ml.Metrics[logmet][0], qc.Channels[dbses[0].channel][0])), fontsize = gv.title_fontsize, y = 1.03)
			# Axes labels
			ax = plt.gca()
			ax.set_xlabel("$\\mathcal{N}_{0}$: Physical noise strength", fontsize = gv.axes_labels_fontsize)
			ax.set_xscale('log')
			ax.set_ylabel("$\\mathcal{N}_{%d}$: %s" % (l + 1, ml.Metrics[logmet][1]), fontsize = gv.axes_labels_fontsize)
			ax.set_yscale('log')
			ax.tick_params(axis = 'both', which = 'both', pad = gv.ticks_pad, direction = 'inout', length = gv.ticks_length, width = gv.ticks_width, labelsize = gv.ticks_fontsize)
			# Legend
			dblegend = plt.legend(handles = dblines, labels = [name[1] for name in dbnames], numpoints = 1, loc = 1, shadow = True, fontsize = gv.legend_fontsize, markerscale = gv.legend_marker_scale)
			plt.legend(handles = phlines, labels = phynames, numpoints = 1, loc = 4, shadow = True, fontsize = gv.legend_fontsize, markerscale = gv.legend_marker_scale)
			ax.add_artist(dblegend)
			
			# Save the plot
			pdf.savefig(fig)
			plt.close()
		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo['Title'] = ("%s at levels %s, with physical %s for %d channels." % (ml.Metrics[logmet][0], ", ".join(map(str, range(1, 1 + maxlevel))), ", ".join(phynames), dbses[0].channels))
		pdfInfo['Author'] = "Pavithran Iyer"
		pdfInfo['ModDate'] = dt.datetime.today()
	return None


def LevelWisePlot2D(phymets, logmet, dbs):
	# Plot performance contours for the logical error rates for every concatenation level, with repect to the dephasing and relaxation rates.
	# Each plot will be a contour plot or a color density plot indicating the logical error, with the x-axis as the dephasing rate and the y-axis as the relaxation rate.
	# There will be one plot for every concatenation level.
	logErr = np.load(fn.LogicalErrorRates(dbs, logmet, fmt = "npy"))
	phylist = map(lambda phy: phy.strip(" "), phymets.split(","))
	phyerrs = np.zeros((dbs.channels, len(phylist)), dtype = np.longdouble)
	plotdata = np.zeros((max(100, dbs.channels), len(phylist)), dtype = np.longdouble)
	phyparams = []
	for m in range(len(phylist)):
		if (IsNumber(phylist[m])):
			# If phylist[m] is a number, then it indicates an independent parameter of the channel to serve as a measure of the physical noise strength
			phyerrs[:, m] = dbs.available[:, np.int8(phylist[m])]
			phyparams.append(qc.Channels[dbs.channel][2][np.int8(phylist[m])])
			if (not (dbs.scales[m] == 1)):
				phyerrs[:, m] = np.power(dbs.scales[m], phyerrs[:, m])
		else:
			phyerrs[:, m] = np.load(fn.PhysicalErrorRates(dbs, phylist[m]))
			phyparams.append(ml.Metrics[phylist[m]][1])
		plotdata[:, m] = np.linspace(phyerrs[:, m].min(), phyerrs[:, m].max(), plotdata.shape[0])

	(meshX, meshY) = np.meshgrid(plotdata[:, 0], plotdata[:, 1])
	plotfname = fn.LevelWise(dbs, phymets.replace(",", "_"), logmet)
	with PdfPages(plotfname) as pdf:
		nqubits = 1
		dist = 1
		for l in range(1 + dbs.levels):
			if (l == 0):
				nqubits = 1
				dist = 1
			else:
				nqubits = nqubits * dbs.eccs[l - 1].N
				dist = dist * dbs.eccs[l - 1].D
			fig = plt.figure(figsize = gv.canvas_size)
			meshZ = griddata((phyerrs[:, 0], phyerrs[:, 1]), logErr[:, l], (meshX, meshY), method = "cubic")
			
			clevels = np.logspace(np.log10(np.abs(logErr[:, l].min())), np.log10(logErr[:, l].max()), gv.contour_nlevs, base = 10.0)
			cplot = plt.contourf(meshX, meshY, meshZ, cmap = cm.bwr, locator = ticker.LogLocator(), linestyles = gv.contour_linestyle, levels = clevels)
			plt.scatter(phyerrs[:, 0], phyerrs[:, 1], marker = 'o', color = 'k')
			# Title
			plt.title("N = %d, d = %d" % (nqubits, dist), fontsize = gv.title_fontsize, y = 1.03)
			# Axes labels
			ax = plt.gca()
			ax.set_xlabel(phyparams[0], fontsize = gv.axes_labels_fontsize)
			ax.set_ylabel(phyparams[1], fontsize = gv.axes_labels_fontsize)
			
			if (not (dbs.scales[0] == 1)):
				ax.set_xscale('log', basex = dbs.scales[0], basey = dbs.scales[0])
				ax.invert_xaxis()
			if (not (dbs.scales[1] == 1)):
				ax.set_yscale('log', basex = dbs.scales[1], basey = dbs.scales[1])
				ax.invert_yaxis()
		
			ax.tick_params(axis = 'both', which = 'both', pad = gv.ticks_pad, direction = 'inout', length = gv.ticks_length, width = gv.ticks_width, labelsize = gv.ticks_fontsize)
			# Legend
			cbar = plt.colorbar(cplot, extend = "both", spacing = "proportional", drawedges = False, ticks = clevels)
			cbar.ax.set_xlabel(ml.Metrics[logmet][1], fontsize = gv.colorbar_fontsize)
			cbar.ax.tick_params(labelsize = gv.legend_fontsize, pad = gv.ticks_pad, length = gv.ticks_length, width = gv.ticks_width)
			cbar.ax.xaxis.labelpad = gv.ticks_pad
			cbar.ax.set_yticklabels([("$%.2f \\times 10^{%d}$" % (clevels[i] * np.power(10, np.abs(np.int(np.log10(clevels[i])))), np.int(np.log10(clevels[i])))) for i in range(len(clevels))])
			# Save the plot
			pdf.savefig(fig)
			plt.close()
		#Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo['Title'] = ("%s vs. %s at levels %s, for %d %s channels." % (ml.Metrics[logmet][0], ",".join(phylist), ", ".join(map(str, range(1, 1 + dbs.levels))), dbs.channels, qc.Channels[dbs.channel][0]))
		pdfInfo['Author'] = "Pavithran Iyer"
		pdfInfo['ModDate'] = dt.datetime.today()
	return None


def CompareSubs(logmet, *dbses):
	# Compare the Logical error rates from two submissions.
	# The comparision only makes sense when the logical error rates are measured for two submissions that have the same physical channels.
	ndb = len(dbses)
	cdepth = min([dbses[i].levels for i in range(ndb)])
	logErrs = np.zeros((ndb, dbs1.channels, cdepth), dtype = np.longdouble)
	for i in range(ndb):
		logErrs[i, :, :] = np.load(fn.LogicalErrorRates(dbses[i], logmet))
	plotfname = fn.CompareLogErrRates(dbses, logmet)
	with PdfPages(plotfname) as pdf:
		for l in range(cdepth + 1):
			for i in range(ndb):
				for j in range(i + 1, ndb):
					fig = plt.figure(figsize = gv.canvas_size)
					plt.plot(logErrs[i, :, l + 1], logErrs[j, :, l + 1], color = ml.Metrics[logmet][3], marker = ml.Metrics[logmet][2], markersize = gv.marker_size, linestyle = 'None')
					plt.title("Level %d %s for %s vs. %s." % (l, ml.Metrics[logmet][1], dbses[i].timestamp, dbses[j].timestamp))
					# Axes labels
					ax = plt.gca()
					ax.set_xlabel(dbses[i].timestamp, fontsize = gv.axes_labels_fontsize)
					ax.set_xscale('log')
					ax.set_xlabel(dbses[j].timestamp, fontsize = gv.axes_labels_fontsize)
					ax.set_yscale('log')
					ax.tick_params(axis = 'both', which = 'major', pad = gv.ticks_pad, direction = 'inout', length = gv.ticks_length, width = gv.ticks_width, labelsize = gv.ticks_fontsize)
					# Save the plot
					pdf.savefig(fig)
					plt.close()
		#Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo['Title'] = ("Comparison of %s for databases %s up to %d levels." % (ml.Metrics[logmet][0], "_".join([dbses[i].timestamp for i in range(ndb)]), cdepth))
		pdfInfo['Author'] = "Pavithran Iyer"
		pdfInfo['ModDate'] = dt.datetime.today()
	return None

def CompareAnsatzToMetrics(dbs, pmet, lmet):
	# Compare the fluctuations in the logical error rates with respect to the physical noise metrics -- obtained from fit and those computed from standard metrics
	logerr = np.load(fn.LogicalErrorRates(dbs, lmet))
	phyerr = np.load(fn.PhysicalErrorRates(dbs, pmet))
	fiterr = np.load(fn.FitPhysRates(dbs, lmet))
	weightenums = np.load(fn.FitWtEnums(dbs, lmet))
	expo = np.load(fn.FitExpo(dbs, lmet))
	plotfname = fn.AnsatzComparePlot(dbs, lmet, pmet)
	with PdfPages(plotfname) as pdf:
		for l in range(dbs.levels):
			fig = plt.figure(figsize = gv.canvas_size)
			# Plots
			plt.plot(phyerr, logerr[:, l + 1], label = ml.Metrics[pmet][1], color = ml.Metrics[pmet][3], marker = ml.Metrics[pmet][2], markersize = gv.marker_size, linestyle = 'None')
			plt.plot(fiterr, logerr[:, l + 1], label = ("$\\epsilon$ where $\\widetilde{\\mathcal{N}}_{%d} = %s \\times \\left[\epsilon(\\mathcal{E})\\right]^{%.2f t}$" % (l + 1, latex_float(weightenums[l + 1]), expo)), color = 'blue', marker = 'o', markersize = gv.marker_size, linestyle = 'None')
			# Axes
			ax = plt.gca()
			ax.set_xlabel("$\\mathcal{N}_{0}$", fontsize = gv.axes_labels_fontsize)
			ax.set_xscale('log')
			ax.set_ylabel(("$\\mathcal{N}_{%d}$  $\\left(%s\\right)$" % (l + 1, ml.Metrics[lmet][1].replace("$", ""))), fontsize = gv.axes_labels_fontsize)
			ax.set_yscale('log')
			ax.tick_params(axis = 'both', which = 'both', pad = gv.ticks_pad, direction = 'inout', length = gv.ticks_length, width = gv.ticks_width, labelsize = gv.ticks_fontsize)
			# Legend
			lgnd = plt.legend(numpoints = 1, loc = 4, shadow = True, fontsize = gv.legend_fontsize, markerscale = gv.legend_marker_scale)
			# Save the plot
			pdf.savefig(fig)
			plt.close()
		#Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo['Title'] = ("Comparing fit obtained p to physical %s at levels %s, by studying fluctuations of output %s for %d channels." % (ml.Metrics[pmet][0], ", ".join(map(str, range(1, 1 + dbs.levels))), ml.Metrics[lmet][0], dbs.channels))
		pdfInfo['Author'] = "Pavithran Iyer"
		pdfInfo['ModDate'] = dt.datetime.today()
	return None


def ValidatePrediction(dbs, pmet, lmet):
	# Validate a prediction by comparing the fluctuations in the logical error rate with reprect to
	# (i) The predicted physical noise rate
	# (ii) Any standard metric.
	logerr = np.load(fn.LogicalErrorRates(dbs, lmet))
	phyerr = np.load(fn.PhysicalErrorRates(dbs, pmet))
	macerr = np.load(fn.PredictedPhyRates(dbs))
	plotfname = fn.PredictComparePlot(dbs, lmet, pmet)
	with PdfPages(plotfname) as pdf:
		for l in range(1 + dbs.levels):
			fig = plt.figure(figsize = gv.canvas_size)
			plt.plot(phyerr, logerr[:, l], label = ml.Metrics[pmet][1], color = ml.Metrics[pmet][3], marker = ml.Metrics[pmet][2], markersize = gv.marker_size, linestyle = 'None')
			plt.plot(macerr, logerr[:, l], label = "$\\epsilon_{\\rm predicted}$", color = 'blue', marker = 'o', markersize = gv.marker_size, linestyle = 'None')
			# Axes labels
			ax = plt.gca()
			ax.set_xlabel("$\\mathcal{N}_{0}$", fontsize = gv.axes_labels_fontsize)
			ax.set_xscale('log')
			ax.set_ylabel(("$\\widetilde{\\mathcal{N}}_{%d}$  $\\left(%s\\right)$" % (l + 1, ml.Metrics[lmet][1].replace("$", ""))), fontsize = gv.axes_labels_fontsize)
			ax.set_yscale('log')
			ax.tick_params(axis = 'both', which = 'both', pad = gv.ticks_pad, direction = 'inout', length = gv.ticks_length, width = gv.ticks_width, labelsize = gv.ticks_fontsize)
			# Legend
			lgnd = plt.legend(numpoints = 1, loc = 4, shadow = True, fontsize = gv.legend_fontsize, markerscale = gv.legend_marker_scale)
			# Save the plot
			pdf.savefig(fig)
			plt.close()
		#Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo['Title'] = ("Comparing predicted p to physical %s at levels %s, by studying fluctuations of output %s for %d channels." % (ml.Metrics[pmet][0], ", ".join(map(str, range(1, 1 + dbs.levels))), ml.Metrics[lmet][0], dbs.channels))
		pdfInfo['Author'] = "Pavithran Iyer"
		pdfInfo['ModDate'] = dt.datetime.today()
	return None
