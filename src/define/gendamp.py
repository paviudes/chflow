import datetime as dt
import numpy as np
try:
	import matplotlib
	matplotlib.use("Agg")
	from matplotlib import colors, ticker, cm
	# from matplotlib.mlab import griddata
	from matplotlib.backends.backend_pdf import PdfPages
	import matplotlib.pyplot as plt
except Exception:
	sys.stderr.write("\033[91m\033[2mMATPLOTLIB does not exist, cannot make plots.\n\033[0m")
try:
	from scipy.interpolate import griddata
except Exception:
	sys.stderr.write("\033[91m\033[2mSCIPY does not exist, cannot make 3D plots.\n\033[0m")

import globalvars as gv
import fnames as fn
import metrics as ml

def GeneralizedDamping(relax, dephase):
	# Return the Krauss operators of the generalized daping channel given the dephasing and relaxation rates
	# Generalized amplitude-phase damping channel:
		# This is the effect of applying amplitude daping on the output of a phase damping channel (or vise-versa).
		# If the amplitude daping Krauss operators are A0 and A1, then the generalized amplitude-phase damping channel has 4 Krauss operators given as follows.
		# K0 = Sqrt(1-p) A0, K1 = Sqrt(1-p) A1, K2 = Sqrt(p) Z A0 and K3 = Sqrt(p) A1.
	    # we will have p given by the input "dephase" and A0, A1 specified by "lambda".
	krauss = np.zeros((4, 2, 2), dtype = np.complex128)
	krauss[0, :, :] = np.sqrt(1 - dephase) * np.array([[1, 0], [0, np.sqrt(1 - relax)]], dtype = np.complex128)
	krauss[1, :, :] = np.sqrt(1 - dephase) * np.array([[0, np.sqrt(relax)], [0, 0]], dtype = np.complex128)
	krauss[2, :, :] = np.sqrt(dephase) * np.array([[1, 0], [0, (-1) * np.sqrt(1 - relax)]], dtype = np.complex128)
	krauss[3, :, :] = np.sqrt(dephase) * np.array([[0, np.sqrt(relax)], [0, 0]], dtype = np.complex128)
	return krauss


def GDTimeScales(trelax, ratioT2T1):
	# Return the Krauss operators of the generalized damping channel, given the parameters T2/T1 and t/T1.
	# The parameters are related to p and gamma, as defined above in the following way, see also https://arxiv.org/pdf/1404.3747.pdf.
	# exp(-t/T1) = 1 - gamma.
	# exp(-t/T2) = sqrt((1 - gamma) * (1 - p)).
	# Hence, gamma = 1 - exp(-t/T1) and p = 1 - exp(-2 t/T2)/(1 - gamma),
	# where t/T2 = (t/T1) / (T2/T1).
	relax = 1 - np.exp((-1) * trelax)
	# print("p_AD = %g" % (relax))
	dephase = 1 - np.exp((-2) * trelax/ratioT2T1)/(1 - relax)
	# print("p_PD = %g" % (dephase))
	krauss = GeneralizedDamping(relax, dephase)
	# print("----")
	return krauss


def GDTimeScalesExplicit(t, T1, T2):
	# Return the Krauss operators of the generalized damping channel, given T1, T2 and t.
	# We will use the definitions in https://arxiv.org/pdf/1404.3747.pdf.
	trelax = t/np.longdouble(T1)
	ratioT2T1 = T2/np.longdouble(T1)
	krauss = GDTimeScales(trelax, ratioT2T1)
	return krauss


def GDPerfPlots(submit, logmet, xcol = 0, ycol = 1):
	# Plot performance contours for the logical error rates for every concatenation level, with repect to the dephasing and relaxation rates.
	# Each plot will be a contour plot or a color density plot indicating the logical error, with the x-axis as the dephasing rate and the y-axis as the relaxation rate.
	# There will be one plot for every concatenation level.
	logErr = np.load(fn.LogicalErrorRates(submit, logmet, fmt = "npy"))
	xaxis = np.power(2/np.longdouble(3), submit.noiserates[:, 0])
	yaxis = np.power(2/np.longdouble(3), submit.noiserates[:, 1])
	(meshX, meshY) = np.meshgrid(np.linspace(xaxis.min(), xaxis.max(), max(100, xaxis.shape[0])), np.linspace(xaxis.min(), xaxis.max(), max(100, yaxis.shape[0])))
	plotfname = fn.LevelWise(submit, "lambda_p", logmet)
	with PdfPages(plotfname) as pdf:
		nqubits = 1
		dist = 1
		for l in range(submit.levels):
			nqubits = nqubits * submit.eccs[l].N
			dist = dist * submit.eccs[l].D
			fig = plt.figure(figsize = gv.canvas_size)
			meshZ = np.abs(griddata((xaxis, yaxis), logErr[:, l + 1], (meshX, meshY), method = "cubic"))
			# print("Z\n%s" % (np.array_str(meshZ)))
			cplot = plt.contourf(meshX, meshY, meshZ, cmap = cm.bwr, locator = ticker.LogLocator(), linestyles = gv.contour_linestyle, pad = gv.ticks_pad)
			# plt.contour(meshX, meshY, meshZ, colors = 'k', locator = ticker.LogLocator(), linewidth = gv.line_width)
			plt.scatter(xaxis, yaxis, marker = 'o', color = 'k')
			# Title
			plt.title("N = %d, d = %d" % (nqubits, dist), fontsize = gv.title_fontsize, y = 1.03)
			# Axes labels
			ax = plt.gca()
			ax.set_xlabel(qc.Channels[submit.channel][2][xcol], fontsize = gv.axes_labels_fontsize)
			ax.set_xscale('log')
			ax.set_ylabel(qc.Channels[submit.channel][2][ycol], fontsize = gv.axes_labels_fontsize)
			ax.set_yscale('log')
			ax.tick_params(axis = 'both', which = 'both', pad = gv.ticks_pad, direction = 'inout', length = gv.ticks_length, width = gv.ticks_width, labelsize = gv.ticks_fontsize)
			# Legend
			cbar = plt.colorbar(cplot)
			cbar.ax.set_xlabel(ml.Metrics[logmet][1], fontsize = gv.colorbar_fontsize)
			cbar.ax.tick_params(labelsize = gv.legend_fontsize)
			cbar.ax.xaxis.labelpad = gv.ticks_pad
			# Save the plot
			pdf.savefig(fig)
			plt.close()
		#Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo['Title'] = ("%s at levels %s, for %d Generalized damping channels." % (ml.Metrics[logmet][0], ", ".join(map(str, range(1, 1 + submit.levels))), submit.channels))
		pdfInfo['Author'] = "Pavithran Iyer"
		pdfInfo['ModDate'] = dt.datetime.today()
	return None

def GenTicksInfo(axis, base, npoints):
	# Generate the tick positions and tick labels, given the axis points, for a log scale.
	tickpos = np.power(base, np.linspace(axis[0], axis[-1], npoints))
	ticklabels = ["$%g^{-%g}$" % (1/base, expo) for expo in np.linspace(axis[0], axis[-1], npoints)]
	print("Tick positions\n%s\nTick labels\n%s" % (tickpos, ticklabels))
	return (tickpos, ticklabels)

def GDPerfTimeScales(submit, logmet, colx = 0, coly = 1):
	# Plot performance contours for the logical error rates for every concatenation level, with repect to the dephasing and relaxation rates.
	# Each plot will be a contour plot or a color density plot indicating the logical error, with the x-axis as the dephasing rate and the y-axis as the relaxation rate.
	# There will be one plot for every concatenation level.
	npoints = 5
	logErr = np.load(fn.LogicalErrorRates(submit, logmet, fmt = "npy"))
	if (submit.scale == 1):
		(meshX, meshY) = np.meshgrid(np.linspace(submit.noiserates[:, colx].min(), submit.noiserates[:, colx].max(), max(100, submit.noiserates.shape[0])), np.linspace(submit.noiserates[:, coly].min(), submit.noiserates[:, coly].max(), max(100, submit.noiserates.shape[0])))
	else:
		(meshX, meshY) = np.meshgrid(np.linspace(np.power(submit.scale, submit.noiserates[:, colx]).min(), np.power(submit.scale, submit.noiserates[:, colx]).max(), max(100, submit.noiserates.shape[0])), np.linspace(np.power(submit.scale, submit.noiserates[:, coly]).min(), np.power(submit.scale, submit.noiserates[:, coly]).max(), max(100, submit.noiserates.shape[0])))
	# print("meshY\n%s" % (np.array_str(meshY)))
	plotfname = fn.LevelWise(submit, "t1_t2", logmet)
	with PdfPages(plotfname) as pdf:
		nqubits = 1
		dist = 1
		for l in range(1 + submit.levels):
			if (l == 0):
				nqubits = 1
				dist = 1
			else:
				nqubits = nqubits * submit.eccs[l - 1].N
				dist = dist * submit.eccs[l - 1].D
			fig = plt.figure(figsize = gv.canvas_size)
			if (submit.scale == 1):
				meshZ = griddata((submit.noiserates[:, colx], submit.noiserates[:, coly]), logErr[:, l], (meshX, meshY), method = "nearest")
			else:
				meshZ = griddata((np.power(submit.scale, submit.noiserates[:, colx]), np.power(submit.scale, submit.noiserates[:, coly])), logErr[:, l], (meshX, meshY), method = "linear")
			
			print("meshZ\n%s" % (np.array_str(meshZ)))

			# print("meshZ[np.nonzero(meshZ < 0)]\n%s" % (np.array_str(meshZ[np.nonzero(meshZ < 0)])))
			clevels = np.logspace(np.log10(logErr[:, l].min()), np.log10(logErr[:, l].max()), gv.contour_nlevs, base = 10.0)
			cplot = plt.contourf(meshX, meshY, meshZ, cmap = cm.winter, locator = ticker.LogLocator(), linestyles = gv.contour_linestyle, levels = clevels)
			# plt.contour(meshX, meshY, meshZ, colors = 'k', locator = ticker.LogLocator(), linewidth = gv.line_width)
			if (submit.scale == 1):
				plt.scatter(submit.noiserates[:, colx], submit.noiserates[:, coly], marker = 'o', color = 'k')
			else:
				plt.scatter(np.power(submit.scale, submit.noiserates[:, colx]), np.power(submit.scale, submit.noiserates[:, coly]), marker = 'o', color = 'k')
			# Title
			plt.title("N = %d, d = %d" % (nqubits, dist), fontsize = gv.title_fontsize, y = 1.03)
			# Axes labels
			ax = plt.gca()
			ax.set_xlabel(qc.Channels[submit.channel][2][xcol], fontsize = gv.axes_labels_fontsize)
			ax.set_ylabel(qc.Channels[submit.channel][2][ycol], fontsize = gv.axes_labels_fontsize)
			if (not (submit.scale == 1)):
				ax.set_xscale('log', basex = submit.scale, basey = submit.scale)
				ax.invert_xaxis()
				ax.set_yscale('log', basex = submit.scale, basey = submit.scale)
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
		pdfInfo['Title'] = ("%s at levels %s, for %d Generalized damping channels." % (ml.Metrics[logmet][0], ", ".join(map(str, range(1, 1 + submit.levels))), submit.channels))
		pdfInfo['Author'] = "Pavithran Iyer"
		pdfInfo['ModDate'] = dt.datetime.today()
	return None


if __name__ == '__main__':
	import chanreps as crep
	# test the properties of the Photon Loss channel
	relax = 0.1
	dephase = 0.1
	gdkrauss = GeneralizedDamping(relax, dephase)
	print("The Krauss operators for the Generalized Damping channel with relax = %g, dephase = %g are the following.\n%s" % (relax, dephase, np.array_str(gdkrauss)))
	gdprocess = crep.ConvertRepresentations(gdkrauss, "krauss", "process")
	print("The process matrix for the Photon Loss channel with relax = %g, dephase = %g is the following.\n%s" % (relax, dephase, np.array_str(gdprocess)))
	print("Physical metrics for the channel")
	metrics = ["fidelity", "frb"]
	metvals = ml.ComputeNorms(crep.ConvertRepresentations(gdprocess, "process", "choi"), metrics)
	print("{:<20} {:<10}".format("Metric", "Value"))
	print("-------------------------------")
	for m in range(len(metrics)):
		print("{:<20} {:<10}".format(ml.Metrics[metrics[m]][0], ("%.2e" % metvals[m])))
	print("xxxxxx")