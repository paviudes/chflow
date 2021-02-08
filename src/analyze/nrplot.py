# Critical packages
import datetime as dt
import numpy as np
from scipy.special import comb
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Functions from other modules
from define import qcode as qec
import define.globalvars as gv
from define.fnames import NRWeightsFile, NRWeightsPlotFile, RawPhysicalChannel
from define.decoder import ComputeNRBudget, GetTotalErrorBudget, GetLeadingPaulis
from analyze.utils import scientific_float

def NRWeightsPlot(dbses_input, noise, sample):
	# Compute the relative budget taken up by the set of Pauli error rates for each weight, in the NR dataset.
	# Plot histograms one on top of each other: stacked histogram.
	qcode = dbses_input[0].eccs[0]
	# Only show alpha values that correspond to distinct number of Pauli error rates.
	max_weight = 1 + qcode.N//2
	budgets, uniques = np.unique(np.array([GetTotalErrorBudget(dbs, noise, sample) for dbs in dbses_input[1:-1]], dtype=np.int), return_index=True)
	# print("uniques = {}".format(type(uniques)))
	dbses = [dbses_input[d + 1] for d in uniques]
	
	nr_weights = np.loadtxt(NRWeightsFile(dbses[0], noise), dtype = np.int)[sample, :]
	alphas = np.array([dbs.decoder_fraction for dbs in dbses], dtype = np.float)
	xticklabels_bottom = budgets
	(__, percentages) = ComputeNRBudget(nr_weights, alphas, qcode.N)
	(n_rows, n_cols) = percentages.shape

	# The top xticklabels show the Pauli error budget left out in the NR data set.
	chan_probs = np.load(RawPhysicalChannel(dbses_input[0], noise))[sample, :]
	xticklabels_top = [None for __ in alphas]
	for (i, alpha) in enumerate(alphas):
		(__, __, knownPaulis) = GetLeadingPaulis(alpha, qcode, chan_probs, "weight", nr_weights)
		xticklabels_top[i] = scientific_float(1 - np.sum(knownPaulis))
	
	# print("percentages\n{}".format(np.round(percentages, 1)))
	print("alphas\n{}\nxticklabels_bottom\n{}\nxticklabels top\n{}\nrows\n{}".format(alphas, xticklabels_bottom, xticklabels_top, np.arange(n_rows)))

	plotfname = NRWeightsPlotFile(dbses_input[0], noise, sample)
	with PdfPages(plotfname) as pdf:
		fig = plt.figure(figsize=gv.canvas_size)

		# We want the histograms for each weight, stacked vertically. So we need to compute the bottom of each bar.
		bottoms = np.zeros(n_rows)
		for w in range(n_cols):
			plt.bar(np.arange(n_rows), percentages[:, w], width = 0.7, bottom = bottoms, label = "w = %d" % (w), color=gv.Colors[w % gv.n_Colors])
			bottoms += percentages[:, w]
		
		plt.ylabel("Relative budget", fontsize=gv.axes_labels_fontsize)
		plt.xlabel("Number of Pauli errors", fontsize=gv.axes_labels_fontsize)
		
		ax = plt.gca()
		ax_top = ax.twiny()
		
		# Legend
		ax.legend(numpoints=1, loc=1, shadow=True, fontsize=gv.legend_fontsize, markerscale=gv.legend_marker_scale)
		
		# Bottom X ticks show the size of the NR data set.
		ax.set_xticks(np.arange(n_rows))
		ax.set_xticklabels(xticklabels_bottom, rotation = 45)
		ax.tick_params(axis="both", which="both", pad=gv.ticks_pad * 0.5, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize)
		# Top X ticks show the budget left in the NR data set.
		ax_top.set_xticks(np.arange(0, n_rows, 2))
		ax_top.set_xticklabels(xticklabels_top[::2], rotation = 45)
		ax_top.tick_params(axis="both", which="both", pad=gv.ticks_pad * 0.5, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize)
		ax_top.set_xlim(ax.get_xlim())
		# Save the plot
		pdf.savefig(fig)
		plt.close()
		
		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Pauli distribution of errors."
		pdfInfo["ModDate"] = dt.datetime.today()

	return None