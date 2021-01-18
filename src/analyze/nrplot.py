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
from define.fnames import NRWeightsFile, NRWeightsPlotFile
from define.decoder import ComputeNRBudget, GetTotalErrorBudget


def NRWeightsPlot(dbses_input, noise, sample):
	# Compute the relative budget taken up by the set of Pauli error rates for each weight, in the NR dataset.
	# Plot histograms one on top of each other: stacked histogram.
	
	# Only show alpha values that correspond to distinct number of Pauli error rates.
	max_weight = 1 + dbses_input[0].eccs[0].N//2
	budgets, uniques = np.unique(np.array([GetTotalErrorBudget(dbs, noise, sample) for dbs in dbses_input[1:-1]], dtype=np.int), return_index=True)
	# print("uniques = {}".format(type(uniques)))
	dbses = [dbses_input[d + 1] for d in uniques]

	nq = dbses[0].eccs[0].N
	nr_weights = np.loadtxt(NRWeightsFile(dbses[0], noise), dtype = np.int)[sample, :]
	alphas = np.array([dbs.decoder_fraction for dbs in dbses], dtype = np.float)
	xticklabels = budgets
	(__, percentages) = ComputeNRBudget(nr_weights, alphas, nq)
	(n_rows, n_cols) = percentages.shape
	
	# print("percentages\n{}".format(np.round(percentages, 1)))
	# print("alphas\n{}\nxticklabels\n{}rows\n{}".format(alphas, xticklabels, np.arange(n_rows)))

	plotfname = NRWeightsPlotFile(dbses_input[0], noise, sample)
	with PdfPages(plotfname) as pdf:
		fig = plt.figure(figsize=gv.canvas_size)

		# We want the histograms for each weight, stacked vertically. So we need to compute the bottom of each bar.
		bottoms = np.zeros(n_rows)
		for w in range(n_cols):
			plt.bar(np.arange(n_rows), percentages[:, w], width = 0.7, bottom = bottoms, label = "w = %d" % (w), color=gv.Colors[w % gv.n_Colors])
			bottoms += percentages[:, w]
		
		plt.ylabel("Relative budget", fontsize=gv.axes_labels_fontsize)
		plt.xlabel("$\\alpha$", fontsize=gv.axes_labels_fontsize)
		
		# plt.title('Weight wise distribution of NR data')
		# plt.xticks(ind[::5], np.round(alphas,4)[::5], rotation = 45)
		
		plt.xticks(np.arange(n_rows), xticklabels, rotation = 45)
		ax = plt.gca()
		ax.tick_params(axis="both", which="both", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize)

		# plt.yticks(np.arange(0, 100, 25))

		# Legend
		ax.legend(numpoints=1, loc=1, shadow=True, fontsize=gv.legend_fontsize, markerscale=gv.legend_marker_scale)
		
		# Save the plot
		pdf.savefig(fig)
		plt.close()
		
		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Pauli distribution of errors."
		pdfInfo["ModDate"] = dt.datetime.today()

	return None