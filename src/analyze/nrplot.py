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
from define.qcode import GetOperatorsForLSTIndex
from define.fnames import NRWeightsFile, NRWeightsPlotFile, RawPhysicalChannel, NRProbsPlotFile
from define.decoder import ComputeNRBudget, GetTotalErrorBudget, GetLeadingPaulis
from analyze.utils import scientific_float

def NRBudgetInfo(dbses, noise, samples):
	# Compute the total probabilities of errors by weight for each alpha.
	qcode = dbses[0].eccs[0]
	max_weight = qcode.N // 2 + 1
	budgets = np.zeros((len(dbses), 1 + max_weight, len(samples)), dtype = np.double)
	for d in range(len(dbses)):
		alpha = dbses[d].decoder_fraction
		for s in range(len(samples)):
			chan_probs = np.load(RawPhysicalChannel(dbses[d], noise))[samples[s], :]
			nr_weights = np.load(NRWeightsFile(dbses[d], noise))[samples[s], :].astype(np.int64)
			(__, leading_paulis, leading_probs) = GetLeadingPaulis(alpha, qcode, chan_probs, "split", nr_weights_all = nr_weights, max_weight = max_weight)
			# print("leading_probs for alpha = {}\n{}".format(alpha, leading_probs))
			for p in range(leading_paulis.size):
				(operator, __) = GetOperatorsForLSTIndex(qcode, [leading_paulis[p]])
				weight = np.count_nonzero(operator[0])
				if (weight <= max_weight):
					budgets[d, weight, s] = budgets[d, weight, s] + np.real(leading_probs[p])

	# Compute the total number of Pauli errors for each alpha
	nr_paulis = np.mean(np.array([[GetTotalErrorBudget(dbs, noise, samp) for dbs in dbses[:]] for samp in samples], dtype=np.int64), axis=0).astype(int)

	# Average the budgets computed over all samples.
	budgets_averaged = np.sum(budgets, axis=2)
	
	# Convert to percentages
	budgets_averaged_percentages = np.zeros_like(budgets_averaged)
	# Print the budgets
	for d in range(len(dbses)):
		budgets_averaged_percentages[d, :] = budgets_averaged[d, :] / np.sum(budgets_averaged[d]) * 100
		# print("NR data set with {} Pauli error rates.".format(nr_paulis[d]))
		print(nr_paulis[d], end = "")
		for w in range(budgets_averaged.shape[1]):
			# print("Total fraction of weight w = {} errors: {} %".format(w, np.round(budgets_averaged_percentages[d, w], 3)))
			print(" & %.5f" % (budgets_averaged_percentages[d, w]), end="")
		print(" \\\\\n\\hline")

	(n_rows, n_cols) = budgets_averaged_percentages.shape
	# Plot the histogram
	plotfname = NRProbsPlotFile(dbses[0], noise, [-1])
	with PdfPages(plotfname) as pdf:
		fig = plt.figure(figsize=(36,30))

		# We want the histograms for each weight, stacked vertically. So we need to compute the bottom of each bar.
		# bottoms = np.zeros(n_rows)
		# for w in range(n_cols):
		# 	plt.bar(np.arange(n_rows), budgets_averaged_percentages[:, w], width = 0.7, bottom = bottoms, label = "w = %d" % (w), color=gv.Colors[w % gv.n_Colors])
		# 	bottoms += budgets_averaged_percentages[:, w]

		# Plot the budgets
		for d in range(len(dbses)):
			plt.plot(np.linspace(0, n_cols - 1, n_cols), budgets_averaged_percentages[d, :], marker = gv.Markers[d % gv.n_Markers], markersize = gv.marker_size, linewidth = gv.line_width, color = gv.Colors[d % gv.n_Colors], label = "$K = %d$" % (nr_paulis[d]))
			
		plt.yscale('log')
		plt.ylabel("Relative budget", fontsize=gv.axes_labels_fontsize)
		# plt.xlabel("Number of Pauli errors", fontsize=gv.axes_labels_fontsize)
		plt.xlabel("Weight $(w)$", fontsize=gv.axes_labels_fontsize)
		
		ax = plt.gca()
		
		# Legend
		ax.legend(numpoints=1, loc=1, shadow=True, fontsize=2 * gv.legend_fontsize, markerscale=gv.legend_marker_scale)
		
		# Bottom X ticks show the size of the NR data set.
		# ax.set_xticks(np.arange(n_rows))
		ax.set_xticks(np.linspace(0, n_cols - 1, n_cols))
		# ax.set_xticklabels(nr_paulis, rotation = 45)
		ax.tick_params(axis="both", which="both", pad=gv.ticks_pad * 0.5, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize)
		
		# Save the plot
		pdf.savefig(fig)
		plt.close()
		
		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Pauli distribution of errors."
		pdfInfo["ModDate"] = dt.datetime.today()

	return None


def NRWeightsPlot(dbses, noise, samples):
	# Compute the relative budget taken up by the set of Pauli error rates for each weight, in the NR dataset.
	# Plot histograms one on top of each other: stacked histogram.
	
	# Show the information regarding the distribution of error probabilities by weight.
	NRBudgetInfo(dbses, noise, samples)

	qcode = dbses[0].eccs[0]
	max_weight = 1 + qcode.N//2
	budgets = np.mean(np.array([[GetTotalErrorBudget(dbs, noise, samp) for dbs in dbses[:]] for samp in samples], dtype=np.int64), axis=0).astype(int)
	# print("budgets = {}".format(budgets))
	# dbses = [dbses_input[d + 1] for d in uniques]
	
	# Compute the average number of errors of each weight in the NR data.
	nr_weights = np.load(NRWeightsFile(dbses[0], noise))[samples, :].astype(np.int64)

	nr_weights_avg = np.mean(nr_weights, axis=0).astype(np.int64)
	alphas = np.array([dbs.decoder_fraction for dbs in dbses], dtype = np.float64)
	xticklabels_bottom = budgets

	# Compute the average relative budget of errors for each weight in the NR data.
	max_weight = qcode.N//2
	# (__, percentages) = ComputeNRBudget(nr_weights_avg, alphas, qcode.N, max_weight=max_weight)
	# print("percentages\n{}".format(percentages))
	percentages = np.zeros((len(dbses), 1 + max_weight), dtype = np.float64)
	for s in range(len(samples)):
		(__, percentage_samp) = ComputeNRBudget(nr_weights[s, :], alphas, qcode.N, max_weight=max_weight)
		percentages = percentages + percentage_samp
		# print("Sample {}\npercentages\n{}".format(s, percentage_samp))
	percentages = percentages / len(samples)

	# print("percentages\n{}".format(np.round(percentages, 1)))
	
	# (__, percentages) = np.array([ComputeNRBudget(nr_weights[s, :], alphas, qcode.N) for s in range(len(samples))], dtype = np.int64)
	# print("percentages\n{}".format(percentages))
	(n_rows, n_cols) = percentages.shape

	# The top xticklabels show the Pauli error budget left out in the NR data set.
	chan_probs = np.real(np.mean(np.load(RawPhysicalChannel(dbses[0], noise))[samples, :], axis=0))
	xticklabels_top = [None for __ in alphas]
	for (i, alpha) in enumerate(alphas):
		(__, __, knownPaulis) = GetLeadingPaulis(alpha, qcode, chan_probs, "weight", nr_weights_avg)
		xticklabels_top[i] = scientific_float(1 - np.sum(knownPaulis))
	
	# print("alphas\n{}\nxticklabels_bottom\n{}\nxticklabels top\n{}\nrows\n{}".format(alphas, xticklabels_bottom, xticklabels_top, np.arange(n_rows)))

	# print("samples: {}".format(samples))

	plotfname = NRWeightsPlotFile(dbses[0], noise, samples)
	with PdfPages(plotfname) as pdf:
		fig = plt.figure(figsize=(36,30))

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
		ax.legend(numpoints=1, loc=1, shadow=True, fontsize=2 * gv.legend_fontsize, markerscale=gv.legend_marker_scale)
		
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