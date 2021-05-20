# Critical packages
import os
import sys
import datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import colors, ticker, cm
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset
from matplotlib.ticker import LogLocator
from scipy.interpolate import griddata

# Functions from other modules
from define.fnames import LogicalErrorRates, PartialHammerPlotFile
from define import metrics as ml
from define import globalvars as gv
from analyze.load import LoadPhysicalErrorRates
from analyze.bins import ComputeBinVariance, CollapseBins, ComputeBinPositions
from analyze.utils import scientific_float, latex_float, SetTickLabels


def PartialNRPlot(logmet, phylist, dsets, inset_flag, nbins, thresholds):
	# Compare the effect of p_u + RC on predictability.
	# Plot no RC with infid and RC with p_u.
	# phylist = list(map(lambda phy: phy.strip(" "), phymets.split(",")))
	level = dsets[0].levels
	ndb = len(dsets)
	plotfname = PartialHammerPlotFile(dsets[0], logmet, phylist)
	with PdfPages(plotfname) as pdf:
		fig = plt.figure(figsize=(gv.canvas_size[0], gv.canvas_size[1]*1.2))
		ax = plt.gca()
		ax_top = ax.twiny()
		ndb = len(dbses)
		phyerrs = np.zeros((ndb, dbses[0].channels), dtype=np.double)
		logerrs = np.zeros((ndb, dbses[0].channels), dtype=np.double)
		for d in range(ndb):
			phyerrs[d, :] = LoadPhysicalErrorRates(dbses[d], pmets[d], None, level)
			logerrs[d, :] = np.load(LogicalErrorRates(dbses[d], lmet))[:, level]
			if d == 0:
				# print("Getting X cutoff for alpha = {} level = {}".format(dsets[d].decoder_fraction, level))
				xcutoff = GetXCutOff(phyerrs[d, :], np.load(LogicalErrorRates(dsets[d], logmet))[:, level], thresholds[level - 1], nbins=50, space="log")
				include = np.nonzero(np.logical_and(phyerrs[d, :] >= xcutoff["left"], phyerrs[d, :] <= xcutoff["right"]))[0]
			
			# Bin the X-axis and record the averages
			bins = ComputeBinVariance(phyerrs[d, include], logerrs[d, include], space="log", nbins=nbins)
			collapsed_bins[d] = CollapseBins(bins, min_bin_fraction * dbses[d].channels / nbins)
			print("Number of bins for alpha = {} is {}.".format(dbses[d].decoder_fraction, collapsed_bins[d].shape[0]))
		
			# Plot the scatter metric with the bin averages
			xaxis = np.arange(collapsed_bins[d].shape[0])
			yaxis = collapsed_bins[d][:, 3]
			if (d == 0):
				# Plot the scatter metric for the infid plot
				current_axes = ax_top
			else:
				# Plot the scatter metric for the uncorr plot
				current_axes = ax
			current_axes.plot(xaxis, yaxis, marker=gv.Markers[d % gv.n_Markers], color=gv.Colors[d % gv.n_Colors], linestyle="-", linewidth=gv.line_width, markersize=gv.marker_size, alpha=0.75)
			
			# Tick parameters for both axes
			current_axes.tick_params(
				axis="both",
				which="both",
				pad=relative_paddings[d % len(relative_paddings)] * gv.ticks_pad,
				direction="inout",
				length=gv.ticks_length,
				width=gv.ticks_width,
				labelsize=gv.ticks_fontsize,
			)
			print("Plot done for alpha = {}".format(dbses[d].decoder_fraction))

		# Axes labels
		ax.set_xlabel("Critical parameter computed from NR data", fontsize=gv.axes_labels_fontsize, labelpad=0.7, color="0.4")
		ax_top.set_xlabel(ml.Metrics[pmets[0]]["latex"], fontsize=gv.axes_labels_fontsize, labelpad=0.6, color="red")
		ax.set_ylabel("$\\Delta$", fontsize=gv.axes_labels_fontsize)
		
		# Axes scales
		ax.set_yscale("log")

		## Ticks and ticklabels for the X-axes
		# Ticks and labels for bottom X-axis with uncorr
		raw_inset_ticks = (collapsed_bins[1][:, 0] + collapsed_bins[1][:, 1]) / 2
		intended_bottom_ticks = SetTickLabels(raw_inset_ticks)
		(positions_bottom, bottom_ticks) = ComputeBinPositions(intended_bottom_ticks, raw_inset_ticks)
		ax.set_xticks(np.nonzero(positions_bottom > -1)[0])
		ax.set_xticklabels(list(map(lambda x: "$%s$" % latex_float(x), bottom_ticks[positions_bottom > -1])), rotation=45, color="red", rotation_mode="anchor", ha="left", va="baseline")
		# Ticks and labels for top X-axis with infid
		raw_inset_ticks = (collapsed_bins[1][:, 0] + collapsed_bins[1][:, 1]) / 2
		intended_top_ticks = SetTickLabels(raw_inset_ticks)
		(positions_top, top_ticks) = ComputeBinPositions(intended_top_ticks, raw_inset_ticks)
		ax_top.set_xticks(np.nonzero(positions_top > -1)[0])
		ax_top.set_xticklabels(list(map(lambda x: "$%s$" % latex_float(x), top_ticks[positions_top > -1])), rotation=45, color="red", rotation_mode="anchor", ha="left", va="baseline")

		# Legend
		ax.legend(numpoints=1, loc="upper right", shadow=True, fontsize=gv.legend_fontsize, markerscale=gv.legend_marker_scale)

		# Save the plot
		fig.tight_layout(pad=5)
		pdf.savefig(fig)
		plt.close()
		
		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Hammer plot."
		pdfInfo["Author"] = "Pavithran Iyer"
		pdfInfo["ModDate"] = dt.datetime.today()
	return None