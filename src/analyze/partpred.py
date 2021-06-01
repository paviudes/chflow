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
from analyze.bins import ComputeBinVariance, CollapseBins, ComputeBinPositions, GetXCutOff
from analyze.utils import scientific_float, latex_float, SetTickLabels


def PartialNRPlot(logmet, pmets, dsets, inset_flag, nbins, thresholds):
	# Compare the effect of p_u + RC on predictability.
	# Plot no RC with infid and RC with p_u.
	# pmets = list(map(lambda phy: phy.strip(" "), phymets.split(",")))
	min_bin_fraction = 0.1
	level = dsets[0].levels
	ndb = len(dsets)
	plotfname = PartialHammerPlotFile(dsets[0], logmet, pmets)
	with PdfPages(plotfname) as pdf:
		fig = plt.figure(figsize=(gv.canvas_size[0], gv.canvas_size[1]*1.2))
		ax_infid = plt.gca()
		ax_uncorr = ax_infid.twiny()
		ndb = len(dsets)
		phyerrs = np.zeros((ndb, dsets[0].channels), dtype=np.double)
		logerrs = np.zeros((ndb, dsets[0].channels), dtype=np.double)
		collapsed_bins = [None for d in range(ndb)]
		relative_paddings = [0.25, 1]
		for d in range(ndb):
			phyerrs[d, :] = LoadPhysicalErrorRates(dsets[d], pmets[d], None, level)
			logerrs[d, :] = np.load(LogicalErrorRates(dsets[d], logmet))[:, level]
			
			# Compute the X-cutoff from the Y-threshold.
			# print("Getting X cutoff for alpha = {} level = {}".format(dsets[d].decoder_fraction, level))
			xcutoff = GetXCutOff(phyerrs[d, :], np.load(LogicalErrorRates(dsets[d], logmet))[:, level], thresholds[level - 1], nbins=50, space="log")
			include = np.nonzero(np.logical_and(phyerrs[d, :] >= xcutoff["left"], phyerrs[d, :] <= xcutoff["right"]))[0]
			
			# Bin the X-axis and record the averages
			bins = ComputeBinVariance(phyerrs[d, include], logerrs[d, include], space="log", nbins=nbins)
			collapsed_bins[d] = CollapseBins(bins, min_bin_fraction * dsets[d].channels / nbins)
			print("Number of bins for alpha = {} is {}.".format(dsets[d].decoder_fraction, collapsed_bins[d].shape[0]))
		
			# Plot the scatter metric with the bin averages
			xaxis = np.arange(collapsed_bins[d].shape[0])
			yaxis = collapsed_bins[d][:, 3]
			if (d == 0):
				# Plot the scatter metric for the infid plot
				current_axes = ax_infid
				current_axes.xaxis.tick_top()
				current_axes.xaxis.set_label_position('top')
			else:
				# Plot the scatter metric for the uncorr plot
				current_axes = ax_uncorr
				current_axes.xaxis.tick_bottom()
				current_axes.xaxis.set_label_position('bottom')

			current_axes.plot(xaxis[1:-1], yaxis[1:-1], marker=gv.Markers[d % gv.n_Markers], color=gv.Colors[d % gv.n_Colors], linestyle="-", linewidth=gv.line_width, markersize=gv.marker_size, alpha=0.75)
			
			# Empty plot for the legend.
			ax_infid.plot([], [], marker=gv.Markers[d % gv.n_Markers], color=gv.Colors[d % gv.n_Colors], linestyle="-", linewidth=gv.line_width, markersize=gv.marker_size, alpha=0.75, label = "$\\alpha = %g$" % (dsets[d].decoder_fraction))

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
			print("Plot done for alpha = {}".format(dsets[d].decoder_fraction))

		# Draw grid lines
		ax_infid.grid(which="major")
		
		# Axes labels
		ax_infid.set_xlabel(ml.Metrics[pmets[0]]["latex"], fontsize=gv.axes_labels_fontsize, labelpad=0.6, color="0.4")
		ax_uncorr.set_xlabel("Critical parameter computed from NR data", fontsize=gv.axes_labels_fontsize, labelpad=1, color="red")
		ax_infid.set_ylabel("$\\Delta$", fontsize=gv.axes_labels_fontsize)
		
		# Axes scales
		ax_infid.set_yscale("log")

		## Ticks and ticklabels for the X-axes
		# Bottom X-axes
		raw_inset_ticks_bottom = (collapsed_bins[0][:, 0] + collapsed_bins[0][:, 1]) / 2
		print("raw_inset_ticks_bottom: {}".format(raw_inset_ticks_bottom))
		(intended_bottom_ticks, __) = SetTickLabels(raw_inset_ticks_bottom)
		# print("intended_bottom_ticks: {}".format(intended_bottom_ticks))
		(positions_bottom, bottom_ticks) = ComputeBinPositions(intended_bottom_ticks, raw_inset_ticks_bottom)
		print("bottom_ticks = {}".format(bottom_ticks))
		ax_infid.set_xticks(positions_bottom[positions_bottom > -1])
		ax_infid.set_xticklabels(list(map(lambda x: "$%s$" % latex_float(x), bottom_ticks[positions_bottom > -1])), rotation=45, color=gv.Colors[0], rotation_mode="anchor", ha="left", va="baseline")
		print("Infid ticks\n{}".format(list(ax_infid.get_xticklabels())))
		print("-----------")
		# Top X-axes
		raw_inset_ticks_top = (collapsed_bins[-1][:, 0] + collapsed_bins[-1][:, 1]) / 2
		print("raw_inset_ticks_top: {}".format(raw_inset_ticks_top))
		(intended_top_ticks, __) = SetTickLabels(raw_inset_ticks_top)
		# print("intended_top_ticks: {}".format(intended_top_ticks))
		(positions_top, top_ticks) = ComputeBinPositions(intended_top_ticks, raw_inset_ticks_top)
		# print("top_ticks = {}".format(top_ticks))
		ax_uncorr.set_xticks(positions_top[positions_top > -1])
		ax_uncorr.set_xticklabels(list(map(lambda x: "$%s$" % latex_float(x), top_ticks[positions_top > -1])), rotation=-45, color=gv.Colors[1], rotation_mode="anchor", ha="left", va="baseline")	
		print("Uncorr ticks\n{}".format(ax_uncorr.get_xticks()))
		print("xxxxxxxxxxxx")
		
		# Legend
		ax_infid.legend(numpoints=1, loc="upper right", shadow=True, fontsize=gv.legend_fontsize, markerscale=gv.legend_marker_scale)

		# Save the plot
		fig.tight_layout(pad=5)
		pdf.savefig(fig)
		plt.close()
		
		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Partial Hammer plot."
		pdfInfo["Author"] = "Pavithran Iyer"
		pdfInfo["ModDate"] = dt.datetime.today()
	return None