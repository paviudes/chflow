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


def PartialNRPlot(logmet, pmets, dsets, inset_flag, nbins, thresholds, minimal=0):
	# Compare the effect of p_u + RC on predictability.
	# Plot no RC with infid and RC with p_u.
	# pmets = list(map(lambda phy: phy.strip(" "), phymets.split(",")))
	###
	min_bin_fraction = 0.9
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
		relative_paddings = [1, 1]
		
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
			# print("Number of bins for alpha = {} is {}.".format(dsets[d].decoder_fraction, collapsed_bins[d].shape[0]))
		
		# Compute the uncorr data with minimum width
		# max_width_uncorr = 1 + np.argmin([np.min(collapsed_bins[d][:, 0])/np.max(collapsed_bins[d][:, 0]) for d in range(1, len(collapsed_bins))])
		uncorr_widths = np.zeros(ndb - 1, dtype = np.double)
		for d in range(1, ndb):
			xaxis = np.sqrt(collapsed_bins[d][:, 0] * collapsed_bins[d][:, 1])
			uncorr_widths[d - 1] = np.max(xaxis) - np.min(xaxis)
		# max_width_uncorr = 1 + np.argmax(uncorr_widths)
		max_width_uncorr = 1

		# print("max_width_uncorr = {}\n{}".format(max_width_uncorr, [np.min(collapsed_bins[d][:, 0])/np.max(collapsed_bins[d][:, 0]) for d in range(1, len(collapsed_bins))]))
		##### Explicitly set the X-cutoff for uncorr
		# xcutoff_uncorr = {"left": np.min(collapsed_bins[max_width_uncorr][:, 0]), "right": np.max(collapsed_bins[max_width_uncorr][:, 0])}
		xcutoff_uncorr = {"left": 1E-8, "right": 5E-3}
		##### Explicitly set the X-cutoff for infid
		xcutoff_infid = {"left": 4E-3, "right": 8E-2}
		#####
		# xcutoff_infid = {"left": 0, "right": 1}
		# xcutoff_uncorr = {"left": 0, "right": 1}
		
		reference_uncorr_width = uncorr_widths[max_width_uncorr - 1]
		for d in range(ndb):
			# Plot the scatter metric with the bin averages
			xaxis = np.sqrt(collapsed_bins[d][:, 0] * collapsed_bins[d][:, 1])
			if (d > 0):
				# uncorr_width = np.max(collapsed_bins[d][:, 0]) - np.min(collapsed_bins[d][:, 0])
				ratio = reference_uncorr_width / uncorr_widths[d - 1]
				xaxis = np.min(np.sqrt(collapsed_bins[max_width_uncorr][:, 0] * collapsed_bins[max_width_uncorr][:, 1])) + (xaxis - np.min(xaxis)) * ratio

			yaxis = collapsed_bins[d][:, 3]
			if (d == 0):
				# Plot the scatter metric for the infid plot
				current_axes = ax_infid
				current_axes.xaxis.tick_top()
				current_axes.xaxis.set_label_position('top')
				focus, = np.nonzero(np.logical_and(xaxis >= xcutoff_infid["left"], xaxis < xcutoff_infid["right"]))
			else:
				# Plot the scatter metric for the uncorr plot
				current_axes = ax_uncorr
				current_axes.xaxis.tick_bottom()
				current_axes.xaxis.set_label_position('bottom')
				focus, = np.nonzero(np.logical_and(xaxis >= xcutoff_uncorr["left"], xaxis < xcutoff_uncorr["right"]))
			
			current_axes.plot(xaxis[focus], yaxis[focus], marker=gv.Markers[d % gv.n_Markers], color=gv.Colors[d % gv.n_Colors], linestyle="-", linewidth=gv.line_width, markersize=gv.marker_size, alpha=0.75)
			
			print("Focus: {}\nNumber of channels: {}".format(focus, np.sum([collapsed_bins[d][b, 2] for b in focus])))


			# Empty plot for the legend.
			n_errors = int(dsets[d].decoder_fraction * np.power(4, dsets[0].eccs[0].N))
			if (n_errors == 0):
				label = ml.Metrics["infid"]["name"]
			# elif (n_errors == np.power(4, dsets[0].eccs[0].N)):
			# 	label = "All NR data"
			else:
				label = "$K = %d$" % (n_errors)
			ax_infid.plot([], [], marker=gv.Markers[d % gv.n_Markers], color=gv.Colors[d % gv.n_Colors], linestyle="-", linewidth=gv.line_width, markersize=gv.marker_size, alpha=0.75, label = label)

			# Tick parameters for both axes
			current_axes.tick_params(
				axis="both",
				which="both",
				pad=relative_paddings[d % len(relative_paddings)] * gv.ticks_pad,
				direction="inout",
				length=gv.ticks_length,
				width=gv.ticks_width,
				labelsize=1.5 * gv.ticks_fontsize,
			)
			print("Plot done for alpha = {}".format(dsets[d].decoder_fraction))

		# Draw grid lines
		ax_infid.grid(which="major")
		
		# Axes labels
		if (minimal == 0):
			ax_infid.set_xlabel(ml.Metrics[pmets[0]]["phys"], fontsize=1.5 * gv.axes_labels_fontsize, labelpad=0.6)
			ax_uncorr.set_xlabel("Logical estimator", fontsize=1.5 * gv.axes_labels_fontsize, labelpad=1)
			ax_infid.set_ylabel("Amount of dispersion $(\\Delta)$", fontsize=1.5 * gv.axes_labels_fontsize)
		
		# Axes scales
		ax_infid.set_xscale("log")
		ax_uncorr.set_xscale("log")
		ax_infid.set_yscale("log")

		## Ticks and ticklabels for the X-axes
		# Infid X-axes
		raw_inset_ticks_infid = collapsed_bins[0][:, 0]
		print("raw_inset_ticks_infid: {}".format(raw_inset_ticks_infid))
		(intended_infid_ticks, __) = SetTickLabels(raw_inset_ticks_infid)
		print("intended_infid_ticks: {}".format(intended_infid_ticks))
		(positions_infid, infid_ticks) = ComputeBinPositions(intended_infid_ticks, raw_inset_ticks_infid)
		print("infid_ticks = {}".format(infid_ticks))
		ax_infid.set_xticks(infid_ticks[infid_ticks > -1])
		ax_infid.set_xticklabels(list(map(lambda x: "$%s$" % latex_float(x), infid_ticks[infid_ticks > -1])), rotation=45, rotation_mode="anchor", ha="left", va="baseline", fontsize = 1.5 * gv.ticks_fontsize)
		# print("Infid ticks\n{}".format(list(ax_infid.get_xticklabels())))
		# print("-----------")
		# infid_xlim = [np.min(raw_inset_ticks_infid), np.max(raw_inset_ticks_infid)]
		infid_xlim = [xcutoff_infid["left"], xcutoff_infid["right"]]
		ax_infid.set_xlim(*infid_xlim)
		
		# uncorr_bins = [collapsed_bins[d][:, 0] for d in range(1, len(collapsed_bins))]
		# max_width_uncorr = 1 + np.argmin([(np.max(collapsed_bins[d][:, 0]) - np.min(collapsed_bins[d][:, 0])) for d in range(1, len(collapsed_bins))])
		# Uncorr X-axes
		# raw_inset_ticks_uncorr = np.sort(np.concatenate(tuple(uncorr_bins)))
		raw_inset_ticks_uncorr = collapsed_bins[max_width_uncorr][:, 0]
		# print("raw_inset_ticks_uncorr: {}".format(raw_inset_ticks_uncorr))
		(intended_uncorr_ticks, __) = SetTickLabels(raw_inset_ticks_uncorr)
		# print("intended_uncorr_ticks: {}".format(intended_uncorr_ticks))
		(positions_uncorr, uncorr_ticks) = ComputeBinPositions(intended_uncorr_ticks, raw_inset_ticks_uncorr)
		# print("uncorr_ticks = {}".format(uncorr_ticks))
		ax_uncorr.set_xticks(uncorr_ticks[uncorr_ticks > -1])
		ax_uncorr.set_xticklabels(list(map(lambda x: "$%s$" % latex_float(x), uncorr_ticks[uncorr_ticks > -1])), rotation=-45, rotation_mode="anchor", ha="left", va="baseline", fontsize = 1.5 * gv.ticks_fontsize)
		# print("Uncorr ticks\n{}".format(ax_uncorr.get_xticks()))
		# print("xxxxxxxxxxxx")
		# uncorr_xlim = [np.min(raw_inset_ticks_uncorr), np.max(raw_inset_ticks_uncorr)]
		uncorr_xlim = [xcutoff_uncorr["left"], xcutoff_uncorr["right"]]
		ax_uncorr.set_xlim(*uncorr_xlim)

		# Legend
		ax_infid.legend(numpoints=1, loc=(0.77,0.485), shadow=True, fontsize=1.25 * gv.legend_fontsize, markerscale=gv.legend_marker_scale)

		# Globally set the font family.
		matplotlib.rcParams["font.family"] = "Times New Roman"
		plt.rcParams["font.family"] = "Times New Roman"
		matplotlib.rc('mathtext', fontset='stix')
		# plt.xticks(fontname = "Times New Roman")
		# plt.yticks(fontname = "Times New Roman")
		###

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