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
from matplotlib.patches import ConnectionPatch
from scipy.interpolate import griddata

# Functions from other modules
from define import metrics as ml
from define import globalvars as gv
from define.fnames import LogicalErrorRates, HammerPlot
from analyze.load import LoadPhysicalErrorRates
from analyze.bins import ComputeBinVariance, CollapseBins, ComputeBinPositions, GetXCutOff
from analyze.utils import scientific_float, latex_float, SetTickLabels


def BinVariancePlot(ax_principal, dbses, level, lmet, pmets, nbins, include):
	# Compare scatter for different physical metrics
	min_bin_fraction = 0.1
	relative_paddings = [0.25, 0.5]

	# Inset axes
	ax_inset = plt.axes([0, 0, 1, 1])
	# Position and relative size of the inset axes within ax_principal
	ip = InsetPosition(ax_principal, [0.64, 0.13, 0.32, 0.27]) # Positon: bottom right
	ax_inset.set_axes_locator(ip)
	# Mark the region corresponding to the inset axes on ax_principal and draw lines in grey linking the two axes.
	mark_inset(ax_principal, ax_inset, loc1=2, loc2=4, fc="none")
	# Top X-axis
	ax_inset_top = ax_inset.twiny()
	ax_inset_top.set_axes_locator(ip)
		
	ndb = len(dbses)
	phyerrs = np.zeros((ndb, dbses[0].channels), dtype=np.double)
	logerrs = np.zeros((ndb, dbses[0].channels), dtype=np.double)
	
	collapsed_bins = [None for d in range(ndb)]
	for d in range(ndb):
		phyerrs[d, :] = LoadPhysicalErrorRates(dbses[d], pmets[d], None, level)
		logerrs[d, :] = np.load(LogicalErrorRates(dbses[d], lmet))[:, level]
		
		bins = ComputeBinVariance(phyerrs[d, include[d]], logerrs[d, include[d]], space="log", nbins=nbins)
		collapsed_bins[d] = CollapseBins(bins, min_bin_fraction * dbses[d].channels / nbins)
		
		# print("Number of bins for alpha = {} is {}.".format(dbses[d].decoder_fraction, collapsed_bins[d].shape[0]))
		
		# xaxis = np.arange(collapsed_bins[d].shape[0])
		xaxis = np.sqrt(collapsed_bins[d][:, 0] * collapsed_bins[d][:, 1])
		yaxis = collapsed_bins[d][:, 3]
		
		if (d == 0):
			# Plot the scatter metric for the infid plot
			current_axes = ax_inset
			current_axes.xaxis.tick_top()
			current_axes.xaxis.set_label_position('top')
		else:
			# Plot the scatter metric for the uncorr plot
			current_axes = ax_inset_top
			current_axes.xaxis.tick_bottom()
			current_axes.xaxis.set_label_position('bottom')
		
		# Plot the scatter metric. Drop the first and the last bins to avoid effects due to including more points for one type of scatter.
		current_axes.plot(xaxis[1:-1], yaxis[1:-1], marker=gv.Markers[d % gv.n_Markers], color=gv.Colors[d % gv.n_Colors], linestyle="-", linewidth=gv.line_width, markersize=gv.marker_size, alpha=0.75)
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

		print("Plot done for d = {}".format(d))
	
	# Draw grid lines
	ax_inset.grid(which="major")
	
	# Axes labels
	ax_inset.set_ylabel("$\\Delta$", fontsize=gv.axes_labels_fontsize)
	
	# Axes scales
	ax_inset.set_xscale("log")
	ax_inset_top.set_xscale("log")
	ax_inset.set_yscale("log")

	## Ticks and ticklabels for the X-axes
	# Bottom X-axes
	raw_inset_ticks_bottom = collapsed_bins[0][:, 0]
	# print("raw_inset_ticks_bottom: {}".format(raw_inset_ticks_bottom))
	(intended_bottom_ticks, __) = SetTickLabels(raw_inset_ticks_bottom)
	# intended_bottom_ticks = raw_inset_ticks_bottom
	# print("intended_bottom_ticks: {}".format(intended_bottom_ticks))
	(positions_bottom, bottom_ticks) = ComputeBinPositions(intended_bottom_ticks, raw_inset_ticks_bottom)
	# print("bottom_ticks = {}".format(bottom_ticks))
	# ax_inset.set_xticks(positions_bottom[positions_bottom > -1])
	ax_inset.set_xticks(bottom_ticks[bottom_ticks > -1])
	ax_inset.set_xticklabels(list(map(lambda x: "$%s$" % latex_float(x), bottom_ticks[bottom_ticks > -1])), rotation=45, color=gv.Colors[0], rotation_mode="anchor", ha="left", va="baseline")
	bottom_xlim = [np.min(raw_inset_ticks_bottom), np.max(raw_inset_ticks_bottom)]
	ax_inset.set_xlim(*bottom_xlim)
	# ax_inset.minorticks_off()
	# ax_inset.set_xticks(raw_inset_ticks_bottom)
	# ax_inset.set_xticklabels(list(map(lambda x: "$%s$" % latex_float(x), raw_inset_ticks_bottom)), rotation=45, color=gv.Colors[0], rotation_mode="anchor", ha="left", va="baseline")
	# print("Inset infid ticks\n{}".format(ax_inset.get_xticks()))
	# print("-----------")
	# Top X-axes
	raw_inset_ticks_top = collapsed_bins[1][:, 0]
	# print("raw_inset_ticks_top: {}".format(raw_inset_ticks_top))
	(intended_top_ticks, __) = SetTickLabels(raw_inset_ticks_top)
	# intended_top_ticks = raw_inset_ticks_top
	# print("intended_top_ticks: {}".format(intended_top_ticks))
	(positions_top, top_ticks) = ComputeBinPositions(intended_top_ticks, raw_inset_ticks_top)
	# print("top_ticks = {}".format(top_ticks))
	# ax_inset_top.set_xticks(positions_top[positions_top > -1])
	ax_inset_top.set_xticks(top_ticks[top_ticks > -1])
	ax_inset_top.set_xticklabels(list(map(lambda x: "$%s$" % latex_float(x), top_ticks[top_ticks > -1])), rotation=-45, color=gv.Colors[1], rotation_mode="anchor", ha="left", va="baseline")	
	# ax_inset_top.minorticks_off()
	# ax_inset_top.set_xticks(raw_inset_ticks_top)
	# ax_inset_top.set_xticklabels(list(map(lambda x: "$%s$" % latex_float(x), raw_inset_ticks_top)), rotation=-45, color=gv.Colors[1], rotation_mode="anchor", ha="left", va="baseline")
	top_xlim = [np.min(raw_inset_ticks_top), np.max(raw_inset_ticks_top)]
	ax_inset_top.set_xlim(*top_xlim)
	# print("Inset uncorr ticks\n{}".format(ax_inset_top.get_xticks()))
	# print("xxxxxxxxxxxx")
	# return (raw_inset_ticks_bottom, raw_inset_ticks_top)
	return (bottom_ticks, top_ticks, bottom_xlim, top_xlim)


def DoubleHammerPlot(lmet, pmets, dsets, is_inset, nbins, thresholds):
	# Compare the effect of p_u + RC on predictability.
	# Plot no RC with infid and RC with p_u.
	# pmets = list(map(lambda phy: phy.strip(" "), phymets.split(",")))
	(ticks_bottom, ticks_top) = (None, None)
	framesize = (1.3 * gv.canvas_size[0], 1.3 * gv.canvas_size[1])
	plotfname = HammerPlot(dsets[0], lmet, pmets)
	with PdfPages(plotfname) as pdf:
		for l in range(1, 1 + dsets[0].levels):
			fig = plt.figure(figsize=framesize)
			ax_bottom = plt.gca()
			ax_top = ax_bottom.twiny()
			settings = [[], []]
			logerrs = np.zeros((len(dsets), dsets[0].channels), dtype = np.double)
			include = [[], []]
			for c in range(2):
				logerrs[c, :] = np.load(LogicalErrorRates(dsets[c], lmet))[:, l]
				if c == 0:
					phyerrs = LoadPhysicalErrorRates(dsets[c], pmets[c], None, l)
					xaxis = phyerrs
				else:
					uncorr = LoadPhysicalErrorRates(dsets[c], pmets[c], None, l)
					xaxis = uncorr
				# Compute the X-cutoff to include a subset of channels in the plot.
				xcutoff = GetXCutOff(xaxis, logerrs[c, :], thresholds[l - 1], nbins=50, space="log")
				include[c], = np.nonzero(np.logical_and(xaxis >= xcutoff["left"], xaxis <= xcutoff["right"]))

			for c in range(2):
				if (c == 0):
					# Plotting the logical error rates of the non RC channel vs. standard metrics
					current_axes = ax_bottom
					xaxis = phyerrs
					current_axes.xaxis.tick_top()
					current_axes.xaxis.set_label_position('top')
				else:
					# Plotting the logical error rates of the RC channel vs. uncorr
					current_axes = ax_top
					xaxis = uncorr
					current_axes.xaxis.tick_bottom()
					current_axes.xaxis.set_label_position('bottom')

				# Plot logical error rates vs. physical error metric
				current_axes.plot(
					xaxis[include[c]],
					logerrs[c, include[c]],
					color=gv.Colors[c % gv.n_Colors],
					alpha=0.75,
					marker=gv.Markers[c % gv.n_Markers],
					markersize=gv.marker_size,
					linestyle="None",
					linewidth=gv.line_width
				)

				# Empty plot for legend entries
				ax_bottom.plot([], [],
					color=gv.Colors[c % gv.n_Colors],
					alpha=0.75,
					marker=gv.Markers[c % gv.n_Markers],
					markersize=gv.marker_size,
					linestyle="None",
					linewidth=gv.line_width,
					label="%s %s"
					% (ml.Metrics[pmets[c]]["latex"], dsets[c].plotsettings["name"]),
				)

			# X = Y line for the top axis with the uncorr data.
			ax_top.plot(uncorr[include[1]], uncorr[include[1]], color="k", linestyle="solid", linewidth=gv.line_width)
			# Empty plot for the X = Y legend entry.
			ax_bottom.plot([], [], color="k", linestyle="solid", linewidth=gv.line_width, label="Ideal")
			
			# Flow lines only when the number of channels is less than 20
			if (dsets[0].channels <= 50):
				# The flow lines should connect the (infid, logical error under non-RC) to the (uncorr, logical error under RC) points.
				for i in range(dsets[0].channels):
					# Add a label with the noise rate and sample
					ax_top.annotate("%g, %d" % (dsets[1].available[i, 0], dsets[1].available[i, -1]), (uncorr[i], logerrs[1, i]), fontsize=gv.legend_fontsize)
					
				for i in range(len(include)):
					# ax_bottom.plot(
					# 	[phyerrs[include[c]], uncorr[include[c]]],
					# 	[logerrs[0, include[c]], logerrs[1, include[c]]],
					# 	color="0.5",
					# 	linestyle="dashed",
					# 	linewidth=gv.line_width
					# )
					nonRC_infid = [phyerrs[include[0][i]], logerrs[0, include[0][i]]]
					RC_uncorr = [uncorr[include[1][i]], logerrs[1, include[1][i]]]
					con = ConnectionPatch(xyA=nonRC_infid, xyB=RC_uncorr, coordsA="data", coordsB="data", axesA=ax_bottom, axesB=ax_top, color="0.7", linestyle="dashed", linewidth=gv.line_width)
					ax_bottom.add_artist(con)

			# Draw grid lines
			ax_bottom.grid(which="major")

			# Inset plot
			if (is_inset == 1):
				if (dsets[0].channels > 50):
					(ticks_bottom, ticks_top, bottom_xlim, top_xlim) = BinVariancePlot(ax_bottom, dsets, l, lmet, pmets, nbins, include)
				# BinVariancePlot(ax_bottom, dsets, l, lmet, pmets, nbins, include)
			
			## Axes labels
			# Axes label for the Y-axes
			ax_bottom.set_ylabel("$\\overline{%s_{%d}}$" % (ml.Metrics[lmet]["latex"].replace("$",""), l), fontsize=gv.axes_labels_fontsize * 1.7, labelpad = gv.axes_labelpad)
			# Axes labels for the bottom X-axes
			bottom_xlabel = "%s %s" % (ml.Metrics[pmets[0]]["latex"], dsets[0].plotsettings["name"])
			ax_bottom.set_xlabel(bottom_xlabel, fontsize=gv.axes_labels_fontsize * 1.7, labelpad = 0.5 * gv.axes_labelpad, color=gv.Colors[0])
			# Axes labels for the top axes
			top_xlabel = "%s %s" % (ml.Metrics[pmets[1]]["latex"], dsets[1].plotsettings["name"])
			ax_top.set_xlabel(top_xlabel, fontsize=gv.axes_labels_fontsize * 1.7, labelpad = 0.25 * gv.axes_labelpad * 2.5, color=gv.Colors[1])
			
			# Scales for the axes
			ax_bottom.set_xscale("log")
			ax_top.set_xscale("log")
			ax_bottom.set_yscale("log")

			## Locations and labels for the ticks
			if ticks_bottom is not None:
				# Locations and labels for the bottom X-axis ticks
				include_ticks, = np.nonzero(ticks_bottom > -1)
				ax_bottom.set_xticks(ticks_bottom[include_ticks])
				ax_bottom.set_xticklabels(list(map(lambda x: "$%s$" % latex_float(x), ticks_bottom[include_ticks])), rotation = 30, rotation_mode="anchor", ha="left", va="baseline")
				ax_bottom.set_xlim(*bottom_xlim)
				# ax_bottom.minorticks_off()
	
			print("Infid ticks\n{}".format(ax_bottom.get_xticks()))

			if ticks_top is not None:
				# Locations and labels for the top X-axis ticks
				include_ticks, = np.nonzero(ticks_top > -1)
				ax_top.set_xticks(ticks_top[include_ticks])
				ax_top.set_xticklabels(list(map(lambda x: "$%s$" % latex_float(x), ticks_top[include_ticks])), rotation = -30, rotation_mode="anchor", ha="left", va="baseline")
				ax_top.set_xlim(*top_xlim)
				# ax_top.minorticks_off()

			print("Uncorr ticks\n{}".format(ax_top.get_xticks()))
			
			# Locations and labels for the Y-axis ticks
			loc = LogLocator(base=10, numticks=10) # this locator puts ticks at regular intervals
			ax_bottom.yaxis.set_major_locator(loc)
			
			# Tick params for Y-axes
			ax_bottom.tick_params(
				axis="y",
				which="both",
				pad=gv.ticks_pad,
				direction="inout",
				length=gv.ticks_length,
				width=gv.ticks_width,
				labelsize=1.5 * gv.ticks_fontsize
			)
			# Tick params for the top and bottom X-axes
			relative_pads = [0.5, 1]
			for (a, ax) in enumerate([ax_bottom, ax_top]):
				ax.tick_params(
					axis="x",
					which="both",
					pad=relative_pads[a] * gv.ticks_pad,
					direction="inout",
					length=gv.ticks_length,
					width=gv.ticks_width,
					labelsize=1.5 * gv.ticks_fontsize,
					color=gv.Colors[a % gv.n_Colors],
				)
			
			# Color of the X-axis line				
			ax_top.spines['bottom'].set_color(gv.Colors[1])
			ax_top.spines['top'].set_color(gv.Colors[0])

			# Color of the tick labels
			for t in ax_bottom.xaxis.get_ticklabels(which="both"):
				t.set_color(gv.Colors[0])
			for t in ax_top.xaxis.get_ticklabels(which="both"):
				t.set_color(gv.Colors[1])
			# Force the tick lines to be black
			for t in ax_bottom.xaxis.get_majorticklines():
				t.set_color("k")
			for t in ax_bottom.xaxis.get_minorticklines():
				t.set_color("k")
			for t in ax_top.xaxis.get_majorticklines():
				t.set_color("k")
			for t in ax_top.xaxis.get_minorticklines():
				t.set_color("k")
			
			# Legend for the bottom axes
			leg = ax_bottom.legend(numpoints=1, loc="upper left", shadow=True, fontsize=1.5 * gv.legend_fontsize, markerscale=1.2 * gv.legend_marker_scale)
			
			# Match legend text with the color of the markers
			colors = [gv.Colors[0], gv.Colors[1]]
			for (color, text) in zip(colors, leg.get_texts()):
				text.set_color(color)

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