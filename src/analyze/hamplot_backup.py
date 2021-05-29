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
from define import fnames as fn
from define import metrics as ml
from define import globalvars as gv
from analyze.load import LoadPhysicalErrorRates
from analyze.bins import PlotBinVarianceDataSets, GetXCutOff
from analyze.utils import scientific_float, latex_float, SetTickLabels


def DoubleHammerPlot(logmet, phylist, dsets, inset_flag, nbins, thresholds):
	# Compare the effect of p_u + RC on predictability.
	# Plot no RC with infid and RC with p_u.
	# phylist = list(map(lambda phy: phy.strip(" "), phymets.split(",")))
	framesize = (1.3 * gv.canvas_size[0], 1.3 * gv.canvas_size[1])
	plotfname = fn.HammerPlot(dsets[0], logmet, phylist)
	with PdfPages(plotfname) as pdf:
		for l in range(1, 1 + dsets[0].levels):
			fig = plt.figure(figsize=framesize)
			ax_bottom = plt.gca()
			ax_top = ax_bottom.twiny()
			settings = [[], []]
			include = {}
			for c in range(2):
				settings[c] = {
					"xaxis": None,
					"xlabel": None,
					"yaxis": np.load(fn.LogicalErrorRates(dsets[c], logmet))[:, l],
					"ylabel": "$\\overline{%s_{%d}}$" % (ml.Metrics[logmet]["latex"].replace("$",""), l),
					"color": gv.Colors[c % gv.n_Colors],
					"marker": gv.Markers[c % gv.n_Markers],
					"linestyle": "",
				}
				LoadPhysicalErrorRates(dsets[c], phylist[c], settings[c], l)
				if c == 0:
					# print("Getting X cutoff for l = {}".format(l))
					xcutoff = GetXCutOff(
						settings[c]["xaxis"],
						settings[c]["yaxis"],
						thresholds[l - 1],
						nbins=50,
						space="log",
					)
					include[phylist[c]] = np.nonzero(
						np.logical_and(
							settings[c]["xaxis"] >= xcutoff["left"],
							settings[c]["xaxis"] <= xcutoff["right"],
						)
					)[0]
					# Plotting the logical error rates of the non RC channel vs. standard metrics
					current_axes = ax_bottom
					(ticks_bottom, tick_labels_bottom) = SetTickLabels(settings[c]["xaxis"][include[phylist[c]]])
					# print("Ticks bottom\n{}\nLabels Bottom\n{}".format(ticks_bottom, tick_labels_bottom))
				else:
					include[phylist[c]] = include[phylist[0]]
					# Plotting the logical error rates of the RC channel vs. uncorr
					current_axes = ax_top
					(ticks_top, tick_labels_top) = SetTickLabels(settings[c]["xaxis"][include[phylist[c]]])

				# LoadPhysicalErrorRates(dsets[c], phylist[c], settings[c], l)

				current_axes.plot(
					settings[c]["xaxis"][include[phylist[c]]],
					settings[c]["yaxis"][include[phylist[c]]],
					color=settings[c]["color"],
					alpha=0.75,
					marker=settings[c]["marker"],
					markersize=gv.marker_size,
					linestyle=settings[c]["linestyle"],
					linewidth=gv.line_width,
					# label="%s %s"
					# % (ml.Metrics[phylist[c]]["latex"], dsets[c].plotsettings["name"]),
				)

				# Empty plot for legend entries
				ax_bottom.plot([], [],
					color=settings[c]["color"],
					alpha=0.75,
					marker=settings[c]["marker"],
					markersize=gv.marker_size,
					linestyle=settings[c]["linestyle"],
					linewidth=gv.line_width,
					label="%s %s"
					% (ml.Metrics[phylist[c]]["latex"], dsets[c].plotsettings["name"]),
				)
			
			# X = Y line for the top axis with the uncorr data.
			xaxis = settings[1]["xaxis"][include[phylist[1]]]
			ax_top.plot(
				xaxis,
				xaxis,
				color="k",
				linestyle="solid",
				linewidth=gv.line_width,
			)

			# Inset plot
			(ticks_bottom, ticks_top) = PlotBinVarianceDataSets(ax_bottom, dsets, l, logmet, phylist, nbins, include, inset_flag, ticks_bottom, ticks_top)
			# PlotBinVarianceDataSets(ax_bottom, dsets, l, logmet, phylist, nbins, include, inset_flag, ticks_bottom, ticks_top) # Use this only for the diamond distance plot.
			# print("ticks_bottom = {}\nticks_top = {}".format(ticks_bottom, ticks_top))
			
			# Axes labels for the bottom axes
			bottom_xlabel = "%s %s" % (ml.Metrics[phylist[0]]["latex"], dsets[0].plotsettings["name"])
			ax_bottom.set_xlabel(bottom_xlabel, fontsize=gv.axes_labels_fontsize * 1.7, labelpad = 0.5 * gv.axes_labelpad, color=settings[0]["color"])
			ax_bottom.set_ylabel(settings[0]["ylabel"], fontsize=gv.axes_labels_fontsize * 1.7, labelpad = gv.axes_labelpad)
			
			# Axes labels for the top axes
			top_xlabel = "%s %s" % (ml.Metrics[phylist[1]]["latex"], dsets[1].plotsettings["name"])
			ax_top.set_xlabel(top_xlabel, fontsize=gv.axes_labels_fontsize * 1.7, labelpad = gv.axes_labelpad * 2.5, color=settings[1]["color"])
			
			# Grid
			ax_bottom.grid(which="both")

			# Scales for the axes
			ax_bottom.set_xscale("log")
			ax_top.set_xscale("log")
			ax_bottom.set_yscale("log")

			# Set a Y-axes limit
			# ax_bottom.set_ylim([None, 0.1]) # Display the inset plot.
			
			# Locations and labels for the X-axis ticks
			include_ticks, = np.nonzero(ticks_bottom > -1)
			ax_bottom.set_xticks(ticks_bottom[include_ticks])
			ax_bottom.set_xticklabels([tick_labels_bottom[tk] for tk in include_ticks], rotation = -30, rotation_mode="anchor", ha="left", va="baseline")
			
			include_ticks, = np.nonzero(ticks_top > -1)
			ax_top.set_xticks(ticks_top[include_ticks])
			ax_top.set_xticklabels([tick_labels_top[tk] for tk in include_ticks])
			
			# print("Bottom ticks for the main plot\n{}\nTop ticks for the main plot\n{}".format(list(ax_bottom.xaxis.get_ticklabels()), list(ax_top.xaxis.get_ticklabels())))
			
			# Locations and labels for the Y-axis ticks
			loc = LogLocator(base=10, numticks=10) # this locator puts ticks at regular intervals
			ax_bottom.yaxis.set_major_locator(loc)
			# ax_bottom.xaxis.set_major_locator(loc)
			# ax_top.xaxis.set_major_locator(loc)

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
			tick_colors = [settings[0]["color"], settings[1]["color"]]
			relative_pads = [1, 0.5]
			for (a, ax) in enumerate([ax_bottom, ax_top]):
				# Tick params for the top and bottom X-axes
				ax.tick_params(
					axis="x",
					which="both",
					pad=relative_pads[a] * gv.ticks_pad,
					direction="inout",
					length=gv.ticks_length,
					width=gv.ticks_width,
					labelsize=1.5 * gv.ticks_fontsize,
					color=tick_colors[a],
				)
			
			# Color of the X-axis line				
			ax_top.spines['bottom'].set_color(tick_colors[0])
			ax_top.spines['top'].set_color(tick_colors[1])

			# Color of the tick labels
			for t in ax_bottom.xaxis.get_ticklabels():
				t.set_color(tick_colors[0])
			for t in ax_top.xaxis.get_ticklabels():
				t.set_color(tick_colors[1])
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
			leg = ax_bottom.legend(
				numpoints=1,
				loc="upper left",
				shadow=True,
				fontsize=gv.legend_fontsize,
				markerscale=gv.legend_marker_scale,
			)
			
			# Match legend text with the color of the markers
			colors = [settings[0]["color"], settings[1]["color"]]
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


def NewDoubleHammerPlot(lmet, pmets, dsets, is_inset, nbins, thresholds):
	# Compare the effect of p_u + RC on predictability.
	# Plot no RC with infid and RC with p_u.
	# pmets = list(map(lambda phy: phy.strip(" "), phymets.split(",")))
	framesize = (1.3 * gv.canvas_size[0], 1.3 * gv.canvas_size[1])
	plotfname = fn.HammerPlot(dsets[0], lmet, pmets)
	with PdfPages(plotfname) as pdf:
		for l in range(1, 1 + dsets[0].levels):
			fig = plt.figure(figsize=framesize)
			ax_bottom = plt.gca()
			ax_top = ax_bottom.twiny()
			settings = [[], []]
			include = {}
			for c in range(2):
				settings[c] = {
					"xaxis": None,
					"xlabel": None,
					"yaxis": np.load(fn.LogicalErrorRates(dsets[c], lmet))[:, l],
					"ylabel": "$\\overline{%s_{%d}}$" % (ml.Metrics[lmet]["latex"].replace("$",""), l),
					"color": gv.Colors[c % gv.n_Colors],
					"marker": gv.Markers[c % gv.n_Markers],
					"linestyle": "",
				}
				LoadPhysicalErrorRates(dsets[c], pmets[c], settings[c], l)
				if c == 0:
					# print("Getting X cutoff for l = {}".format(l))
					xcutoff = GetXCutOff(settings[c]["xaxis"], settings[c]["yaxis"], thresholds[l - 1], nbins=50, space="log")
					include[pmets[c]], = np.nonzero(np.logical_and(settings[c]["xaxis"] >= xcutoff["left"], settings[c]["xaxis"] <= xcutoff["right"]))
					# Plotting the logical error rates of the non RC channel vs. standard metrics
					current_axes = ax_bottom
					# print("Ticks bottom\n{}\nLabels Bottom\n{}".format(ticks_bottom, tick_labels_bottom))
				else:
					include[pmets[c]] = include[pmets[0]]
					# Plotting the logical error rates of the RC channel vs. uncorr
					current_axes = ax_top

				# Plot logical error rates vs. physical error metric
				current_axes.plot(
					settings[c]["xaxis"][include[pmets[c]]],
					settings[c]["yaxis"][include[pmets[c]]],
					color=settings[c]["color"],
					alpha=0.75,
					marker=settings[c]["marker"],
					markersize=gv.marker_size,
					linestyle=settings[c]["linestyle"],
					linewidth=gv.line_width
				)

				# Empty plot for legend entries
				ax_bottom.plot([], [],
					color=settings[c]["color"],
					alpha=0.75,
					marker=settings[c]["marker"],
					markersize=gv.marker_size,
					linestyle=settings[c]["linestyle"],
					linewidth=gv.line_width,
					label="%s %s"
					% (ml.Metrics[pmets[c]]["latex"], dsets[c].plotsettings["name"]),
				)
			
			# X = Y line for the top axis with the uncorr data.
			xaxis = settings[1]["xaxis"][include[pmets[1]]]
			ax_top.plot(
				xaxis,
				xaxis,
				color="k",
				linestyle="solid",
				linewidth=gv.line_width,
			)
			# Empty plot on the bottom axis to add a label for the X=Y line.
			ax_top.plot(
				[], [],
				color="k",
				linestyle="solid",
				linewidth=gv.line_width,
				label="X = Y"
			)

			# Inset plot
			(ticks_bottom, ticks_top) = NewPlotBinVarianceDataSets(ax_bottom, dsets, l, lmet, pmets, nbins, include, is_inset)
			
			## Axes labels
			# Axes label for the Y-axes
			ax_bottom.set_ylabel(settings[0]["ylabel"], fontsize=gv.axes_labels_fontsize * 1.7, labelpad = gv.axes_labelpad)
			# Axes labels for the bottom X-axes
			bottom_xlabel = "%s %s" % (ml.Metrics[pmets[0]]["latex"], dsets[0].plotsettings["name"])
			ax_bottom.set_xlabel(bottom_xlabel, fontsize=gv.axes_labels_fontsize * 1.7, labelpad = 0.5 * gv.axes_labelpad, color=settings[0]["color"])
			# Axes labels for the top axes
			top_xlabel = "%s %s" % (ml.Metrics[pmets[1]]["latex"], dsets[1].plotsettings["name"])
			ax_top.set_xlabel(top_xlabel, fontsize=gv.axes_labels_fontsize * 1.7, labelpad = gv.axes_labelpad * 2.5, color=settings[1]["color"])
			
			# Scales for the axes
			ax_bottom.set_xscale("log")
			ax_top.set_xscale("log")
			ax_bottom.set_yscale("log")

			## Locations and labels for the ticks
			# Locations and labels for the bottom X-axis ticks
			include_ticks, = np.nonzero(ticks_bottom > -1)
			ax_bottom.set_xticks(ticks_bottom[include_ticks])
			ax_bottom.set_xticklabels([tick_labels_bottom[tk] for tk in include_ticks], rotation = -30, rotation_mode="anchor", ha="left", va="baseline")
			# Locations and labels for the top X-axis ticks
			include_ticks, = np.nonzero(ticks_top > -1)
			ax_top.set_xticks(ticks_top[include_ticks])
			ax_top.set_xticklabels([tick_labels_top[tk] for tk in include_ticks])
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
			tick_colors = [settings[0]["color"], settings[1]["color"]]
			relative_pads = [1, 0.5]
			for (a, ax) in enumerate([ax_bottom, ax_top]):
				ax.tick_params(
					axis="x",
					which="both",
					pad=relative_pads[a] * gv.ticks_pad,
					direction="inout",
					length=gv.ticks_length,
					width=gv.ticks_width,
					labelsize=1.5 * gv.ticks_fontsize,
					color=tick_colors[a],
				)
			
			# Color of the X-axis line				
			ax_top.spines['bottom'].set_color(tick_colors[0])
			ax_top.spines['top'].set_color(tick_colors[1])

			# Color of the tick labels
			for t in ax_bottom.xaxis.get_ticklabels():
				t.set_color(tick_colors[0])
			for t in ax_top.xaxis.get_ticklabels():
				t.set_color(tick_colors[1])
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
			leg = ax_bottom.legend(
				numpoints=1,
				loc="upper left",
				shadow=True,
				fontsize=gv.legend_fontsize,
				markerscale=gv.legend_marker_scale,
			)
			
			# Match legend text with the color of the markers
			colors = [settings[0]["color"], settings[1]["color"]]
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


def PartialHammerPlot(logmet, phylist, dsets, inset_flag, nbins, thresholds):
	# Compare the effect of p_u + RC on predictability.
	# Plot no RC with infid and RC with p_u.
	# phylist = list(map(lambda phy: phy.strip(" "), phymets.split(",")))
	level = dsets[0].levels
	ndb = len(dsets)
	plotfname = fn.PartialHammerPlotFile(dsets[0], logmet, phylist)
	with PdfPages(plotfname) as pdf:
		fig = plt.figure(figsize=(gv.canvas_size[0], gv.canvas_size[1]*1.2))
		ax = plt.gca()
		settings = [[] for __ in range(ndb)]
		include = {}
		for d in range(ndb):
			# print("Getting X cutoff for alpha = {} level = {}".format(dsets[d].decoder_fraction, level))
			if phylist[d] == "uncorr":
				xaxis = np.load(fn.PhysicalErrorRates(dsets[d], phylist[d]))[:, level]
			else:
				xaxis = np.load(fn.PhysicalErrorRates(dsets[d], phylist[d]))
				xcutoff = GetXCutOff(
					xaxis,
					np.load(fn.LogicalErrorRates(dsets[d], logmet))[:, level],
					thresholds[level - 1],
					nbins=50,
					space="log"
				)
			include[phylist[d]] = np.nonzero(np.logical_and(xaxis >= xcutoff["left"], xaxis <= xcutoff["right"]))[0]
			# include[phylist[d]] = include[phylist[0]]

		PlotBinVarianceDataSets(ax, dsets, level, logmet, phylist, nbins, include, is_inset=inset_flag)

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