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


def DoubleHammerPlot(logmet, phylist, dsets, inset_flag, nbins, thresholds):
	# Compare the effect of p_u + RC on predictability.
	# Plot no RC with infid and RC with p_u.
	# phylist = list(map(lambda phy: phy.strip(" "), phymets.split(",")))
	plotfname = fn.HammerPlot(dsets[0], logmet, phylist)
	with PdfPages(plotfname) as pdf:
		for l in range(1, 1 + dsets[0].levels):
			fig = plt.figure(figsize=gv.canvas_size)
			ax1 = plt.gca()
			ax_top = ax1.twiny()
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
				if c == 0:
					LoadPhysicalErrorRates(dsets[c], phylist[c], settings[c], l)
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
					current_axes = ax1
				else:
					include[phylist[c]] = include[phylist[0]]
					LoadPhysicalErrorRates(dsets[c], phylist[c], settings[c], l)
					# Plotting the logical error rates of the RC channel vs. uncorr
					current_axes = ax_top

				current_axes.plot(
					settings[c]["xaxis"][include[phylist[c]]],
					settings[c]["yaxis"][include[phylist[c]]],
					color=settings[c]["color"],
					alpha=0.75,
					marker=settings[c]["marker"],
					markersize=gv.marker_size,
					linestyle=settings[c]["linestyle"],
					linewidth=gv.line_width,
					label="%s %s"
					% (ml.Metrics[phylist[c]]["latex"], dsets[c].plotsettings["name"]),
				)
				
			PlotBinVarianceDataSets(ax1, dsets, l, logmet, phylist, nbins, include)

			# Axes labels for the bottom axes
			bottom_xlabel = "%s %s" % (ml.Metrics[phylist[0]]["latex"], dsets[0].plotsettings["name"])
			ax1.set_xlabel(bottom_xlabel, fontsize=gv.axes_labels_fontsize * 0.8, labelpad = gv.axes_labelpad)
			ax1.set_ylabel(settings[0]["ylabel"], fontsize=gv.axes_labels_fontsize, labelpad = gv.axes_labelpad)
			
			# Axes labels for the top axes
			top_xlabel = "%s %s" % (ml.Metrics[phylist[1]]["latex"], dsets[1].plotsettings["name"])
			ax_top.set_xlabel(top_xlabel, fontsize=gv.axes_labels_fontsize * 0.8, labelpad = gv.axes_labelpad)
			
			# Scales for the axes
			ax1.set_xscale("log")
			ax_top.set_xscale("log")
			ax1.set_yscale("log")
			
			# Ticks and legend
			legend_locations = ["upper_left", "lower_right"]
			for (a, ax) in enumerate([ax1, ax_top]):
				# Tick params for the top and bottom axes
				ax.tick_params(
					axis="both",
					which="both",
					pad=gv.ticks_pad,
					direction="inout",
					length=gv.ticks_length,
					width=gv.ticks_width,
					labelsize=gv.ticks_fontsize,
				)
				loc = LogLocator(base=10, numticks=10) # this locator puts ticks at regular intervals
				ax.xaxis.set_major_locator(loc)
				# Legend both axes
				ax.legend(
					numpoints=1,
					loc=legend_locations[a],
					shadow=True,
					fontsize=gv.legend_fontsize,
					markerscale=gv.legend_marker_scale,
				)
			
			# Save the plot
			pdf.savefig(fig)
			plt.close()
		
		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Hammer plot."
		pdfInfo["Author"] = "Pavithran Iyer"
		pdfInfo["ModDate"] = dt.datetime.today()
	return None
