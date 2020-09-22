import os
import sys
import datetime as dt

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from define import fnames as fn
from define import globalvars as gv
from define import metrics as ml
from analyze.bins import GetXCutOff, ComputeBinVariance, CollapseBins
from analyze.load import LoadPhysicalErrorRates

def PlotDeviationYX(lmet, pmet, dbses, nbins, thresholds):
	# Compare scatter for different physical metrics
	min_bin_fraction = 0.3
	ndb = len(dbses)
	maxlevel = max([dbses[i].levels for i in range(ndb)])
	plotfname = fn.DeviationPlotFile(dbses[0], pmet, lmet)
	with PdfPages(plotfname) as pdf:
		for level in range(1, 1 + maxlevel):
			fig = plt.figure(figsize=gv.canvas_size)
			ax = plt.gca()
			phyerrs = np.zeros((ndb, dbses[0].channels), dtype=np.double)
			logerrs = np.zeros((ndb, dbses[0].channels), dtype=np.double)
			collapsed_bins = [None for d in range(ndb)]
			for d in range(ndb):
				settings = {
					"xaxis": None,
					"xlabel": None,
					"yaxis": np.load(fn.LogicalErrorRates(dbses[d], lmet))[:, level],
					"ylabel": "$\\overline{%s_{%d}}$"
					% (ml.Metrics[lmet]["latex"].replace("$", ""), level),
					"color": gv.Colors[d % gv.n_Colors]
					if ndb > 1
					else ml.Metrics[pmet]["color"],
					"marker": gv.Markers[d % gv.n_Markers]
					if ndb > 1
					else ml.Metrics[pmet]["marker"],
					"linestyle": "",
				}
				LoadPhysicalErrorRates(dbses[d], pmet, settings, level)
				phyerrs[d, :] = settings["xaxis"]
				logerrs[d, :] = settings["yaxis"]
				if d == 0:
				    # print("Getting X cutoff for l = {}".format(l))
				    xcutoff = GetXCutOff(
				        settings["xaxis"],
				        settings["yaxis"],
				        thresholds[level - 1],
				        nbins=50,
				        space="log",
				    )
				    include = np.nonzero(
				        np.logical_and(
				            settings["xaxis"] >= xcutoff["left"],
				            settings["xaxis"] <= xcutoff["right"],
				        )
				    )[0]
				bins = ComputeBinVariance(
					phyerrs[d, include], logerrs[d, include], space="log", nbins=nbins
				)

				collapsed_bins[d] = CollapseBins(
					bins, min_bin_fraction * dbses[d].channels / nbins
				)

				# print(
				#     "\033[2mnbins for level {} = {}\nPoints in bins = {}\nAverage points in a bin = {}\nthreshold: {}\n----\033[0m".format(
				#         level,
				#         collapsed_bins[d].shape[0],
				#         collapsed_bins[d][:, 2],
				#         np.mean(bins[:, 2]),
				#         min_bin_fraction * dbses[d].channels / nbins,
				#     )
				# )

				xaxis = collapsed_bins[d][:, 6]
				# xaxis = (bins[pmets[p]][:, 0] + bins[pmets[p]][:, 1]) / 2
				yaxis = np.abs(collapsed_bins[d][:, 6] - collapsed_bins[d][:, 7])/collapsed_bins[d][:, 6]
				# print("Values for dvplot , xaxis = {}, yaxis = yaxis".format(xaxis,yaxis))
				plotobj = plt.plot(
					xaxis,
					yaxis,
					color=settings["color"],
					alpha=0.75,
					marker=settings["marker"],
					markersize=gv.marker_size,
					linestyle="--",
					linewidth=gv.line_width,
					label=dbses[d].plotsettings["name"],
				)

			# Axes
			ax.set_xlabel("$\\overline{%s_{%d}}$" % (ml.Metrics[lmet]["latex"].replace("$", ""), level), fontsize=gv.axes_labels_fontsize)
			ax.set_ylabel("$\\frac{|\\overline{%s_{%d}} - %s|}{\\overline{%s_{%d}}}$" % (ml.Metrics[lmet]["latex"].replace("$", ""), level, ml.Metrics[pmet]["latex"].replace("$",""),ml.Metrics[lmet]["latex"].replace("$", ""), level), fontsize=gv.axes_labels_fontsize)
			# ax.set_ylim([10e-9, None])
			ax.set_yscale("log")
			ax.set_xscale("log")
			ax.tick_params(
				axis="both",
				which="both",
				# pad=gv.ticks_pad,
				direction="inout",
				length=gv.ticks_length,
				width=gv.ticks_width,
				labelsize=gv.ticks_fontsize,
			)
			# ax.set_xticks(np.arange(0, collapsed_bins[1].shape[0], dtype=np.int))
			# ax.set_xticklabels(
			# 	list(
			# 		map(
			# 			lambda num: "%s" % scientific_float(num),
			# 			(collapsed_bins[1][:, 0] + collapsed_bins[1][:, 1]) / 2,
			# 		)
			# 	),
			# 	rotation=45,
			# 	color=gv.Colors[1],
			# )
			# Save the plot
			plt.legend(
				numpoints=1,
				loc="best",  # center_left
				shadow=True,
				fontsize=gv.legend_fontsize,
				markerscale=gv.legend_marker_scale,
			)
			pdf.savefig(fig)
			plt.close()
		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Typical deviation of Y from X."
		pdfInfo["Author"] = "Pavithran Iyer"
		pdfInfo["ModDate"] = dt.datetime.today()
	return None
