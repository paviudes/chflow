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
from analyze.bins import GetXCutOff, ComputeBinVariance
from analyze.load import LoadPhysicalErrorRates


def PlotDeviationYX(dbses, lmet, pmet, thresholds, nbins):
	# Compare scatter for different physical metrics
	min_bin_fraction = 0.1
	ax = plt.axes([0, 0, 1, 1])
	ndb = len(dbses)
	maxlevel = max([dsets[i].levels for i in range(ndb)])
	plotfname = fn.DeviationPlotFile(dsets[0], pmet, lmet)
	with PdfPages(plotfname) as pdf:
		for level in range(1, 1 + maxlevel):
			phyerrs = np.zeros((ndb, dbses[0].channels), dtype=np.double)
			logerrs = np.zeros((ndb, dbses[0].channels), dtype=np.double)
			collapsed_bins = [None for d in range(ndb)]
			for d in range(ndb):
				if pmet == "uncorr":
					phyerrs[d, :] = np.load(fn.PhysicalErrorRates(dbses[d], pmet))[:, level]
				else:
					phyerrs[d, :] = np.load(fn.PhysicalErrorRates(dbses[d], pmet))
				logerrs[d, :] = np.load(fn.LogicalErrorRates(dbses[d], lmet))[:, level]

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

				xaxis = np.arange(collapsed_bins[d].shape[0])
				# xaxis = (bins[pmets[p]][:, 0] + bins[pmets[p]][:, 1]) / 2
				yaxis = np.abs(collapsed_bins[d][:, 6] - collapsed_bins[d][:, 7])
			
			# Axes
			ax.set_xlabel("%s" % (ml.Metrics[pmet]["latex"]), fontsize=gv.axes_labels_fontsize)
			ax.set_ylabel("$|\\overline{%s_{%d}} - %s|$" % (ml.Metrics[logmet]["latex"].replace("$", ""), level, ml.Metrics[pmet]["latex"].replace("$","")), fontsize=gv.axes_labels_fontsize)
			# ax.set_ylim([10e-9, None])
			ax.set_yscale("log")
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
			pdf.savefig(fig)
			plt.close()
		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Typical deviation of Y from X." % (
		    ml.Metrics[logmet]["log"],
		    ", ".join(map(str, range(1, 1 + maxlevel))),
		    ", ".join(phynames),
		    dsets[0].channels,
		)
		pdfInfo["Author"] = "Pavithran Iyer"
		pdfInfo["ModDate"] = dt.datetime.today()
	return None