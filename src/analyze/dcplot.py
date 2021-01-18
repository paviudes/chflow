# Critical packages
import os
import sys
import datetime as dt
import numpy as np
from scipy.special import comb
import matplotlib

matplotlib.use("Agg")
from matplotlib import colors, ticker, cm
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset
from scipy.interpolate import griddata

# Functions from other modules
from define import metrics as ml
from define import globalvars as gv
from define.decoder import GetTotalErrorBudget
from analyze.utils import latex_float
from analyze.bins import ComputeBinVariance
from analyze.load import LoadPhysicalErrorRates
from define.fnames import DecodersPlot, DecodersInstancePlot, LogicalErrorRates, PhysicalErrorRates


def DecoderCompare(
	phymet, logmet, dbses, nbins=10, thresholds={"y": 10e-16, "x": 10e-16}
):
	# Compare performance of various decoders.
	ndb = len(dbses)
	plotfname = DecodersPlot(dbses[0], phymet, logmet)
	nlevels = max([db.levels for db in dbses])
	with PdfPages(plotfname) as pdf:
		for l in range(1, nlevels + 1):
			fig = plt.figure(figsize=gv.canvas_size)
			ax1 = plt.gca()
			for d in range(ndb):
				settings = {
					"xaxis": None,
					"xlabel": None,
					"yaxis": np.load(LogicalErrorRates(dbses[d], logmet))[:, l],
					"ylabel": "$\\overline{%s_{%d}}$"
					% (ml.Metrics[logmet]["latex"].replace("$", ""), l),
					"color": gv.Colors[d % gv.n_Colors]
					if ndb > 1
					else ml.Metrics[phylist[p]]["color"],
					"marker": gv.Markers[d % gv.n_Markers]
					if ndb > 1
					else ml.Metrics[phylist[p]]["marker"],
					"linestyle": "",
				}
				if dbses[d].decoders[l - 1] == 0:
					decoder_info = "Maximum likelihood"
				elif dbses[d].decoders[l - 1] == 1:
					decoder_info = "Minimum weight"
				else:
					decoder_info = "$\\alpha = %g$" % (dbses[d].decoder_fraction)
				LoadPhysicalErrorRates(dbses[d], phymet, settings, l)
				bins = ComputeBinVariance(
					settings["xaxis"], settings["yaxis"], nbins=nbins, space="log"
				)
				# Plotting
				plotobj = ax1.plot(
					(bins[:, 0] + bins[:, 1]) / 2,
					bins[:, 6],
					color=settings["color"],
					alpha=0.75,
					marker="o",  # settings["marker"]
					markersize=gv.marker_size,
					linestyle=settings["linestyle"],
					linewidth=gv.line_width,
					label=decoder_info,
				)
			# Axes labels
			ax1.set_xlabel(
				settings["xlabel"],
				fontsize=gv.axes_labels_fontsize * 0.8,
				labelpad=gv.axes_labelpad,
			)
			ax1.set_xscale("log")
			ax1.set_ylabel(
				settings["ylabel"],
				fontsize=gv.axes_labels_fontsize,
				labelpad=gv.axes_labelpad,
			)
			ax1.set_yscale("log")
			ax1.tick_params(
				axis="both",
				which="both",
				pad=gv.ticks_pad,
				direction="inout",
				length=gv.ticks_length,
				width=gv.ticks_width,
				labelsize=gv.ticks_fontsize,
			)
			ax1.legend(
				numpoints=1,
				loc="lower right",
				shadow=True,
				fontsize=gv.legend_fontsize,
				markerscale=gv.legend_marker_scale,
			)
			# Save the plot
			pdf.savefig(fig)
			plt.close()
		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Comparing different decoders"
		pdfInfo["Author"] = "Pavithran Iyer"
		pdfInfo["ModDate"] = dt.datetime.today()
	return None


def DecoderInstanceCompare(
	phymet, logmet, dbses_input, chids = [0], thresholds={"y": 10e-16, "x": 10e-16}
):
	# Compare performance of various decoders.
	# Only show alpha values that correspond to distinct number of Pauli error rates.
	max_weight = 1 + dbses_input[0].eccs[0].N//2
	(budgets, uniques) = np.unique(np.array([GetTotalErrorBudget(dbs, dbs.available[chids[0], :-1], int(dbs.available[chids[0], -1])) for dbs in dbses_input[1:]], dtype=np.int), return_index=True)
	# Add the entries for the minimum weight decoder.
	if (np.prod(dbses_input[0].decoders) == 1):
		budgets = np.concatenate(([0], budgets))
		uniques = np.concatenate(([0], 1 + uniques))

	# print("budgets = {}".format(budgets))

	dbses = [dbses_input[d] for d in uniques]
	ndb = len(dbses)
	plotfname = DecodersInstancePlot(dbses_input[0], phymet, logmet)
	nlevels = max([db.levels for db in dbses])
	phyerrs = np.load(PhysicalErrorRates(dbses[0], phymet))

	# max_budget = np.sum([comb(dbses[0].eccs[0].N, i) * 3**i for i in range(max_weight + 1)])/4**dbses[0].eccs[0].N

	with PdfPages(plotfname) as pdf:
		for l in range(1, nlevels + 1):
			fig = plt.figure(figsize=gv.canvas_size)
			ax1 = plt.gca()
			ax1.plot(
				[],
				[],
				color="k",
				linestyle="--",
				label="$%s = %s$"
				% (
					ml.Metrics[phymet]["latex"].replace("$", ""),
					latex_float(phyerrs[chids[0]]),
				),
			)
			for (c, ch) in enumerate(chids):
				settings = {
					"xaxis": [],
					"xlabel": "$\\alpha$",
					"yaxis": [],
					"ylabel": "$\\overline{%s_{%d}}$"
					% (ml.Metrics[logmet]["latex"].replace("$", ""), l),
					"color": gv.Colors[c % gv.n_Colors],
					"marker": gv.Markers[c % gv.n_Markers],
					"linestyle": "--",
				}
				for d in range(ndb):
					if dbses[d].decoders[l - 1] == 0:
						ax1.axhline(
							y=np.load(LogicalErrorRates(dbses[d], logmet))[ch, l],
							linestyle="--",
							linewidth=gv.line_width,
							color="green",
							label="MLD",
						)
					elif dbses[d].decoders[l - 1] == 1:
						minwt = np.load(LogicalErrorRates(dbses[d], logmet))[ch, l]
						ax1.axhline(
							y=np.load(LogicalErrorRates(dbses[d], logmet))[ch, l],
							linestyle="--",
							linewidth=gv.line_width,
							color="red",
							label="MWD",
						)
					else:
						settings["xaxis"].append(budgets[d]/(4**dbses[0].eccs[0].N))
						settings["yaxis"].append(
							np.load(LogicalErrorRates(dbses[d], logmet))[ch, l]
						)
						# print("{} --- {}".format(int(dbses[d].decoder_fraction * (4 ** dbses[0].eccs[0].N)), dbses[d].timestamp))
				sortorder = np.argsort(settings["xaxis"])
				settings["xaxis"] = np.array(settings["xaxis"])[sortorder]
				settings["yaxis"] = np.array(settings["yaxis"])[sortorder]
				# Plotting
				# print("X: {}\nY: {}\nMWD: {}".format(settings["xaxis"], settings["yaxis"], minwt))
				plotobj = ax1.plot(
					settings["xaxis"],
					settings["yaxis"],
					color=settings["color"],
					alpha=0.75,
					marker="o",  # settings["marker"]
					markersize=gv.marker_size,
					linestyle=settings["linestyle"],
					linewidth=gv.line_width,
					# label="$%s = %s$"
					# % (
					#     ml.Metrics[phymet]["latex"].replace("$", ""),
					#     latex_float(phyerrs[ch]),
					# ),
					label="$\\mathcal{D}_{\\alpha}$",
				)

				# print("X axis for dcplot\n{}".format(settings["xaxis"]))

				for i in range(len(settings["xaxis"])):
					ax1.annotate(
						"%d" % (budgets[i + 1]),
						(settings["xaxis"][i] * 1.02, settings["yaxis"][i] * 1.02),
						color="k",
						fontsize=gv.ticks_fontsize,
						rotation=30
					)
			# Axes labels
			ax1.set_xlabel(
				settings["xlabel"],
				fontsize=gv.axes_labels_fontsize * 0.8,
				labelpad=gv.axes_labelpad,
			)
			ax1.set_xscale("log")
			ax1.set_ylabel(
				settings["ylabel"],
				fontsize=gv.axes_labels_fontsize,
				labelpad=gv.axes_labelpad,
			)
			ax1.set_yscale("log")
			ax1.tick_params(
				axis="both",
				which="both",
				pad=gv.ticks_pad,
				direction="inout",
				length=gv.ticks_length,
				width=gv.ticks_width,
				labelsize=gv.ticks_fontsize,
			)
			ax1.legend(
				numpoints=1,
				loc="best",
				shadow=True,
				fontsize=gv.legend_fontsize,
				markerscale=gv.legend_marker_scale,
			)
			# Save the plot
			pdf.savefig(fig)
			plt.close()
		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Comparing different decoders"
		pdfInfo["Author"] = "Pavithran Iyer"
		pdfInfo["ModDate"] = dt.datetime.today()
	return None
