# Critical packages
import os
import sys
import numpy as np
import datetime as dt
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Non critical packages
try:
	import PyPDF2 as pp
except ImportError:
	pass

# Functions from other modules
from define import globalvars as gv
from define import metrics as ml
from define.fnames import PhysicalErrorRates, LogicalErrorRates, CompareSubsPlot
from analyze.load import LoadPhysicalErrorRates
from analyze.statplot import IsConverged, GetChannelPosition
from analyze.utils import OrderOfMagnitude

def CompareSubs(pmet, lmet, *dbses):
	# Compare the Logical error rates from two submissions.
	# The comparision only makes sense when the logical error rates are measured for two submissions that have the same physical channels.
	MIN = 1E-30
	ndb = len(dbses)
	nlevels = min([dbs.levels for dbs in dbses])

	plotfname = CompareSubsPlot(dbses[0], [dbs.timestamp for dbs in dbses[1:]])
	with PdfPages(plotfname) as pdf:
		ylimits = {"left": {"min": 1, "max": 0}, "right": {"min": 1, "max": 0}}
		for l in range(1, nlevels + 1):
			fig = plt.figure(figsize=(gv.canvas_size[0]*1.4, gv.canvas_size[1]*1.25))
			ax = plt.gca()
			# ax_right = ax.twinx()
			for d in range(ndb):
				# Plot multiple logical error rates, with respect to the same physical error rates.
				# We use linestyles to distinguish between codes, and colors/markers to distinguish between y-axis metrics.
				rates = np.arange(dbses[d].noiserates.shape[0], dtype = np.int)
				(converged_rates, __) = np.nonzero(IsConverged(dbses[d], lmet, rates, [0], threshold = 10))
				
				# print("Dataset: {}\nConvereged rates\n{}".format(dbses[d].eccs[0].name, converged_rates))

				convereged_channels = GetChannelPosition(dbses[d].noiserates[converged_rates], [0], dbses[d].available).flatten()

				# print("convereged_channels\n{}".format(convereged_channels))

				if os.path.isfile(LogicalErrorRates(dbses[d], lmet)):
					logerrs = np.load(LogicalErrorRates(dbses[d], lmet))[: , l]
				else:
					nchannels = dbses[d].noiserates.shape[0] * dbses[d].samps
					logerrs = np.zeros(nchannels, dtype = np.double)
				settings = {"xaxis": None, "xlabel": None, "yaxis": logerrs, "ylabel": "$\\overline{%s_{%d}}$" % (ml.Metrics[lmet]["latex"].replace("$", ""), l)}
				LoadPhysicalErrorRates(dbses[0], pmet, settings, l)
				
				# Store only the convereged channels
				settings["xaxis"] = settings["xaxis"][convereged_channels]
				settings["yaxis"] = settings["yaxis"][convereged_channels]

				# print("X size = {}: {}\nY size {}: {}".format(len(settings["xaxis"]), settings["xaxis"], len(settings["yaxis"]), settings["yaxis"]))

				settings.update({"color": gv.Colors[0], "marker": ml.Metrics[lmet]["marker"], "linestyle": "dotted"})
				label_logerr = "%s(%s)" % (settings["ylabel"], dbses[d].eccs[0].name)
				# Right axes plot -- with logical error rates.
				ax.plot(settings["xaxis"], settings["yaxis"], color=gv.Colors[d], alpha = 0.75, marker=settings["marker"], markersize=gv.marker_size, linestyle=settings["linestyle"], linewidth=1.5 * gv.line_width)
				if (ylimits["right"]["min"] >= np.min(settings["yaxis"])):
					ylimits["right"]["min"] = np.min(settings["yaxis"])
				if (ylimits["right"]["max"] <= np.max(settings["yaxis"])):
					ylimits["right"]["max"] = np.max(settings["yaxis"])
				
				# Let axes plots -- with uncorr.
				# Left y-axis for uncorr
				uncorr = np.load(PhysicalErrorRates(dbses[d], "uncorr"))[convereged_channels, l]
				label_uncorr = "%s(%s)" % (ml.Metrics["uncorr"]["latex"], dbses[d].eccs[0].name)
				ax.plot(settings["xaxis"], uncorr, color=gv.Colors[d], marker=ml.Metrics["uncorr"]["marker"], markersize=gv.marker_size, linestyle="solid", linewidth=1.5 * gv.line_width)
				if (ylimits["left"]["min"] >= np.min(uncorr)):
					ylimits["left"]["min"] = np.min(uncorr)
				if (ylimits["left"]["max"] <= np.max(uncorr)):
					ylimits["left"]["max"] = np.max(uncorr)
				
				# Empty plots to add legend entries.
				ax.plot([], [], color=gv.Colors[d], alpha = 0.75, marker=settings["marker"], markersize=gv.marker_size, linestyle=settings["linestyle"], linewidth=1.5 * gv.line_width, label = label_logerr)
				ax.plot([], [], color=gv.Colors[d], alpha = 0.75, marker=ml.Metrics["uncorr"]["marker"], markersize=gv.marker_size, linestyle="solid", linewidth=1.5 * gv.line_width, label = label_uncorr)
				
				# Empty plot for the legend entry containing different codes.
				# ax.plot([], [], color=settings["color"], linestyle="solid", linewidth=gv.line_width, label = label)

				# print("level {} and database {}".format(l, dbses[d].timestamp))
				# print("X\n{}\nY left\n{}\nY right\n{}".format(settings["xaxis"], settings["yaxis"], uncorr[:, l]))

			# Empty plots for the legend entries containing different colors/markers.
			# ax.plot([], [], "k", alpha=0.5, marker=ml.Metrics[lmet]["marker"], markersize=gv.marker_size, label = settings["ylabel"], linestyle="dotted", linewidth=gv.line_width)
			# ax.plot([], [], "k", marker=ml.Metrics["uncorr"]["marker"], markersize=gv.marker_size, label = ml.Metrics["uncorr"]["latex"], linestyle="solid", linewidth=gv.line_width)

			# Axes labels for the left (uncorr) plot
			ax.set_xlabel(settings["xlabel"], fontsize=gv.axes_labels_fontsize*1.4)
			# ax.set_ylabel(ml.Metrics["uncorr"]["latex"], fontsize=gv.axes_labels_fontsize*1.4)
			ax.set_xscale("log")
			# ax.set_xlim([None, 30])
			ax.set_yscale("log")
			ax.tick_params(axis="both", which="both", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize*1.4)

			# Grid lines
			ax.grid(color="0.5", which="both")

			# Axes labels for the right (logical error rates) plot
			# ax_right.set_ylabel(settings["ylabel"], fontsize=gv.axes_labels_fontsize*1.4)
			# ax_right.set_yscale("log")
			# ax_right.set_xlabel(settings["xlabel"], fontsize=gv.axes_labels_fontsize*1.4)
			# ax_right.set_xscale("log")
			# ax_right.tick_params(axis="both", which="both", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize*1.4)

			# Axes ticks for the uncorr plot
			# print("ylimits\n{}".format(ylimits))
			yticks_left = np.arange(OrderOfMagnitude(max(MIN, ylimits["left"]["min"]/5)), OrderOfMagnitude(ylimits["left"]["max"] * 5))
			ax.set_yticks(np.power(10.0, yticks_left), minor=True)
			# Axes ticks for the logical error rates plot
			# yticks_right = np.arange(OrderOfMagnitude(max(MIN, ylimits["right"]["min"]/5)), OrderOfMagnitude(ylimits["right"]["max"] * 5))
			# ax_right.set_yticks(np.power(10.0, yticks_right), minor=True)
			# print("Y ticks\nLeft\n{}\nRight\n{}".format(yticks_left, yticks_right))

			# legends for both plots
			# leg_left = ax.legend(loc="upper right", shadow=True, fontsize=1.4 * gv.legend_fontsize, markerscale=gv.legend_marker_scale)
			leg = ax.legend(loc="lower left", shadow=True, fontsize=1.4 * gv.legend_fontsize, markerscale=gv.legend_marker_scale)

			# Match legend text with the color of the markers
			for (t, text) in enumerate(leg.get_texts()):
			    if (t < 2):
			    	text.set_color(gv.Colors[0])
			    else:
			    	text.set_color(gv.Colors[1])
			
			# Save the plot
			fig.tight_layout(pad=5)
			pdf.savefig(fig)
			plt.close()

		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Comparison of %s for databases %s up to %d levels." % (ml.Metrics[lmet]["log"], "_".join([dbses[i].timestamp for i in range(ndb)]), nlevels)
		pdfInfo["Author"] = "Pavithran Iyer"
		pdfInfo["ModDate"] = dt.datetime.today()

	return None

# Ising
# def CompareSubs(pmet, lmet, *dbses):
#     # Compare the Logical error rates from two submissions.
#     # The comparision only makes sense when the logical error rates are measured for two submissions that have the same physical channels.
#     MIN = 1E-30
#     ndb = len(dbses)
#     nlevels = min([dbs.levels for dbs in dbses])
#
#     plotfname = CompareSubsPlot(dbses[0], [dbs.timestamp for dbs in dbses[1:]])
#     with PdfPages(plotfname) as pdf:
#         ylimits = {"left": {"min": 1, "max": 0}, "right": {"min": 1, "max": 0}}
#         for l in range(1, nlevels + 1):
#             fig = plt.figure(figsize=(gv.canvas_size[0]*1.3,gv.canvas_size[1]*1.25))
#             ax = plt.gca()
#             ax_right = ax.twinx()
#             for d in range(ndb):
#                 # Plot multiple logical error rates, with respect to the same physical error rates.
#                 # We use linestyles to distinguish between codes, and colors/markers to distinguish between y-axis metrics.
#                 settings = {"xaxis": None, "xlabel": None, "yaxis": np.load(LogicalErrorRates(dbses[d], lmet))[: , l], "ylabel": "$\\overline{%s_{%d}}$" % (ml.Metrics[lmet]["latex"].replace("$", ""), l)}
#                 LoadPhysicalErrorRates(dbses[0], pmet, settings, l)
#                 settings["xaxis"] = settings["xaxis"]/0.07
#                 settings.update({"color": gv.Colors[d % gv.n_Colors], "marker": ml.Metrics[lmet]["marker"], "linestyle": "dotted"})
#                 # label = ",".join(code.name for code in dbses[d].eccs)
#                 label = dbses[d].eccs[0].name
#                 # Plot logical error rates
#                 ax_right.plot(settings["xaxis"], settings["yaxis"], color=settings["color"], alpha = 0.75, marker=settings["marker"], markersize=gv.marker_size, linestyle=settings["linestyle"], linewidth=1.5 * gv.line_width)
#                 if (ylimits["right"]["min"] >= np.min(settings["yaxis"])):
#                     ylimits["right"]["min"] = np.min(settings["yaxis"])
#                 if (ylimits["right"]["max"] <= np.max(settings["yaxis"])):
#                     ylimits["right"]["max"] = np.max(settings["yaxis"])
#
# 				# Plot uncorr
#                 # Left y-axis for uncorr
#                 uncorr = np.load(PhysicalErrorRates(dbses[d], "uncorr"))
#                 ax.plot(settings["xaxis"], uncorr[:, l], color=gv.Colors[d % gv.n_Colors], marker=ml.Metrics["uncorr"]["marker"], markersize=gv.marker_size, linestyle="solid", linewidth=1.5 * gv.line_width)
#                 if (ylimits["left"]["min"] >= np.min(uncorr[:, l])):
#                     ylimits["left"]["min"] = np.min(uncorr[:, l])
#                 if (ylimits["left"]["max"] <= np.max(uncorr[:, l])):
#                     ylimits["left"]["max"] = np.max(uncorr[:, l])
#                 # Empty plot for the legend entry containing different codes.
#                 ax.plot([], [], color=settings["color"], linestyle="solid", linewidth=gv.line_width, label = label)
#
#                 # print("level {} and database {}".format(l, dbses[d].timestamp))
#                 # print("X\n{}\nY left\n{}\nY right\n{}".format(settings["xaxis"], settings["yaxis"], uncorr[:, l]))
#
#             # Empty plots for the legend entries containing different colors/markers.
#             ax.plot([], [], "k", alpha=0.5, marker=ml.Metrics[lmet]["marker"], markersize=gv.marker_size, label = settings["ylabel"], linestyle="dotted", linewidth=gv.line_width)
#             ax.plot([], [], "k", marker=ml.Metrics["uncorr"]["marker"], markersize=gv.marker_size, label = ml.Metrics["uncorr"]["latex"], linestyle="solid", linewidth=gv.line_width)
#
#             # Axes labels
#             tick_list = np.concatenate((np.linspace(np.min(settings["xaxis"]),1,5)[:-1],np.linspace(1,np.ceil(np.max(settings["xaxis"])),7,dtype=np.int)))
#
# 			# Axes label for uncorr plot
#             ax.set_xlabel(settings["xlabel"], fontsize=gv.axes_labels_fontsize*1.4, labelpad = 30)
#             ax.set_ylabel(ml.Metrics["uncorr"]["latex"], fontsize=gv.axes_labels_fontsize*1.4, labelpad = 30)
#             ax.set_xscale("log")
#             ax.set_yscale("log")
#             ax.tick_params(axis="y", which="both", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize*1.4)
#             ax.tick_params(axis="x", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize*1.4)
#             ax.set_xticks(tick_list)
#             ax.set_xticklabels(list(map(lambda x: "%.2g"%(x) ,tick_list)))
#
# 			# Axes labels for logical error rate plot
#             ax_right.set_ylabel(settings["ylabel"], fontsize=gv.axes_labels_fontsize*1.4, labelpad = 30)
#             ax_right.set_yscale("log")
#             ax_right.set_xlabel(settings["xlabel"], fontsize=gv.axes_labels_fontsize*1.4, labelpad = 30)
#             ax_right.set_xscale("log")
#             ax_right.tick_params(axis="x", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize*1.4)
#             ax_right.tick_params(axis="y", which="both", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize*1.4)
#             ax_right.set_xticks(tick_list)
#             ax_right.set_xticklabels(list(map(lambda x: "%.2g"%(x) ,tick_list)))
#
#             # Axes ticks
#             # print("ylimits\n{}".format(ylimits))
#             yticks_left = np.arange(OrderOfMagnitude(max(MIN, ylimits["left"]["min"]/5)), OrderOfMagnitude(ylimits["left"]["max"] * 5))
#             ax.set_yticks(np.power(10.0, yticks_left), minor=True)
#             yticks_right = np.arange(OrderOfMagnitude(max(MIN, ylimits["right"]["min"]/5)), OrderOfMagnitude(ylimits["right"]["max"] * 5))
#             ax_right.set_yticks(np.power(10.0, yticks_right), minor=True)
#             # print("Y ticks\nLeft\n{}\nRight\n{}".format(yticks_left, yticks_right))
#
#             # legend
#             ax.legend(loc="lower right", shadow=True, fontsize=1.4 * gv.legend_fontsize, markerscale=gv.legend_marker_scale)
#             # Save the plot
#             pdf.savefig(fig)
#             plt.close()
#
#         # Set PDF attributes
#         pdfInfo = pdf.infodict()
#         pdfInfo["Title"] = "Comparison of %s for databases %s up to %d levels." % (ml.Metrics[lmet]["log"], "_".join([dbses[i].timestamp for i in range(ndb)]), nlevels)
#         pdfInfo["Author"] = "Pavithran Iyer"
#         pdfInfo["ModDate"] = dt.datetime.today()
#
#     return None
