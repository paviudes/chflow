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
            for p in range(2):
                fig = plt.figure(figsize=(gv.canvas_size[0]*1.1,gv.canvas_size[1]*1.1))
                ax = plt.gca()
                # ax_right = ax.twinx()
                ax_right = plt.gca()
                for d in range(ndb):
                    # Plot multiple logical error rates, with respect to the same physical error rates.
                    # We use linestyles to distinguish between codes, and colors/markers to distinguish between y-axis metrics.
                    settings = {"xaxis": None, "xlabel": None, "yaxis": np.load(LogicalErrorRates(dbses[d], lmet))[: , l], "ylabel": "$\\overline{%s_{%d}}$" % (ml.Metrics[lmet]["latex"].replace("$", ""), l)}
                    LoadPhysicalErrorRates(dbses[0], pmet, settings, l)
                    settings.update({"color": gv.Colors[d % gv.n_Colors], "marker": ml.Metrics[lmet]["marker"], "linestyle": "dotted"})
                    # label = ",".join(code.name for code in dbses[d].eccs)
                    label = dbses[d].eccs[0].name
                    if (p == 1):
                        ax_right.plot(settings["xaxis"], settings["yaxis"], color=settings["color"], alpha = 0.75, marker=settings["marker"], markersize=gv.marker_size, linestyle=settings["linestyle"], linewidth=1.5 * gv.line_width)
                        if (ylimits["right"]["min"] >= np.min(settings["yaxis"])):
                            ylimits["right"]["min"] = np.min(settings["yaxis"])
                        if (ylimits["right"]["max"] <= np.max(settings["yaxis"])):
                            ylimits["right"]["max"] = np.max(settings["yaxis"])
                        # Empty plot for the legend entry containing different codes.
                        ax_right.plot([], [], color=settings["color"], linestyle="dotted", linewidth=gv.line_width, label = label)

                    if (p == 0):
                        # Left y-axis for uncorr
                        uncorr = np.load(PhysicalErrorRates(dbses[d], "uncorr"))
                        ax.plot(settings["xaxis"], uncorr[:, l], color=gv.Colors[d % gv.n_Colors], marker=ml.Metrics["uncorr"]["marker"], markersize=gv.marker_size, linestyle="solid", linewidth=1.5 * gv.line_width)
                        if (ylimits["left"]["min"] >= np.min(uncorr[:, l])):
                            ylimits["left"]["min"] = np.min(uncorr[:, l])
                        if (ylimits["left"]["max"] <= np.max(uncorr[:, l])):
                            ylimits["left"]["max"] = np.max(uncorr[:, l])
                        # Empty plot for the legend entry containing different codes.
                        ax.plot([], [], color=settings["color"], linestyle="solid", linewidth=gv.line_width, label = label)

                        # print("level {} and database {}".format(l, dbses[d].timestamp))
                        # print("X\n{}\nY left\n{}\nY right\n{}".format(settings["xaxis"], settings["yaxis"], uncorr[:, l]))

                # Empty plots for the legend entries containing different colors/markers.
                if (p == 1):
                    ax.plot([], [], "k", alpha=0.5, marker=ml.Metrics[lmet]["marker"], markersize=gv.marker_size, label = settings["ylabel"], linestyle="dotted", linewidth=gv.line_width)
                if (p == 0):
                    ax.plot([], [], "k", marker=ml.Metrics["uncorr"]["marker"], markersize=gv.marker_size, label = ml.Metrics["uncorr"]["latex"], linestyle="solid", linewidth=gv.line_width)

                # Axes labels
                if (p == 0):
                    ax.set_xlabel(settings["xlabel"], fontsize=gv.axes_labels_fontsize*1.4)
                    ax.set_ylabel(ml.Metrics["uncorr"]["latex"], fontsize=gv.axes_labels_fontsize*1.4)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.tick_params(axis="both", which="both", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize*1.4)

                if (p == 1):
                    ax_right.set_ylabel(settings["ylabel"], fontsize=gv.axes_labels_fontsize*1.4)
                    ax_right.set_yscale("log")
                    ax_right.set_xlabel(settings["xlabel"], fontsize=gv.axes_labels_fontsize*1.4)
                    ax_right.set_xscale("log")
                    ax_right.tick_params(axis="both", which="both", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize*1.4)

                # Axes ticks
                # print("ylimits\n{}".format(ylimits))
                if (p == 0):
                    yticks_left = np.arange(OrderOfMagnitude(max(MIN, ylimits["left"]["min"]/5)), OrderOfMagnitude(ylimits["left"]["max"] * 5))
                    ax.set_yticks(np.power(10.0, yticks_left), minor=True)
                if (p == 1):
                    yticks_right = np.arange(OrderOfMagnitude(max(MIN, ylimits["right"]["min"]/5)), OrderOfMagnitude(ylimits["right"]["max"] * 5))
                    ax_right.set_yticks(np.power(10.0, yticks_right), minor=True)
                # print("Y ticks\nLeft\n{}\nRight\n{}".format(yticks_left, yticks_right))

                # legend
                if (p == 0):
                    ax.legend(loc="best", shadow=True, fontsize=1.4 * gv.legend_fontsize, markerscale=gv.legend_marker_scale)
                if (p == 1):
                    ax_right.legend(loc="best", shadow=True, fontsize=1.4 * gv.legend_fontsize, markerscale=gv.legend_marker_scale)
                # Save the plot
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
#             for p in range(2):
#                 fig = plt.figure(figsize=(gv.canvas_size[0]*1.25,gv.canvas_size[1]*1.25))
#                 ax = plt.gca()
#                 # ax_right = ax.twinx()
#                 ax_right = plt.gca()
#                 for d in range(ndb):
#                     # Plot multiple logical error rates, with respect to the same physical error rates.
#                     # We use linestyles to distinguish between codes, and colors/markers to distinguish between y-axis metrics.
#                     settings = {"xaxis": None, "xlabel": None, "yaxis": np.load(LogicalErrorRates(dbses[d], lmet))[: , l], "ylabel": "$\\overline{%s_{%d}}$" % (ml.Metrics[lmet]["latex"].replace("$", ""), l)}
#                     LoadPhysicalErrorRates(dbses[0], pmet, settings, l)
#                     settings["xaxis"] = settings["xaxis"]/0.07
#                     settings.update({"color": gv.Colors[d % gv.n_Colors], "marker": ml.Metrics[lmet]["marker"], "linestyle": "dotted"})
#                     # label = ",".join(code.name for code in dbses[d].eccs)
#                     label = dbses[d].eccs[0].name
#                     if (p == 1):
#                         ax_right.plot(settings["xaxis"], settings["yaxis"], color=settings["color"], alpha = 0.75, marker=settings["marker"], markersize=gv.marker_size, linestyle=settings["linestyle"], linewidth=1.5 * gv.line_width)
#                         if (ylimits["right"]["min"] >= np.min(settings["yaxis"])):
#                             ylimits["right"]["min"] = np.min(settings["yaxis"])
#                         if (ylimits["right"]["max"] <= np.max(settings["yaxis"])):
#                             ylimits["right"]["max"] = np.max(settings["yaxis"])
#                         # Empty plot for the legend entry containing different codes.
#                         ax_right.plot([], [], color=settings["color"], linestyle="dotted", linewidth=gv.line_width, label = label)
#
#                     if (p == 0):
#                         # Left y-axis for uncorr
#                         uncorr = np.load(PhysicalErrorRates(dbses[d], "uncorr"))
#                         ax.plot(settings["xaxis"], uncorr[:, l], color=gv.Colors[d % gv.n_Colors], marker=ml.Metrics["uncorr"]["marker"], markersize=gv.marker_size, linestyle="solid", linewidth=1.5 * gv.line_width)
#                         if (ylimits["left"]["min"] >= np.min(uncorr[:, l])):
#                             ylimits["left"]["min"] = np.min(uncorr[:, l])
#                         if (ylimits["left"]["max"] <= np.max(uncorr[:, l])):
#                             ylimits["left"]["max"] = np.max(uncorr[:, l])
#                         # Empty plot for the legend entry containing different codes.
#                         ax.plot([], [], color=settings["color"], linestyle="solid", linewidth=gv.line_width, label = label)
#
#                         # print("level {} and database {}".format(l, dbses[d].timestamp))
#                         # print("X\n{}\nY left\n{}\nY right\n{}".format(settings["xaxis"], settings["yaxis"], uncorr[:, l]))
#
#                 # Empty plots for the legend entries containing different colors/markers.
#                 if (p == 1):
#                     ax.plot([], [], "k", alpha=0.5, marker=ml.Metrics[lmet]["marker"], markersize=gv.marker_size, label = settings["ylabel"], linestyle="dotted", linewidth=gv.line_width)
#                 if (p == 0):
#                     ax.plot([], [], "k", marker=ml.Metrics["uncorr"]["marker"], markersize=gv.marker_size, label = ml.Metrics["uncorr"]["latex"], linestyle="solid", linewidth=gv.line_width)
#
#                 # Axes labels
#                 tick_list = np.concatenate((np.linspace(np.min(settings["xaxis"]),1,5)[:-1],np.linspace(1,np.ceil(np.max(settings["xaxis"])),7,dtype=np.int)))
#
#
#                 if (p == 0):
#                     ax.set_xlabel(settings["xlabel"], fontsize=gv.axes_labels_fontsize*1.4, labelpad = 30)
#                     ax.set_ylabel(ml.Metrics["uncorr"]["latex"], fontsize=gv.axes_labels_fontsize*1.4, labelpad = 30)
#                     ax.set_xscale("log")
#                     ax.set_yscale("log")
#                     ax.tick_params(axis="y", which="both", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize*1.4)
#                     ax.tick_params(axis="x", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize*1.4)
#                     ax.set_xticks(tick_list)
#                     ax.set_xticklabels(list(map(lambda x: "%.2g"%(x) ,tick_list)))
#
#                 if (p == 1):
#                     ax_right.set_ylabel(settings["ylabel"], fontsize=gv.axes_labels_fontsize*1.4, labelpad = 30)
#                     ax_right.set_yscale("log")
#                     ax_right.set_xlabel(settings["xlabel"], fontsize=gv.axes_labels_fontsize*1.4, labelpad = 30)
#                     ax_right.set_xscale("log")
#                     ax_right.tick_params(axis="x", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize*1.4)
#                     ax_right.tick_params(axis="y", which="both", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize*1.4)
#                     ax_right.set_xticks(tick_list)
#                     ax_right.set_xticklabels(list(map(lambda x: "%.2g"%(x) ,tick_list)))
#
#                 # Axes ticks
#                 # print("ylimits\n{}".format(ylimits))
#                 if (p == 0):
#                     yticks_left = np.arange(OrderOfMagnitude(max(MIN, ylimits["left"]["min"]/5)), OrderOfMagnitude(ylimits["left"]["max"] * 5))
#                     ax.set_yticks(np.power(10.0, yticks_left), minor=True)
#                 if (p == 1):
#                     yticks_right = np.arange(OrderOfMagnitude(max(MIN, ylimits["right"]["min"]/5)), OrderOfMagnitude(ylimits["right"]["max"] * 5))
#                     ax_right.set_yticks(np.power(10.0, yticks_right), minor=True)
#                 # print("Y ticks\nLeft\n{}\nRight\n{}".format(yticks_left, yticks_right))
#
#                 # legend
#                 if (p == 0):
#                     ax.legend(loc="lower right", shadow=True, fontsize=1.4 * gv.legend_fontsize, markerscale=gv.legend_marker_scale)
#                 if (p == 1):
#                     ax_right.legend(loc="lower right", shadow=True, fontsize=1.4 * gv.legend_fontsize, markerscale=gv.legend_marker_scale)
#                 # Save the plot
#                 pdf.savefig(fig)
#                 plt.close()
#
#         # Set PDF attributes
#         pdfInfo = pdf.infodict()
#         pdfInfo["Title"] = "Comparison of %s for databases %s up to %d levels." % (ml.Metrics[lmet]["log"], "_".join([dbses[i].timestamp for i in range(ndb)]), nlevels)
#         pdfInfo["Author"] = "Pavithran Iyer"
#         pdfInfo["ModDate"] = dt.datetime.today()
#
#     return None
