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

    plotfname = CompareSubsPlot(dbses)
    with PdfPages(plotfname) as pdf:
        ylimits = {"left": {"min": 1, "max": 0}, "right": {"min": 1, "max": 0}}
        for l in range(1, nlevels + 1):
            fig = plt.figure(figsize=gv.canvas_size)
            ax = plt.gca()
            ax_right = ax.twinx()
            for d in range(ndb):
                # Plot multiple logical error rates, with respect to the same physical error rates.
                # We use linestyles to distinguish between codes, and colors/markers to distinguish between y-axis metrics.
                settings = {"xaxis": None, "xlabel": None, "yaxis": np.load(LogicalErrorRates(dbses[d], lmet))[: , l], "ylabel": "$\\overline{%s_{%d}}$" % (ml.Metrics[lmet]["latex"].replace("$", ""), l)}
                LoadPhysicalErrorRates(dbses[0], pmet, settings, l)
                settings.update({"color": "grey", "marker": ml.Metrics[lmet]["marker"], "linestyle": gv.line_styles[d % gv.n_line_styles]})
                ax_right.plot(settings["xaxis"], settings["yaxis"], color=settings["color"], alpha = 0.5, marker=settings["marker"], markersize=gv.marker_size, linestyle=settings["linestyle"], linewidth=gv.line_width)
                if (ylimits["right"]["min"] >= np.min(settings["yaxis"])):
                    ylimits["right"]["min"] = np.min(settings["yaxis"])
                if (ylimits["right"]["max"] <= np.max(settings["yaxis"])):
                    ylimits["right"]["max"] = np.max(settings["yaxis"])
                # Empty plot for the legend entry containing different linestyles.
                label = ",".join(code.name[:3] for code in dbses[d].eccs)
                ax.plot([], [], color="k", linestyle=settings["linestyle"], linewidth=gv.line_width, label = label)

                # Left y-axis for uncorr
                uncorr = np.load(PhysicalErrorRates(dbses[d], "uncorr"))
                ax.plot(settings["xaxis"], uncorr[:, l], color=ml.Metrics["uncorr"]["color"], marker=ml.Metrics["uncorr"]["marker"], markersize=gv.marker_size, linestyle=settings["linestyle"], linewidth=gv.line_width)
                if (ylimits["left"]["min"] >= np.min(uncorr[:, l])):
                    ylimits["left"]["min"] = np.min(uncorr[:, l])
                if (ylimits["left"]["max"] <= np.max(uncorr[:, l])):
                    ylimits["left"]["max"] = np.max(uncorr[:, l])

                print("level {} and database {}".format(l, dbses[d].timestamp))
                print("X\n{}\nY left\n{}\nY right\n{}".format(settings["xaxis"], settings["yaxis"], uncorr[:, l]))

            # Empty plots for the legend entries containing different colors/markers.
            ax.plot([], [], color="grey", alpha=0.5, marker=ml.Metrics[lmet]["marker"], markersize=gv.marker_size, label = settings["ylabel"], linestyle="None")
            ax.plot([], [], color=ml.Metrics["uncorr"]["color"], marker=ml.Metrics["uncorr"]["marker"], markersize=gv.marker_size, label = ml.Metrics["uncorr"]["latex"], linestyle="None")

            # Axes labels
            ax.set_xlabel(settings["xlabel"], fontsize=gv.axes_labels_fontsize)
            ax_right.set_ylabel(settings["ylabel"], fontsize=gv.axes_labels_fontsize)
            ax.set_ylabel(ml.Metrics["uncorr"]["latex"], fontsize=gv.axes_labels_fontsize)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax_right.set_yscale("log")
            ax.tick_params(axis="both", which="both", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize)
            ax_right.tick_params(axis="both", which="both", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize)

            # Axes ticks
            print("ylimits\n{}".format(ylimits))
            yticks_left = np.arange(OrderOfMagnitude(max(MIN, ylimits["left"]["min"]/5)), OrderOfMagnitude(ylimits["left"]["max"] * 5))
            ax.set_yticks(np.power(10.0, yticks_left), minor=True)
            yticks_right = np.arange(OrderOfMagnitude(max(MIN, ylimits["right"]["min"]/5)), OrderOfMagnitude(ylimits["right"]["max"] * 5))
            ax_right.set_yticks(np.power(10.0, yticks_right), minor=True)
            print("Y ticks\nLeft\n{}\nRight\n{}".format(yticks_left, yticks_right))

            # legend
            ax.legend(loc="best", shadow=True, fontsize=gv.legend_fontsize, markerscale=gv.legend_marker_scale)

            # Save the plot
            pdf.savefig(fig)
            plt.close()

        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = "Comparison of %s for databases %s up to %d levels." % (ml.Metrics[lmet]["log"], "_".join([dbses[i].timestamp for i in range(ndb)]), nlevels)
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()

    return None
