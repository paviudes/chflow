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


def CompareSubs(pmet, lmet, *dbses):
    # Compare the Logical error rates from two submissions.
    # The comparision only makes sense when the logical error rates are measured for two submissions that have the same physical channels.
    ndb = len(dbses)
    nlevels = min([dbs.levels for dbs in dbses])
    
    plotfname = CompareSubsPlot(dbses)
    with PdfPages(plotfname) as pdf:
        for l in range(1, nlevels + 1):
            fig = plt.figure(figsize=gv.canvas_size)
            ax = plt.gca()
            ax_right = ax.twinx()
            for d in range(ndb):
                # Plot multiple logical error rates, with respect to the same physical error rates.
                settings = {"xaxis": None, "xlabel": None, "yaxis": np.load(LogicalErrorRates(dbses[d], lmet))[: , l], "ylabel": "$\\overline{%s_{%d}}$" % (ml.Metrics[lmet]["latex"].replace("$", ""), l)}
                LoadPhysicalErrorRates(dbses[0], pmet, settings, l)
                settings.update({"color": gv.Colors[d % gv.n_Colors], "marker": gv.Markers[d % gv.n_Markers], "linestyle": gv.line_styles[d % gv.n_line_styles]})
                ax.plot(settings["xaxis"], settings["yaxis"], color=settings["color"], marker=settings["marker"], markersize=gv.marker_size, linestyle=settings["linestyle"], label = dbses[d].eccs[0].name)
                
                # Right y-axis for uncorr
                uncorr = np.load(PhysicalErrorRates(dbses[d], "uncorr"))
                ax_right.plot(settings["xaxis"], uncorr[:, l], color=settings["color"], marker=ml.Metrics["uncorr"]["marker"], markersize=gv.marker_size, linestyle=settings["linestyle"], label = dbses[d].eccs[0].name)
                
                print("level {} and database {}".format(l, dbses[d].timestamp))
                print("X\n{}\nY left\n{}\nY right\n{}".format(settings["xaxis"], settings["yaxis"], uncorr[:, l]))

            # Axes labels
            ax.set_xlabel(settings["xlabel"], fontsize=gv.axes_labels_fontsize)
            ax.set_ylabel(settings["ylabel"], fontsize=gv.axes_labels_fontsize)
            ax_right.set_ylabel(ml.Metrics["uncorr"]["latex"], fontsize=gv.axes_labels_fontsize)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax_right.set_yscale("log")
            ax.tick_params(axis="both", which="both", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize)
            ax_right.tick_params(axis="both", which="both", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize)
            
            # legend
            ax.legend(numpoints=1, loc="best", shadow=True, fontsize=gv.legend_fontsize, markerscale=gv.legend_marker_scale)

            # Save the plot
            pdf.savefig(fig)
            plt.close()

        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = "Comparison of %s for databases %s up to %d levels." % ( settings["ylabel"], "_".join([dbses[i].timestamp for i in range(ndb)]), nlevels)
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    
    return None