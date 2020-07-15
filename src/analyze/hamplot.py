import os
import sys
import datetime as dt

try:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colors, ticker, cm
    from matplotlib.colors import LogNorm
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid.inset_locator import (
        inset_axes,
        InsetPosition,
        mark_inset,
    )
    from matplotlib.ticker import LogLocator
    from scipy.interpolate import griddata
    import PyPDF2 as pp
except:
    pass
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
                else:
                    include[phylist[c]] = include[phylist[0]]

                LoadPhysicalErrorRates(dsets[c], phylist[c], settings[c], l)
                # Plotting
                ax1.plot(
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

            # Axes labels
            ax1.set_xlabel("Physical error metrics", fontsize=gv.axes_labels_fontsize * 0.8, labelpad = gv.axes_labelpad)
            ax1.set_xscale("log")
            ax1.set_ylabel(settings[0]["ylabel"], fontsize=gv.axes_labels_fontsize, labelpad = gv.axes_labelpad)
            # if l == 2:
            #     ax1.set_ylim([10e-9, None])
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
            loc = LogLocator(base=10, numticks=10) # this locator puts ticks at regular intervals
            ax1.xaxis.set_major_locator(loc)
            # Legend
            ax1.legend(
                numpoints=1,
                loc="lower center",  # center_left
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
