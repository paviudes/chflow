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
from mpl_toolkits.axes_grid.inset_locator import inset_axes, InsetPosition, mark_inset
from matplotlib.ticker import LogLocator

from define import fnames as fn
from define import globalvars as gv
from define import metrics as ml
from define import qchans as qc
from analyze.utils import GetNKDString
from analyze.load import LoadPhysicalErrorRates
from analyze.bins import PlotBinVarianceDataSets, PlotBinVarianceMetrics, GetXCutOff


def LevelWisePlot(phylist, logmet, dsets, inset, flow, nbins, thresholds):
    # Plot logical error rates vs. physical error rates.
    # Use a new figure for every new concatenated level.
    # In each figure, each curve will represent a new physical metric.
    nphy = len(phylist)
    ndb = len(dsets)
    maxlevel = max([dsets[i].levels for i in range(ndb)])
    plotfname = fn.LevelWise(dsets[0], "_".join(phylist), logmet)
    with PdfPages(plotfname) as pdf:
        for l in range(1, 1 + maxlevel):
            phlines = []
            phynames = []
            dblines = []
            dbnames = []
            fig = plt.figure(figsize=gv.canvas_size)
            ax1 = plt.gca()
            include = {}
            xcutoff = {}
            for p in range(nphy):
                for d in range(ndb):
                    settings = {
                        "xaxis": None,
                        "xlabel": None,
                        "yaxis": np.load(fn.LogicalErrorRates(dsets[d], logmet))[:, l],
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
                    LoadPhysicalErrorRates(dsets[d], phylist[p], settings, l)

                    if d == 0:
                        # print("Getting X cutoff for l = {}".format(l))
                        xcutoff = GetXCutOff(
                            settings["xaxis"],
                            settings["yaxis"],
                            thresholds[l - 1],
                            nbins=50,
                            space="log",
                        )
                        include[phylist[p]] = np.nonzero(
                            np.logical_and(
                                settings["xaxis"] >= xcutoff["left"],
                                settings["xaxis"] <= xcutoff["right"],
                            )
                        )[0]
                    # Plotting
                    plotobj = ax1.plot(
                        settings["xaxis"][include[phylist[p]]],
                        settings["yaxis"][include[phylist[p]]],
                        color=settings["color"],
                        alpha=0.75,
                        marker=settings["marker"],
                        markersize=gv.marker_size,
                        linestyle=settings["linestyle"],
                        linewidth=gv.line_width,
                    )
                    # Forcing axes ticks
                    # loc = LogLocator(base=10, numticks=10) # this locator puts ticks at regular intervals
                    # ax1.yaxis.set_major_locator(loc)
                    # ax1.set_xlim([xcutoff / offset, 1])
                    # ax1.set_ylim([ycutoff, 1])
                    # if we find a new physical metric, we must add it to metric legend labels
                    if not (settings["xlabel"] in phynames):
                        phlines.append(plotobj[0])
                        phynames.append(settings["xlabel"])
                    if not (dsets[d].timestamp in [name[0] for name in dbnames]):
                        dblines.append(plotobj[0])
                        if l == 0:
                            dbnames.append(
                                [dsets[d].timestamp, GetNKDString(dsets[d], l)]
                            )
                        if "name" in dsets[d].plotsettings:
                            dbnames.append(
                                [dsets[d].timestamp, dsets[d].plotsettings["name"]]
                            )
                        else:
                            dbnames.append(
                                [dsets[d].timestamp, GetNKDString(dsets[d], l)]
                            )
                if ndb > 1 and all([ds.samps > 1 for ds in dsets]):
                    PlotBinVarianceDataSets(
                        ax1, dsets, l, logmet, phylist, nbins, include
                    )
            # Add flow lines
            if flow == 1:
                AddFlowLines(ax1, dsets, phylist[0], logmet, l, include, nselect=10)
            # Inset plot
            # If there is more than one database, we will assume that there is only one physical metric.
            # Else we will assume many physical metrics can be compared.
            if l > 0:
                if (ndb == 1 or any([ds.samps == 1 for ds in dsets])) and (inset == 1):
                    # print(
                    #     "X\n{}\nY\n{}".format(
                    #         settings["xaxis"][include], settings["yaxis"][include]
                    #     )
                    # )
                    PlotBinVarianceMetrics(
                        ax1, dsets[0], l, logmet, phylist, nbins, include
                    )
                    # print("Problem in the scatter metric computation.")

            # Title
            # ax1.title(
            #     (
            #         "%s vs. physical error metrics for the %s channel."
            #         % (ml.Metrics[logmet]["log"], qc.Channels[dsets[0].channel]["name"])
            #     ),
            #     fontsize=gv.title_fontsize,
            #     y=1.03,
            # )
            # Axes labels
            if len(phylist) > 1:
                ax1.set_xlabel(
                    "Physical error metrics",
                    fontsize=gv.axes_labels_fontsize * 0.8,
                    labelpad=gv.axes_labelpad,
                )
            else:
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
            # if l == 1:
            #     ax1.set_ylim([3 * 10e-3, None])
            # if l == 2:
            #     ax1.set_ylim([10e-6, None])
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
            # Legend
            if ndb == 1:
                ax1.legend(
                    handles=phlines,
                    labels=phynames,
                    numpoints=1,
                    loc="lower right",
                    shadow=True,
                    fontsize=gv.legend_fontsize,
                    markerscale=gv.legend_marker_scale,
                )
            else:
                dblegend = ax1.legend(
                    handles=dblines,
                    labels=[name[1] for name in dbnames],
                    numpoints=1,
                    loc="lower right",  # center_left
                    shadow=True,
                    fontsize=gv.legend_fontsize,
                    markerscale=gv.legend_marker_scale,
                )
                ax1.add_artist(dblegend)
            # Save the plot
            pdf.savefig(fig)
            plt.close()
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = "%s at levels %s, with physical %s for %d channels." % (
            ml.Metrics[logmet]["log"],
            ", ".join(map(str, range(1, 1 + maxlevel))),
            ", ".join(phynames),
            dsets[0].channels,
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None


def AddFlowLines(ax1, dsets, pmet, lmet, level, include, nselect=10):
    """
    Add flow lines between the channel's non RC and RC performance.
    """
    selected = np.random.choice(include[pmet], nselect)
    scat1 = {}
    LoadPhysicalErrorRates(dsets[0], pmet, scat1, level)
    scat1["yaxis"] = np.load(fn.LogicalErrorRates(dsets[0], lmet))[selected, level]
    scat1["xaxis"] = scat1["xaxis"][selected]
    scat2 = {}
    LoadPhysicalErrorRates(dsets[1], pmet, scat2, level)
    scat2["xaxis"] = scat2["xaxis"][selected]
    scat2["yaxis"] = np.load(fn.LogicalErrorRates(dsets[1], lmet))[selected, level]
    ax1.plot(
        scat1["xaxis"],
        scat1["yaxis"],
        marker="o",
        color="darkblue",
        markersize=1.1 * gv.marker_size,
        linestyle="None",
    )
    ax1.plot(
        scat2["xaxis"],
        scat2["yaxis"],
        marker="o",
        color="fuchsia",
        markersize=1.1 * gv.marker_size,
        linestyle="None",
    )
    for i in range(nselect):
        ax1.plot(
            [scat1["xaxis"][i], scat2["xaxis"][i]],
            [scat1["yaxis"][i], scat2["yaxis"][i]],
            color="gold",
            linestyle="--",
            linewidth=gv.line_width,
        )
    return None


def LevelWisePlot2D(phymets, logmet, dbs):
    # Plot performance contours for the logical error rates for every concatenation level, with repect to the dephasing and relaxation rates.
    # Each plot will be a contour plot or a color density plot indicating the logical error, with the x-axis as the dephasing rate and the y-axis as the relaxation rate.
    # There will be one plot for every concatenation level.
    logErr = np.load(fn.LogicalErrorRates(dbs, logmet, fmt="npy"))
    phylist = list(map(lambda phy: phy.strip(" "), phymets.split(",")))
    phyerrs = np.zeros((dbs.channels, len(phylist)), dtype=np.longdouble)
    plotdata = np.zeros((max(100, dbs.channels), len(phylist)), dtype=np.longdouble)
    phyparams = []
    for m in range(len(phylist)):
        if sub.IsNumber(phylist[m]):
            # If phylist[m] is a number, then it indicates an independent parameter of the channel to serve as a measure of the physical noise strength
            phyerrs[:, m] = dbs.available[:, np.int8(phylist[m])]
            phyparams.append(qc.Channels[dbs.channel]["latex"][np.int8(phylist[m])])
            if not (dbs.scales[m] == 1):
                phyerrs[:, m] = np.power(dbs.scales[m], phyerrs[:, m])
        else:
            phyerrs[:, m] = np.load(fn.PhysicalErrorRates(dbs, phylist[m]))
            phyparams.append(ml.Metrics[phylist[m]]["latex"])
        plotdata[:, m] = np.linspace(
            phyerrs[:, m].min(), phyerrs[:, m].max(), plotdata.shape[0]
        )

    (meshX, meshY) = np.meshgrid(plotdata[:, 0], plotdata[:, 1])
    plotfname = fn.LevelWise(dbs, phymets.replace(",", "_"), logmet)
    with PdfPages(plotfname) as pdf:
        nqubits = 1
        dist = 1
        for l in range(1 + dbs.levels):
            if l == 0:
                nqubits = 1
                dist = 1
            else:
                nqubits = nqubits * dbs.eccs[l - 1].N
                dist = dist * dbs.eccs[l - 1].D
            fig = plt.figure(figsize=gv.canvas_size)
            meshZ = griddata(
                (phyerrs[:, 0], phyerrs[:, 1]),
                logErr[:, l],
                (meshX, meshY),
                method="cubic",
            )

            clevels = np.logspace(
                np.log10(np.abs(logErr[:, l].min())),
                np.log10(logErr[:, l].max()),
                gv.contour_nlevs,
                base=10.0,
            )
            cplot = plt.contourf(
                meshX,
                meshY,
                meshZ,
                cmap=cm.bwr,
                locator=ticker.LogLocator(),
                linestyles=gv.contour_linestyle,
                levels=clevels,
            )
            plt.scatter(phyerrs[:, 0], phyerrs[:, 1], marker="o", color="k")
            # Title
            plt.title(
                "N = %d, d = %d" % (nqubits, dist), fontsize=gv.title_fontsize, y=1.03
            )
            # Axes labels
            ax = plt.gca()
            ax.set_xlabel(phyparams[0], fontsize=gv.axes_labels_fontsize)
            ax.set_ylabel(phyparams[1], fontsize=gv.axes_labels_fontsize)

            if not (dbs.scales[0] == 1):
                ax.set_xscale("log", basex=dbs.scales[0], basey=dbs.scales[0])
                ax.invert_xaxis()
            if not (dbs.scales[1] == 1):
                ax.set_yscale("log", basex=dbs.scales[1], basey=dbs.scales[1])
                ax.invert_yaxis()

            ax.tick_params(
                axis="both",
                which="both",
                pad=gv.ticks_pad,
                direction="inout",
                length=gv.ticks_length,
                width=gv.ticks_width,
                labelsize=gv.ticks_fontsize,
            )
            # Legend
            cbar = plt.colorbar(
                cplot,
                extend="both",
                spacing="proportional",
                drawedges=False,
                ticks=clevels,
            )
            cbar.ax.set_xlabel(
                ml.Metrics[logmet]["latex"], fontsize=gv.colorbar_fontsize
            )
            cbar.ax.tick_params(
                labelsize=gv.legend_fontsize,
                pad=gv.ticks_pad,
                length=gv.ticks_length,
                width=gv.ticks_width,
            )
            cbar.ax.xaxis.labelpad = gv.ticks_pad
            cbar.ax.set_yticklabels(
                [
                    (
                        "$%.2f \\times 10^{%d}$"
                        % (
                            clevels[i]
                            * np.power(10, np.abs(np.int(np.log10(clevels[i])))),
                            np.int(np.log10(clevels[i])),
                        )
                    )
                    for i in range(len(clevels))
                ]
            )
            # Save the plot
            pdf.savefig(fig)
            plt.close()
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = "%s vs. %s at levels %s, for %d %s channels." % (
            ml.Metrics[logmet]["log"],
            ",".join(phylist),
            ", ".join(list(map(lambda str: "%d" % str, range(1, 1 + dbs.levels)))),
            dbs.channels,
            qc.Channels[dbs.channel]["name"],
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None
