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
    from scipy.interpolate import griddata
except:
    pass
# Force the module scripts to run locally -- https://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path
# import inspect as ins
# current = os.path.realpath(os.path.abspath(os.path.dirname(ins.getfile(ins.currentframe()))))
# if (not (current in sys.path)):
# 	sys.path.insert(0, current)

from define import qchans as qc
from define import fnames as fn
from define import globalvars as gv
from define import metrics as ml
from define import submission as sub
from analyze import collect as cl


def latex_float(f):
    # Function taken from: https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def DisplayForm(number, base):
    # Express a float number in a scientific notation with two digits after the decimal.
    # N = A b^x where A < b. Then, log N/(log b) = log A/(log b) + x.
    # So, x = floor(log N/(log b)) and log A/(log b) = log N/(log b) - x.
    # then, A = b^(log N/(log b) - x)
    # print("number = %g, base = %g" % (number, base))
    numstr = "0"
    sign = ""
    if number == 0:
        return numstr
    if number < 0:
        sign = "-"
        number = np.abs(number)
    exponent = np.floor(np.log(number) / np.log(base))
    factor = np.int(
        np.power(base, np.log(number) / np.log(base) - exponent) * 100
    ) / np.float(100)
    numstr = "$%s%g \\times %g^{%d}$" % (sign, factor, base, exponent)
    # if (number < 0):
    # 	numstr = ("-%s" % (numstr))
    return numstr


def ThresholdPlot(phymets, logmet, dbs):
    # For each physical noise rate, plot the logical error rate vs. the levels of concatenation.
    # The set of curves should have a bifurcation at the threshold.

    # print("dbs.available")
    # print dbs.available

    phylist = list(map(lambda phy: phy.strip(" "), phymets.split(",")))
    sampreps = np.hstack((np.nonzero(dbs.available[:, -1] == 0)[0], [dbs.channels]))
    # print("sampreps\n%s" % (np.array_str(sampreps)))
    logErrs = np.load(fn.LogicalErrorRates(dbs, logmet, fmt="npy"))
    logErr = np.zeros((sampreps.shape[0], dbs.levels + 1), dtype=np.longdouble)
    for i in range(sampreps.shape[0] - 1):
        for l in range(dbs.levels + 1):
            logErr[i, l] = np.sum(
                logErrs[sampreps[i] : sampreps[i + 1], l], dtype=np.longdouble
            ) / np.longdouble(sampreps[i + 1] - sampreps[i])
    # print("sampreps\n%s" % (np.array_str(sampreps)))
    phyerrs = np.zeros((sampreps.shape[0] - 1, len(phylist)), dtype=np.longdouble)
    phyparams = []
    for m in range(len(phylist)):
        if sub.IsNumber(phylist[m]):
            # If phylist[m] is a number, then it indicates an independent parameter of the channel to serve as a measure of the physical noise strength
            for i in range(sampreps.shape[0] - 1):
                phyerrs[i, m] = np.sum(
                    dbs.available[sampreps[i] : sampreps[i + 1], np.int8(phylist[m])],
                    dtype=np.longdouble,
                ) / np.longdouble(sampreps[i + 1] - sampreps[i])
            phyparams.append(qc.Channels[dbs.channel][2][np.int8(phylist[m])])
        else:
            # print("loading: %s" % (fn.PhysicalErrorRates(dbs, phylist[m])))
            phyrates = np.load(fn.PhysicalErrorRates(dbs, phylist[m]))
            # print("metric = %s, phyrates\n%s" % (phylist[m], np.array_str(phyrates)))
            for i in range(sampreps.shape[0] - 1):
                # print("phyrates[%d:%d, np.int8(phylist[m])]\n%s" % (sampreps[i], sampreps[i + 1], np.array_str(phyrates[sampreps[i]:sampreps[i + 1]])))
                phyerrs[i, m] = np.sum(
                    phyrates[sampreps[i] : sampreps[i + 1]], dtype=np.longdouble
                ) / np.longdouble(sampreps[i + 1] - sampreps[i])
            phyparams.append(ml.Metrics[phylist[m]][1])
    # print("phyerrs")
    # print phyerrs
    plotfname = fn.ThreshPlot(dbs, "_".join(phylist), logmet)
    with PdfPages(plotfname) as pdf:
        for m in range(len(phylist)):
            fig = plt.figure(figsize=gv.canvas_size)
            for p in range(phyerrs.shape[0]):
                fmtidx = list(ml.Metrics.keys())[p % len(ml.Metrics)]
                plt.plot(
                    np.arange(dbs.levels + 1),
                    logErr[p, :],
                    label=("%s = %s" % (phyparams[m], DisplayForm(phyerrs[p, m], 10))),
                    color=ml.Metrics[fmtidx][3],
                    marker=ml.Metrics[fmtidx][2],
                    markersize=gv.marker_size,
                    linestyle="--",
                    linewidth=gv.line_width,
                )
            # Legend
            plt.legend(
                numpoints=1,
                loc=3,
                shadow=True,
                fontsize=gv.legend_fontsize,
                markerscale=gv.legend_marker_scale,
            )
            # Title
            plt.title(
                "Threshold of %s channel in %s."
                % (qc.Channels[dbs.channel][0], phyparams[m]),
                fontsize=gv.title_fontsize,
                y=1.03,
            )
            # Axes labels
            ax = plt.gca()
            xticks = np.cumprod([dbs.eccs[i].D for i in range(dbs.levels)])
            ax.set_xticks(range(dbs.levels + 1))
            ax.set_xlabel("$\\ell$", fontsize=gv.axes_labels_fontsize)
            ax.set_ylabel(
                "$\\widetilde{\\mathcal{N}}_{\\ell}: %s$"
                % (ml.Metrics[logmet][1].replace("$", "")),
                fontsize=gv.axes_labels_fontsize,
            )
            # ax.set_yscale('log')
            ax.tick_params(
                axis="both",
                which="major",
                pad=gv.ticks_pad,
                direction="inout",
                length=gv.ticks_length,
                width=gv.ticks_width,
                labelsize=gv.ticks_fontsize,
            )
            # Save the plot
            pdf.savefig(fig)
            plt.close()
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = "%s at levels %s, with physical %s for %d channels." % (
            ml.Metrics[logmet][0],
            ", ".join(map(str, range(1, 1 + dbs.levels))),
            ", ".join(phyparams),
            dbs.channels,
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None


def ChannelWisePlot(phymet, logmet, dbses):
    # Plot each channel in the database with a different color.
    # Channels of similar type in different databases will be distinguished using different markers.
    plotfname = fn.ChannelWise(dbses[0], phymet, logmet)
    colors = ["b", "g", "r", "k", "c", "y"]
    markers = ["*", "^", "s"]
    annotations = [
        "$\\mathcal{E}_{1}$",
        "$\\mathcal{E}_{2}$",
        "$\\mathcal{E}_{3}$",
        "$\\mathcal{E}_{3}$",
        "$\\mathcal{E}_{4}$",
        "$\\mathcal{E}_{5}$",
    ]
    maxlevel = max([db.levels for db in dbses])
    with PdfPages(plotfname) as pdf:
        for l in range(1 + maxlevel):
            fig = plt.figure(figsize=gv.canvas_size)
            ax = plt.gca()
            # plt.axvline(x=0.06, linestyle="--")
            for d in range(len(dbses)):
                marker = markers[d % len(markers)]
                plt.plot(
                    [],
                    [],
                    marker=marker,
                    color="k",
                    label=dbses[d].plotsettings["name"],
                    markersize=gv.marker_size,
                )
                logerrs = np.load(fn.LogicalErrorRates(dbses[d], logmet))[:, l]
                phyerrs = np.load(fn.PhysicalErrorRates(dbses[d], phymet))
                unitarity = np.load(fn.PhysicalErrorRates(dbses[0], "np1"))
                for i in range(dbses[d].channels):
                    if not (i == 3):
                        plt.plot(
                            phyerrs[i],
                            logerrs[i],
                            color=colors[i % len(colors)],
                            marker=marker,
                            markersize=2 * gv.marker_size,
                        )
                        # if d == 0:
                        #     # Text annotations at data points
                        #     print("{}). nu = {}".format(annotations[i], unitarity[i]))
                        #     ax.annotate(
                        #         "%.2f, %s" % (unitarity[i], annotations[i]),
                        #         (1.03 * phyerrs[i], logerrs[i]),
                        #         color=colors[i % len(colors)],
                        #         fontsize=gv.ticks_fontsize,
                        #     )
                        # Text annotations at data points
                        ax.annotate(
                            annotations[i],
                            (1.03 * phyerrs[i], logerrs[i]),
                            color=colors[i % len(colors)],
                            fontsize=gv.ticks_fontsize,
                        )
            # Axes labels
            if "xlabel" in dbses[d].plotsettings:
                ax.set_xlabel(
                    "%s" % (dbses[d].plotsettings["xlabel"]),
                    fontsize=gv.axes_labels_fontsize,
                )
            else:
                ax.set_xlabel(
                    "$\\mathcal{N}_{0}$: Physical noise strength",
                    fontsize=gv.axes_labels_fontsize,
                )
            # ax.set_xscale("log")
            if "ylabel" in dbses[d].plotsettings:
                ax.set_ylabel(
                    "%s" % (dbses[d].plotsettings["ylabel"]),
                    fontsize=gv.axes_labels_fontsize,
                )
            else:
                ax.set_ylabel(
                    "$\\mathcal{N}_{%d}$: %s" % (l, ml.Metrics[logmet][1]),
                    fontsize=gv.axes_labels_fontsize,
                )
            # ax.set_yscale("log")
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
            plt.legend(
                numpoints=1,
                loc=2,
                shadow=True,
                fontsize=gv.legend_fontsize,
                markerscale=gv.legend_marker_scale,
            )
            # Save the plot
            pdf.savefig(fig)
            plt.close()
    return None


def LevelWisePlot(phymets, logmet, dbses):
    # Plot logical error rates vs. physical error rates.
    # Use a new figure for every new concatenated level.
    # In each figure, each curve will represent a new physical metric.
    phylist = list(map(lambda phy: phy.strip(" "), phymets.split(",")))
    ndb = len(dbses)
    maxlevel = max([dbses[i].levels for i in range(ndb)])
    plotfname = fn.LevelWise(dbses[0], "_".join(phylist), logmet)
    with PdfPages(plotfname) as pdf:
        for l in range(1 + maxlevel):
            phlines = []
            phynames = []
            dblines = []
            dbnames = []
            fig = plt.figure(figsize=gv.canvas_size)
            for p in range(len(phylist)):
                for d in range(ndb):
                    logErrs = np.load(
                        fn.LogicalErrorRates(dbses[d], logmet, fmt="npy")
                    )[:, l]
                    if phylist[p] in ml.Metrics:
                        phymet = ml.Metrics[phylist[p]][1]
                        phyerrs = np.load(fn.PhysicalErrorRates(dbses[d], phylist[p]))
                        plotset = [
                            ml.Metrics[phylist[p]][3],
                            ml.Metrics[phylist[p]][2],
                            ["None", "--"][dbses[d].samps == 1],
                        ]
                    else:
                        phymet = qc.Channels[dbses[d].channel][2][np.int8(phylist[p])]
                        phyerrs = dbses[d].available[:, np.int8(phylist[p])]
                        plotset = [
                            ml.Metrics[list(ml.Metrics.keys())[p % len(ml.Metrics)]][3],
                            ml.Metrics[list(ml.Metrics.keys())[p % len(ml.Metrics)]][2],
                            ["None", "--"][dbses[d].samps == 1],
                        ]
                        if not (dbses[d].scales[p] == 1):
                            phyerrs = np.power(dbses[d].scales[p], phyerrs)
                    # Plotting
                    # Plot the X=Y line denoting the quantum error correction trade-off
                    # plt.plot(phyerrs, phyerrs, linestyle="--", color="k")
                    if d == 0:
                        if "color" in dbses[d].plotsettings:
                            color = dbses[d].plotsettings["color"]
                        else:
                            color = plotset[0]
                        if "marker" in dbses[d].plotsettings:
                            marker = dbses[d].plotsettings["marker"]
                        else:
                            marker = plotset[1]
                        if "linestyle" in dbses[d].plotsettings:
                            linestyle = dbses[d].plotsettings["linestyle"]
                        else:
                            linestyle = plotset[2]
                        plotobj = plt.plot(
                            phyerrs,
                            logErrs,
                            color=color,
                            marker=marker,
                            markersize=gv.marker_size,
                            linestyle=linestyle,
                            linewidth=gv.line_width,
                        )
                    else:
                        plotobj = plt.plot(
                            phyerrs,
                            logErrs,
                            color=dbses[d].plotsettings["color"],
                            marker=dbses[d].plotsettings["marker"],
                            markersize=gv.marker_size,
                            linestyle=dbses[d].plotsettings["linestyle"],
                            linewidth=gv.line_width,
                        )
                    # if we find a new physical metric, we must add it to metric legend labels
                    if not (phymet in phynames):
                        phlines.append(plotobj[0])
                        phynames.append(phymet)
                    if not (dbses[d].timestamp in [name[0] for name in dbnames]):
                        dblines.append(plotobj[0])
                        if l == 0:
                            dbnames.append(
                                [
                                    dbses[d].timestamp,
                                    (
                                        ("Physical channel: %s")
                                        % (qc.Channels[dbses[d].channel][0])
                                    ),
                                ]
                            )
                        if "name" in dbses[d].plotsettings:
                            dbnames.append(
                                [dbses[d].timestamp, dbses[d].plotsettings["name"]]
                            )
                        else:
                            dbnames.append(
                                [
                                    dbses[d].timestamp,
                                    (
                                        ("N = %d, D = %d, %s")
                                        % (
                                            np.prod(
                                                [dbses[d].eccs[j].N for j in range(l)]
                                            ),
                                            np.prod(
                                                [dbses[d].eccs[j].D for j in range(l)]
                                            ),
                                            qc.Channels[dbses[d].channel][0],
                                        )
                                    ),
                                ]
                            )

            # Title
            # plt.title(("%s vs. physical error metrics for the %s channel." % (ml.Metrics[logmet][0], qc.Channels[dbses[0].channel][0])), fontsize = gv.title_fontsize, y = 1.03)
            # Axes labels
            ax = plt.gca()
            if "xlabel" in dbses[d].plotsettings:
                ax.set_xlabel(
                    "%s" % (dbses[d].plotsettings["xlabel"]),
                    fontsize=gv.axes_labels_fontsize,
                )
            else:
                ax.set_xlabel(
                    "$\\mathcal{N}_{0}$: Physical noise strength",
                    fontsize=gv.axes_labels_fontsize,
                )
            ax.set_xscale("log")
            if "ylabel" in dbses[d].plotsettings:
                ax.set_ylabel(
                    "%s" % (dbses[d].plotsettings["ylabel"]),
                    fontsize=gv.axes_labels_fontsize,
                )
            else:
                ax.set_ylabel(
                    "$\\mathcal{N}_{%d}$: %s" % (l, ml.Metrics[logmet][1]),
                    fontsize=gv.axes_labels_fontsize,
                )
            ax.set_yscale("log")
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
            dblegend = plt.legend(
                handles=dblines,
                labels=[name[1] for name in dbnames],
                numpoints=1,
                loc=2,
                shadow=True,
                fontsize=gv.legend_fontsize,
                markerscale=gv.legend_marker_scale,
            )
            plt.legend(
                handles=phlines,
                labels=phynames,
                numpoints=1,
                loc=4,
                shadow=True,
                fontsize=gv.legend_fontsize,
                markerscale=gv.legend_marker_scale,
            )
            ax.add_artist(dblegend)
            # Save the plot
            pdf.savefig(fig)
            plt.close()
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = "%s at levels %s, with physical %s for %d channels." % (
            ml.Metrics[logmet][0],
            ", ".join(map(str, range(1, 1 + maxlevel))),
            ", ".join(phynames),
            dbses[0].channels,
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
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
            phyparams.append(qc.Channels[dbs.channel][2][np.int8(phylist[m])])
            if not (dbs.scales[m] == 1):
                phyerrs[:, m] = np.power(dbs.scales[m], phyerrs[:, m])
        else:
            phyerrs[:, m] = np.load(fn.PhysicalErrorRates(dbs, phylist[m]))
            phyparams.append(ml.Metrics[phylist[m]][1])
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
            cbar.ax.set_xlabel(ml.Metrics[logmet][1], fontsize=gv.colorbar_fontsize)
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
            ml.Metrics[logmet][0],
            ",".join(phylist),
            ", ".join(list(map(lambda str: "%d" % str, range(1, 1 + dbs.levels)))),
            dbs.channels,
            qc.Channels[dbs.channel][0],
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None


def CompareSubs(logmet, *dbses):
    # Compare the Logical error rates from two submissions.
    # The comparision only makes sense when the logical error rates are measured for two submissions that have the same physical channels.
    ndb = len(dbses)
    cdepth = min([dbses[i].levels for i in range(ndb)])
    logErrs = np.zeros((ndb, dbs1.channels, cdepth), dtype=np.longdouble)
    for i in range(ndb):
        logErrs[i, :, :] = np.load(fn.LogicalErrorRates(dbses[i], logmet))
    plotfname = fn.CompareLogErrRates(dbses, logmet)
    with PdfPages(plotfname) as pdf:
        for l in range(cdepth + 1):
            for i in range(ndb):
                for j in range(i + 1, ndb):
                    fig = plt.figure(figsize=gv.canvas_size)
                    plt.plot(
                        logErrs[i, :, l + 1],
                        logErrs[j, :, l + 1],
                        color=ml.Metrics[logmet][3],
                        marker=ml.Metrics[logmet][2],
                        markersize=gv.marker_size,
                        linestyle="None",
                    )
                    plt.title(
                        "Level %d %s for %s vs. %s."
                        % (
                            l,
                            ml.Metrics[logmet][1],
                            dbses[i].timestamp,
                            dbses[j].timestamp,
                        )
                    )
                    # Axes labels
                    ax = plt.gca()
                    ax.set_xlabel(dbses[i].timestamp, fontsize=gv.axes_labels_fontsize)
                    ax.set_xscale("log")
                    ax.set_xlabel(dbses[j].timestamp, fontsize=gv.axes_labels_fontsize)
                    ax.set_yscale("log")
                    ax.tick_params(
                        axis="both",
                        which="major",
                        pad=gv.ticks_pad,
                        direction="inout",
                        length=gv.ticks_length,
                        width=gv.ticks_width,
                        labelsize=gv.ticks_fontsize,
                    )
                    # Save the plot
                    pdf.savefig(fig)
                    plt.close()
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = "Comparison of %s for databases %s up to %d levels." % (
            ml.Metrics[logmet][0],
            "_".join([dbses[i].timestamp for i in range(ndb)]),
            cdepth,
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None


def CompareAnsatzToMetrics(dbs, pmet, lmet):
    # Compare the fluctuations in the logical error rates with respect to the physical noise metrics -- obtained from fit and those computed from standard metrics
    logerr = np.load(fn.LogicalErrorRates(dbs, lmet))
    phyerr = np.load(fn.PhysicalErrorRates(dbs, pmet))
    fiterr = np.load(fn.FitPhysRates(dbs, lmet))
    weightenums = np.load(fn.FitWtEnums(dbs, lmet))
    expo = np.load(fn.FitExpo(dbs, lmet))
    fitlines = np.zeros((2, 2), dtype=np.float)
    plotfname = fn.AnsatzComparePlot(dbs, lmet, pmet)
    with PdfPages(plotfname) as pdf:
        for l in range(1 + dbs.levels):
            fig = plt.figure(figsize=gv.canvas_size)
            # Plots
            plt.plot(
                phyerr,
                logerr[:, l],
                label=ml.Metrics[pmet][1],
                color=ml.Metrics[pmet][3],
                marker=ml.Metrics[pmet][2],
                markersize=gv.marker_size,
                linestyle="None",
            )
            fitlines[0, :] = cl.ComputeBestFitLine(
                np.column_stack((np.log10(phyerr), logerr[:, l]))
            )
            fitlines[1, :] = cl.ComputeBestFitLine(
                np.column_stack((np.log10(fiterr), logerr[:, l]))
            )
            # If the reference line is: ln y = m1 (ln x) + c1
            # and the machine learnt line is: ln y = m2 (ln x) + c2,
            # then we need to scale the machine learning data by: ln x' -> (ln x) * m2/m1 + (c2 - c1) which is the same as: x' = x^(m2/m1) * exp(c2 - c1)
            plt.plot(
                np.power(fiterr, fitlines[1, 0] / fitlines[0, 0]),
                logerr[:, l],
                label=(
                    "$\\epsilon$ where $\\widetilde{\\mathcal{N}}_{%d} = %s \\times \\left[\\epsilon(\\mathcal{E})\\right]^{%.2f t}$"
                    % (l, latex_float(weightenums[l]), expo)
                ),
                color="blue",
                marker="o",
                markersize=gv.marker_size,
                linestyle="None",
            )
            # Axes
            ax = plt.gca()
            ax.set_xlabel("$\\mathcal{N}_{0}$", fontsize=gv.axes_labels_fontsize)
            ax.set_xscale("log")
            ax.set_ylabel(
                (
                    "$\\mathcal{N}_{%d}$  $\\left(%s\\right)$"
                    % (l, ml.Metrics[lmet][1].replace("$", ""))
                ),
                fontsize=gv.axes_labels_fontsize,
            )
            ax.set_yscale("log")
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
            lgnd = plt.legend(
                numpoints=1,
                loc=4,
                shadow=True,
                fontsize=gv.legend_fontsize,
                markerscale=gv.legend_marker_scale,
            )
            # Save the plot
            pdf.savefig(fig)
            plt.close()
            # Add bar plots to compare the scatter
            bins = ComputeBinVariance(np.abs(phyerr), -np.log10(np.abs(logerr[:, l])))
            AddBinVariancePlot(bins, l, lmet, pmet, ml.Metrics[pmet][1], pdf)
            bins = ComputeBinVariance(
                np.power(np.abs(fiterr), fitlines[1, 0] / fitlines[0, 0]),
                -np.log10(np.abs(logerr[:, l])),
            )
            AddBinVariancePlot(bins, l, lmet, 0, "$\\epsilon(\\mathcal{E})$", pdf)
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = (
            "Comparing fit obtained p to physical %s at levels %s, by studying fluctuations of output %s for %d channels."
            % (
                ml.Metrics[pmet][0],
                ", ".join(list(map(lambda str: "%d" % str, range(1, 1 + dbs.levels)))),
                ml.Metrics[lmet][0],
                dbs.channels,
            )
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None


def ValidatePrediction(dbs, pmet, lmet):
    # Validate a prediction by comparing the fluctuations in the logical error rate with reprect to
    # (i) The predicted physical noise rate
    # (ii) Any standard metric for reference.
    # Compute the best fit line through both the datasets and scale the machine learnt data to overlap with that of the reference metric.
    logerr = np.load(fn.LogicalErrorRates(dbs, lmet))
    phyerr = np.load(fn.PhysicalErrorRates(dbs, pmet))
    macerr = np.load(fn.PredictedPhyRates(dbs))

    # print("macerr\n%s\nnon positive elements\n%s" % (np.array_str(macerr), np.array_str(macerr[np.where(macerr <= 0)])))

    fitlines = np.zeros((2, 2), dtype=np.float)
    plotfname = fn.PredictComparePlot(dbs, lmet, pmet)
    with PdfPages(plotfname) as pdf:
        for l in range(1 + dbs.levels):
            fig = plt.figure(figsize=gv.canvas_size)
            plt.plot(
                phyerr,
                logerr[:, l],
                label=ml.Metrics[pmet][1],
                color=ml.Metrics[pmet][3],
                marker=ml.Metrics[pmet][2],
                markersize=gv.marker_size,
                linestyle="None",
            )
            fitlines[0, :] = cl.ComputeBestFitLine(
                np.column_stack((np.log10(phyerr), logerr[:, l]))
            )
            fitlines[1, :] = cl.ComputeBestFitLine(
                np.column_stack((np.log10(macerr), logerr[:, l]))
            )
            # If the reference line is: ln y = m1 (ln x) + c1
            # and the machine learnt line is: ln y = m2 (ln x) + c2,
            # then we need to scale the machine learning data by: ln x' -> (ln x) * m2/m1 + (c2 - c1) which is the same as: x' = x^(m2/m1) * exp(c2 - c1)
            # print("fitlines\n%s\nscale = %g" % (fitlines, fitlines[1, 0]/fitlines[0, 0]))
            # print("macerr\n%s" % (np.array_str(macerr)))
            plt.plot(
                np.power(macerr, fitlines[1, 0] / fitlines[0, 0]),
                logerr[:, l],
                label="$\\epsilon_{\\rm predicted}$",
                color="blue",
                marker="o",
                markersize=gv.marker_size,
                linestyle="None",
            )
            # Axes labels
            ax = plt.gca()
            ax.set_xlabel("$\\mathcal{N}_{0}$", fontsize=gv.axes_labels_fontsize)
            ax.set_xscale("log")
            ax.set_ylabel(
                (
                    "$\\widetilde{\\mathcal{N}}_{%d}$  $\\left(%s\\right)$"
                    % (l, ml.Metrics[lmet][1].replace("$", ""))
                ),
                fontsize=gv.axes_labels_fontsize,
            )
            ax.set_yscale("log")
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
            lgnd = plt.legend(
                numpoints=1,
                loc=4,
                shadow=True,
                fontsize=gv.legend_fontsize,
                markerscale=gv.legend_marker_scale,
            )
            # Save the plot
            pdf.savefig(fig)
            plt.close()
            # Add bar plots to compare the scatter
            bins = ComputeBinVariance(np.abs(phyerr), -np.log10(np.abs(logerr[:, l])))
            AddBinVariancePlot(bins, l, lmet, pmet, ml.Metrics[pmet][1], pdf)
            bins = ComputeBinVariance(
                np.power(np.abs(macerr), fitlines[1, 0] / fitlines[0, 0]),
                -np.log10(np.abs(logerr[:, l])),
            )
            AddBinVariancePlot(bins, l, lmet, 0, "$\\epsilon(\\mathcal{E})$", pdf)
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = (
            "Comparing predicted p to physical %s at levels %s, by studying fluctuations of output %s for %d channels."
            % (
                ml.Metrics[pmet][0],
                ", ".join(list(map(lambda str: "%d" % str, range(1, 1 + dbs.levels)))),
                ml.Metrics[lmet][0],
                dbs.channels,
            )
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None


def MCStatsPlot(dbses, lmet, pmet=-1):
    # Plot the logical error rate vs. number of syndromes sampled at the top level.
    # The figure contains one plot for every concatenated level.
    # In every plot, there is one curve for every value of the noise parameters.
    # This plot only makes sense at the topmost level when each value of noise rate is simulated using many statistics.
    # The various samples for syndromes are specified in dbs.stats.
    # The respective running average of logical error rates is stored in results/.
    # For every batch of M numbers in column l:
    # Plot the running average of the metric with respect the the number of samples used to compute that average.
    # See https://faculty.washington.edu/stuve/log_error.pdf on how to compute error bars for log plot.
    # To estimate d(z = log y), we have: dz = d(log y) = 1/ln(10) * d(ln y) = 1/ln(10) * (dy)/y
    ndbses = len(dbses)
    logerrs = []
    phyerrs = []
    for d in range(ndbses):
        if sub.IsNumber(pmet) == 1:
            pmet = int(pmet)
            if pmet == -1:
                phyerrs.append(dbses[d].available[:, :-1])
            else:
                phyerrs.append(dbses[d].available[:, pmet, np.newaxis])
        else:
            phyerrs.append(np.load(fn.PhysicalErrorRates(dbses[d], pmet)))
        logerrs.append(np.load(fn.RunningAverages(dbses[d], lmet)))
        # print("d = %d: importance = %d\nphyerrs[d]\n%s\ndbs.noiserates\n%s\nlogerrs[d]\n%s" % (d, dbses[d].importance, np.array_str(phyerrs[d]), np.array_str(dbses[d].noiserates), np.array_str(logerrs[d][:, :, :, 0])))

    plotfname = fn.MCStatsPlot(dbses[0], lmet, pmet)
    with PdfPages(plotfname) as pdf:
        for s in range(
            1 + int(all([(dbses[d].importance == 2) for d in range(ndbses)]))
        ):
            labels = []
            fig = plt.figure(figsize=gv.canvas_size)
            # Empty plots to indicate the databse (possibily with varying sampling methods) in the legend
            for d in range(ndbses):
                plt.plot(
                    [],
                    [],
                    color="k",
                    linestyle=gv.line_styles[d % len(gv.line_styles)],
                    label=(
                        "%s sampling"
                        % (
                            ", ".join(
                                [
                                    name
                                    for (name, num) in dbses[d].samplingOptions.items()
                                    if (num == dbses[d].importance)
                                ]
                            )
                        )
                    ),
                )
            for d in range(ndbses):
                for i in range(dbses[d].noiserates.shape[0]):
                    if sub.IsNumber(pmet) == 1:
                        if pmet == -1:
                            label = "%s = %s" % (
                                ", ".join(qc.Channels[dbses[d].channel][2]),
                                ",".join(
                                    list(
                                        map(
                                            lambda num: DisplayForm(num, 10.0),
                                            RealNoise(dbses[d].scales, phyerrs[d][i]),
                                        )
                                    )
                                ),
                            )
                        else:
                            # print("dbses[d].scales = %s, phyerrs[d][i] = %d and real noise = %s" % (np.array_str(dbses[d].scales), phyerrs[d][i], ", ".join(list(map(lambda num: DisplayForm(num, 10.0), RealNoise(dbses[d].scales, phyerrs[d][i]))))))
                            label = "%s = %s" % (
                                qc.Channels[dbses[d].channel][2][pmet],
                                ", ".join(
                                    list(
                                        map(
                                            lambda num: DisplayForm(num, 10.0),
                                            RealNoise(dbses[d].scales, phyerrs[d][i]),
                                        )
                                    )
                                ),
                            )
                        marker = ml.Metrics[
                            list(ml.Metrics.keys())[i % len(list(ml.Metrics.keys()))]
                        ][2]
                        color = ml.Metrics[
                            list(ml.Metrics.keys())[i % len(list(ml.Metrics.keys()))]
                        ][3]
                    else:
                        label = "%s = %g" % (ml.Metrics[pmet][1], phyerrs[d][i])
                        marker = ml.Metrics[pmet][2]
                        color = ml.Metrics[pmet][3]
                    # Plot logerrs[d][i * nstats:(i * nstats + i), l] vs. phyerrs[d][i * nstats:(i * nstats + i), -1]
                    # print("i = %d, label = %s\nXaxis\n%s\nYAxis\n%s" % (i, label, dbses[d].stats, logerrs[d][i, :, s]))
                    if label in labels:
                        label = None
                    else:
                        labels.append(label)
                    delta = (
                        1 / np.log(10) * logerrs[d][i, 1, :, s] / logerrs[d][i, 0, :, s]
                    )
                    yerr = np.power(10, np.log10(logerrs[d][i, 0, :, s]) + delta)
                    # print("d = %d, delta\n%s\nyerr\n%s" % (d, np.array_str(delta), np.array_str(yerr)))
                    plt.plot(
                        dbses[d].stats,
                        logerrs[d][i, 0, :, s],
                        label=label,
                        linewidth=gv.line_width,
                        linestyle=gv.line_styles[d % len(gv.line_styles)],
                        color=ml.Metrics[
                            list(ml.Metrics.keys())[i % len(list(ml.Metrics.keys()))]
                        ][3],
                        marker=ml.Metrics[
                            list(ml.Metrics.keys())[i % len(list(ml.Metrics.keys()))]
                        ][2],
                        markersize=gv.marker_size,
                    )
            # Axes labels
            ax = plt.gca()
            ax.set_xlabel("$N$", fontsize=gv.axes_labels_fontsize)
            ax.set_xscale("log", nonposx="clip")
            ax.set_ylabel(
                (
                    "$\\widetilde{\\mathcal{N}}_{%d}$  $\\left(%s\\right)$"
                    % (dbses[0].levels, ml.Metrics[lmet][1].replace("$", ""))
                ),
                fontsize=gv.axes_labels_fontsize,
            )
            ax.set_yscale("log", nonposy="clip")
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
            lgnd = plt.legend(
                numpoints=1,
                loc=4,
                shadow=True,
                fontsize=gv.legend_fontsize,
                markerscale=gv.legend_marker_scale,
            )
            # Save the plot
            pdf.savefig(fig)
            plt.close()
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = (
            "Convergence of average logical %s with the number of syndrome samples for different %s at level %d."
            % (ml.Metrics[lmet][0], str(pmet), dbses[0].levels)
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None


def RealNoise(scales, ratesexp):
    # Output the set of noise rates given the noise rate specifiers
    rates = np.zeros(ratesexp.shape[0], dtype=np.longdouble)
    for i in range(ratesexp.shape[0]):
        if scales[i] == 1:
            rates[i] = ratesexp[i]
        else:
            rates[i] = np.power(scales[i], ratesexp[i])
    return rates


def BinsPlot(dbs, lmet, pvals):
    # Plot the number of syndromes at each level for a given probability and conditional logical error (infidelity, etc).
    # If no noise rate is specified, one plot for each channel in the database.
    # For each channel, the bins array is formatted as: bins[level, synd prob, metric val].
    npoints = 6
    nchans = 0
    plotfname = fn.SyndromeBinsPlot(dbs, lmet, pvals)
    with PdfPages(plotfname) as pdf:
        for i in range(dbs.channels):
            if pvals == -1:
                cont = 1
            else:
                cont = 0
                if np.all(dbs.available[i, :] == pvals):
                    cont = 1
            if cont == 1:
                # print("p = %s" % (np.array_str(dbs.available[i, :])))
                nchans = nchans + 1
                bins = np.load(
                    fn.SyndromeBins(
                        dbs, dbs.available[i, :-1], dbs.available[i, -1], lmet
                    )
                )
                for l in range(1, dbs.levels):
                    fig = plt.figure(
                        figsize=(gv.canvas_size[0] * 2, gv.canvas_size[1] * 3)
                    )
                    # print("l = %d\n%d non zero rows\n%s\n%d non zero columns\n%s" % (l + 1, np.count_nonzero(~np.all(bins[1 + l, :, :] == 0, axis=1)), np.array_str(np.nonzero(~np.all(bins[1 + l, :, :] == 0, axis=1))[0]), np.count_nonzero(~np.all(bins[1 + l, :, :] == 0, axis=0)), np.array_str(np.nonzero(~np.all(bins[1 + l, :, :] == 0, axis=0))[0])))
                    nzrows = np.nonzero(~np.all(bins[1 + l, :, :] == 0, axis=1))[0]
                    nzcols = np.nonzero(~np.all(bins[1 + l, :, :] == 0, axis=0))[0]
                    (meshX, meshY) = np.meshgrid(nzrows, nzcols, indexing="ij")
                    meshZ = bins[1 + l, nzrows, :][:, nzcols]
                    meshZ = meshZ / np.max(meshZ)
                    # print("meshX: (%d, %d) \n%s\nmeshY: (%d, %d)\n%s\nmeshZ: (%d, %d)\n%s" % (meshX.shape[0], meshX.shape[1], np.array_str(meshX), meshY.shape[0], meshY.shape[1], np.array_str(meshY), meshZ.shape[0], meshZ.shape[1], np.array_str(meshZ)))
                    plt.pcolor(
                        meshY,
                        meshX,
                        meshZ,
                        cmap="binary",
                        norm=LogNorm(
                            vmin=np.min(meshZ[np.nonzero(meshZ)]), vmax=np.max(meshZ)
                        ),
                    )
                    # Title
                    # plt.title("p = %s, s = %d, l = %d" % (np.array_str(dbs.available[i, :-1]), int(dbs.available[i, -1]), l + 1), fontsize = gv.title_fontsize, y = 1.03)
                    # Axes labels
                    ax = plt.gca()
                    ax.set_ylabel(
                        "$- $log${}_{10}\\mathsf{Pr}(s)$",
                        fontsize=gv.axes_labels_fontsize + 144,
                    )
                    ax.set_ylim([0, 50])
                    # yticks = nzrows[np.linspace(0, nzrows.shape[0] - 1, npoints, dtype = np.int)]
                    # ax.set_yticks(yticks)
                    # ax.set_yticklabels(["%d" % (tc) for tc in yticks])
                    ax.set_xlabel(
                        "$- $log${}_{10}\\mathcal{N}(\\mathcal{E}^{\\thickspace s}_{%d})$"
                        % (l + 1),
                        fontsize=gv.axes_labels_fontsize + 144,
                    )
                    # xticks = nzcols[np.linspace(0, nzcols.shape[0] - 1, npoints, dtype = np.int)]
                    # ax.set_xticks(xticks)
                    # ax.set_xticklabels(["%d" % (tc) for tc in xticks])
                    ax.tick_params(
                        axis="both",
                        which="both",
                        pad=gv.ticks_pad + 50,
                        direction="inout",
                        length=gv.ticks_length,
                        width=gv.ticks_width,
                        labelsize=gv.ticks_fontsize + 120,
                    )
                    # Legend
                    cbar = plt.colorbar(spacing="proportional", drawedges=False)
                    # cbar.ax.set_xlabel("$\\mathcal{T}_{%d}$" % (l + 1), fontsize = gv.colorbar_fontsize)
                    cbar.ax.tick_params(
                        labelsize=gv.legend_fontsize + 120,
                        pad=gv.ticks_pad,
                        length=gv.ticks_length,
                        width=gv.ticks_width,
                    )
                    cbar.ax.xaxis.labelpad = gv.ticks_pad
                    # Save the plot
                    pdf.savefig(fig)
                    plt.close()
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = "Syndrome bins for %d channels and %d levels." % (
            nchans,
            dbs.levels,
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None


def PlotBinVariance(dbses, lmet, pmet, nbins=10):
    # Plot the variance in each bin with respect to the bin along with producing a table of those values.
    # prepare the logical error data and use ComputeBinVariance() to compute the bins and AddBinVariancePlot() to plot.
    if sub.IsNumber(pmet) == 1:
        phyerr = np.squeeze(
            np.vstack(tuple([dbses[i].available[:, pmet] for i in range(len(dbses))]))
        )
        pmetname = qc.Channels[dbses[0].channel][2][pmet]
    else:
        phyerr = np.hstack(
            tuple(
                [
                    np.load(fn.PhysicalErrorRates(dbses[i], pmet))
                    for i in range(len(dbses))
                ]
            )
        )
        pmetname = ml.Metrics[pmet][1]
    logerr = np.vstack(
        tuple(
            [np.load(fn.LogicalErrorRates(dbses[i], lmet)) for i in range(len(dbses))]
        )
    )

    # print("logerr\n%s" % (np.array_str(logerr)))
    # np.savetxt("%s/results/logerr.txt" % (dbses[0].outdir), logerr)
    # print("phyerr\n%s" % (np.array_str(phyerr)))
    # np.savetxt("%s/results/phyerr.txt" % (dbses[0].outdir), phyerr)

    maxlevel = min([dbses[i].levels for i in range(len(dbses))])
    plotfname = fn.VarianceBins(dbses[0], lmet, pmet)
    with PdfPages(plotfname) as pdf:
        for l in range(maxlevel + 1):
            # print("logerr[%d]" % (l))
            # print logerr[:, l]
            if l == 0:
                bins = ComputeBinVariance(
                    np.abs(phyerr),
                    -np.log10(np.abs(logerr[:, l])),
                    nbins,
                    binfile=fn.BinSummary(dbses[0], pmet, lmet, l),
                    submit=dbses[0],
                )
            else:
                bins = ComputeBinVariance(
                    np.abs(phyerr), -np.log10(np.abs(logerr[:, l])), nbins
                )
            # bins = ComputeBinVariance(-np.log10(np.abs(phyerr)), np.abs(logerr[:, l]), nbins)
            # bins = ComputeBinVariance(np.abs(phyerr), np.abs(logerr[:, l]), nbins)
            # print "bins"
            # print bins
            AddBinVariancePlot(bins, l, lmet, pmet, pmetname, pdf)
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = "Binwise variance of %s with respect to %s." % (
            lmet,
            str(pmet),
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None


def ComputeBinVariance(xdata, ydata, nbins=10, space="log", binfile=None, submit=None):
    # Compute the amount of scater of data in a plot.
    # Divide the X axis range into bins and compute the variance of Y-data in each bin.
    # The bins must divide the axes on a linear scale -- because they signify the confidence interval in measuring the values of the parameters.
    # The variance of a dataset of values {x1, x2, ..., xn} is just \sum_i (xi - xmean)^2.
    # the Output is formatted as:
    # 	bins: 2D array with N rows and 4 columns
    # 			Each row represents a channel
    # 			bins[i] = [low, high, npoints, var]
    # 			where low and high are the physical error rates that specify the bin.
    # 			npoints is the number of physical error rates in the bin
    # 			var is the variance of logical error rates in the bin.
    base = 10
    bins = np.zeros((nbins - 1, 4), dtype=np.longdouble)
    if space == "log":
        window = np.power(
            base,
            np.linspace(
                np.log10(np.max(xdata)) / np.log10(base),
                np.log10(np.min(xdata)) / np.log10(base),
                nbins,
            ),
        )[::-1]
    else:
        window = np.linspace(np.min(xdata), np.max(xdata), nbins)
    bins[:, 0] = window[:-1]
    bins[:, 1] = window[1:]

    if binfile is not None:
        bf = open(binfile, "w")
        bf.write(
            "# index from to npoints var min chan nrate samp max chan nrate samp\n"
        )
    for i in range(nbins - 1):
        points = np.nonzero(np.logical_and(xdata >= bins[i, 0], xdata < bins[i, 1]))[0]
        bins[i, 2] = np.longdouble(points.shape[0])
        # Variance of the Y axis points in the bin
        bins[i, 3] = np.sqrt(np.var(ydata[points]))
        # bins[i, 3] = np.abs(np.max(ydata[points]) - np.min(ydata[points]))
        # mean = np.power(10, np.mean(np.log10(ydata[points])))
        # bins[i, 3] = np.sqrt(np.sum(np.power(ydata[points] - mean, 2)))/((bins[i, 2] - 1) * mean)
        # bins[i, 3] = np.sqrt(np.var(ydata[points])/np.power(np.mean(ydata[points]), 2))
        # print("bin %d: %d points\n\t[%g, %g] -- max = %g and min = %g, U = %g, D = %g." % (i, bins[i, 2], bins[i, 0], bins[i, 1], np.max(ydata[points]), np.min(ydata[points]), np.mean(ydata[points]), bins[i, 3]))
        if binfile is not None:
            minchan = points[np.argmin(ydata[points])]
            maxchan = points[np.argmax(ydata[points])]
            # print(
            #     "points = {}, minchan = {}, maxchan = {}".format(
            #         points, minchan, maxchan
            #     )
            # )
            bf.write(
                "%d %g %g %d %g %g %d %s %d %g %d %s %d\n"
                % (
                    i,
                    bins[i, 0],
                    bins[i, 1],
                    bins[i, 2],
                    bins[i, 3],
                    np.min(ydata[points]),
                    minchan,
                    " ".join(
                        list(
                            map(lambda num: "%g" % num, submit.available[minchan, :-1])
                        )
                    ),
                    submit.available[minchan, -1],
                    np.max(ydata[points]),
                    maxchan,
                    " ".join(
                        list(
                            map(lambda num: "%g" % num, submit.available[maxchan, :-1])
                        )
                    ),
                    submit.available[maxchan, -1],
                )
            )
    print(
        "\033[2mTotal: %d points and average variance = %g and maximum variance = %g.\033[0m"
        % (np.sum(bins[:, 2], dtype=int), np.mean(bins[:, 3]), np.max(bins[:, 3]))
    )
    if binfile is not None:
        bf.close()
    return bins


def ComputeNDimBinVariance(xdata, ydata, nbins=3, space="linear"):
    # Divide a N-dimensional space into bins and classify xdata points into bins.
    base = 10.0
    ndim = xdata.shape[1]
    # Divide each axis of the d-dimensional space into intervals
    window = np.zeros((ndim, nbins), dtype=np.longdouble)
    for i in range(ndim):
        if space == "log":
            window[i, :] = np.power(
                base,
                np.linspace(
                    np.log10(np.max(xdata[:, i])) / np.log10(base),
                    np.log10(np.min(xdata[:, i])) / np.log10(base),
                    nbins,
                ),
            )[::-1]
        else:
            window[i, :] = np.linspace(np.min(xdata[:, i]), np.max(xdata[:, i]), nbins)

    # print("xdata\n%s" % (np.array_str(xdata)))
    # print("window\n%s" % (np.array_str(window)))

    # For every point in xdata, determine its address in terms of windows, in the n-dim space.
    address = np.zeros((xdata.shape[0], ndim), dtype=np.int)
    binindex = np.zeros(xdata.shape[0], dtype=np.int)
    for i in range(xdata.shape[0]):
        for j in range(ndim):
            # which window does xdata[i, j] fall into ?
            # print("xdata[i, j] = %g, window[j, :] = %s" % (xdata[i, j], np.array_str(window[j, :])))
            # print "np.logical_and(xdata[i, j] >= window[j, :-1], xdata[i, j] < window[j, 1:])"
            # print np.logical_and(xdata[i, j] >= window[j, :-1], xdata[i, j] < window[j, 1:])
            # address[i, j] = np.nonzero(np.logical_and(xdata[i, j] >= window[j, :-1], xdata[i, j] < window[j, 1:]).astype(np.int))[0][0]
            for k in range(nbins):
                if xdata[i, j] > window[j, k]:
                    address[i, j] = k

        # Interprett the address as an encoding of the bin index in base-b alphabet, where b is the number of bins in any axis.
        binindex[i] = np.sum(
            np.multiply(
                address[i, :], np.power(nbins - 1, np.linspace(ndim - 1, 0, ndim))
            ),
            dtype=np.int,
        )

        # print("address[i, :]\n%s\nnp.power(nbins - 1, np.linspace(ndim - 1, 0, ndim))\n%s" % (np.array_str(address[i, :]), np.array_str(np.power(nbins - 1, np.linspace(ndim - 1, 0, ndim)))))

    # print("address\n%s" % (np.array_str(address)))
    # print("binindex\n%s" % (np.array_str(binindex)))

    # Count the number of xdata points with a fixed bin index and record that information in bins.
    bins = np.zeros((np.power(nbins - 1, ndim, dtype=np.int), 4), dtype=np.longdouble)
    isempty = np.zeros(bins.shape[0], dtype=np.int)
    for i in range(bins.shape[0]):
        if np.count_nonzero(binindex == i) == 0:
            isempty[i] = 1
        else:
            points = np.nonzero(binindex == i)[0]
            bins[i, 2] = points.shape[0]
            bins[i, 3] = np.var(ydata[points])
            # print("Bin %d: points = %s\nydata = %s\nvariance = %g and mean = %g." % (i, np.array_str(points), np.array_str(ydata[points]), bins[i, 3], np.mean(ydata[points])))

    # print("Bins\n%s" % (np.array_str(bins[:, 2:])))

    print(
        "\033[2mTotal: %d points and average variance = %g and maximum variance = %g.\033[0m"
        % (np.sum(bins[:, 2], dtype=int), np.mean(bins[:, 3]), np.max(bins[:, 3]))
    )
    return bins


def AddBinVariancePlot(
    bins, level, lmet, pmet, pmetname, pdf=None, plotfname="unknown"
):
    # Plot the variance in each bin with respect to the bin along with producing a table of those values.
    # If a PdfPages object is specified, the plot is simply added to the PDF.
    # Else, it is a separate plot. In this case, the name of the file to which the plot must be stored, must be specified.
    if sub.IsNumber(pmet) == 1:
        color = ml.Metrics[
            list(ml.Metrics.keys())[len(list(ml.Metrics.keys())) % (1 + pmet)]
        ][3]
    else:
        color = ml.Metrics[pmet][3]
    if pdf is None:
        pdfobj = PdfPages(plotfname)
    fig = plt.figure(figsize=((gv.canvas_size[0] * 1.5, gv.canvas_size[1] * 1.5)))
    plt.title(
        "level %d: Average scatter = %g, Maximum scatter = %g"
        % (level, np.mean(bins[:, 3]), np.max(bins[:, 3])),
        fontsize=gv.title_fontsize,
        y=1.03,
    )
    # print("widths\n%s" % (np.array_str(bins[:, 1] - bins[:, 0])))
    barplot = plt.bar(
        bins[:, 0],
        bins[:, 3],
        width=bins[:, 1] - bins[:, 0],
        bottom=0,
        align="edge",
        color=color,
        linewidth=0,
    )
    # Axes
    ax = plt.gca()
    ax.set_xlabel(
        "$-\\log10(\\mathcal{N}_{0}: %s)$" % (pmetname.replace("$", "")),
        fontsize=gv.axes_labels_fontsize,
    )
    # ax.set_xscale('log')
    ax.set_ylabel(
        (
            "Variance in $\\log\\mathcal{N}_{%d}$  $\\left(%s\\right)$"
            % (level, ml.Metrics[lmet][1].replace("$", ""))
        ),
        fontsize=gv.axes_labels_fontsize,
    )
    ax.tick_params(
        axis="both",
        which="both",
        pad=gv.ticks_pad,
        direction="inout",
        length=gv.ticks_length,
        width=gv.ticks_width,
        labelsize=gv.ticks_fontsize,
    )
    # Attach a text label above each bar, indicating the numerical value of the variance.
    for rect in barplot:
        (height, width) = (rect.get_height(), rect.get_width())
        # print("height = %g, width = %d" % (height, width))
        ax.text(
            rect.get_x() + width / float(2),
            1.01 * height,
            "%g" % (height),
            ha="center",
            va="bottom",
            fontsize=gv.ticks_fontsize,
        )
    if pdf is None:
        # Save the plot
        pdfobj.savefig(fig)
        plt.close()
        # Set PDF attributes
        pdfInfo = pdfobj.infodict()
        pdfInfo["Title"] = "Binwise variance of %s with respect to %s." % (
            lmet,
            str(pmet),
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
        pdfobj.close()
    else:
        # Save the plot
        pdf.savefig(fig)
        plt.close()
    return None
