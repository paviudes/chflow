# Critical packages
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset
from scipy.interpolate import griddata

# Non critical packages
try:
    import PyPDF2 as pp
except ImportError:
    pass

# Functions from other modules
from define.fnames import LogicalErrorRates, PhysicalErrorRates, ThreshPlot


def ThresholdPlot(phymets, logmet, dbs):
    # For each physical noise rate, plot the logical error rate vs. the levels of concatenation.
    # The set of curves should have a bifurcation at the threshold.

    # print("dbs.available")
    # print dbs.available

    phylist = list(map(lambda phy: phy.strip(" "), phymets.split(",")))
    sampreps = np.hstack((np.nonzero(dbs.available[:, -1] == 0)[0], [dbs.channels]))
    # print("sampreps\n%s" % (np.array_str(sampreps)))
    logErrs = np.load(LogicalErrorRates(dbs, logmet, fmt="npy"))
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
            phyparams.append(qc.Channels[dbs.channel]["latex"][np.int8(phylist[m])])
        else:
            # print("loading: %s" % (PhysicalErrorRates(dbs, phylist[m])))
            phyrates = np.load(PhysicalErrorRates(dbs, phylist[m]))
            # print("metric = %s, phyrates\n%s" % (phylist[m], np.array_str(phyrates)))
            for i in range(sampreps.shape[0] - 1):
                # print("phyrates[%d:%d, np.int8(phylist[m])]\n%s" % (sampreps[i], sampreps[i + 1], np.array_str(phyrates[sampreps[i]:sampreps[i + 1]])))
                phyerrs[i, m] = np.sum(
                    phyrates[sampreps[i] : sampreps[i + 1]], dtype=np.longdouble
                ) / np.longdouble(sampreps[i + 1] - sampreps[i])
            phyparams.append(ml.Metrics[phylist[m]]["latex"])
    # print("phyerrs")
    # print phyerrs
    plotfname = ThreshPlot(dbs, "_".join(phylist), logmet)
    with PdfPages(plotfname) as pdf:
        for m in range(len(phylist)):
            fig = plt.figure(figsize=gv.canvas_size)
            for p in range(phyerrs.shape[0]):
                fmtidx = list(ml.Metrics.keys())[p % len(ml.Metrics)]
                plt.plot(
                    np.arange(dbs.levels + 1),
                    logErr[p, :],
                    label=("%s = %s" % (phyparams[m], DisplayForm(phyerrs[p, m], 10))),
                    color=ml.Metrics[fmtidx]["color"],
                    marker=ml.Metrics[fmtidx]["marker"],
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
                % (qc.Channels[dbs.channel]["name"], phyparams[m]),
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
                % (ml.Metrics[logmet]["latex"].replace("$", "")),
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
            ml.Metrics[logmet]["log"],
            ", ".join(map(str, range(1, 1 + dbs.levels))),
            ", ".join(phyparams),
            dbs.channels,
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None
