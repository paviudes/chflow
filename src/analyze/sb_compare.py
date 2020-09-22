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
    from scipy.interpolate import griddata
    import PyPDF2 as pp
except:
    pass


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
                        color=ml.Metrics[logmet]["color"],
                        marker=ml.Metrics[logmet]["marker"],
                        markersize=gv.marker_size,
                        linestyle="None",
                    )
                    plt.title(
                        "Level %d %s for %s vs. %s."
                        % (
                            l,
                            ml.Metrics[logmet]["latex"],
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
            ml.Metrics[logmet]["name"],
            "_".join([dbses[i].timestamp for i in range(ndb)]),
            cdepth,
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None
