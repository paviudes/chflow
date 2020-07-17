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
except:
    pass

from define import qcode as qec
from define import fnames as fn
from define import globalvars as gv


def PauliDistributionPlot(
    qcode, pauliprobs, nreps=5, max_weight=None, outdir="./", channel="unknown"
):
    r"""
    Plot the probability distribution of Pauli errors.
    """
    if max_weight is None:
        max_weight = int(np.max(qcode.weightdist))
    max_weight += 1
    group_by_weight = {w: None for w in range(max_weight)}
    for w in range(max_weight):
        (group_by_weight[w],) = np.nonzero(qcode.weightdist == w)
    # print(
    #     "Group by weight 1: {}\nGroup by weight 2: {}".format(
    #         qcode.PauliOperatorsLST[group_by_weight[1]],
    #         qcode.PauliOperatorsLST[group_by_weight[2]],
    #     )
    # )
    leading_by_weight = {w: None for w in range(max_weight)}
    for w in range(max_weight):
        ninclude = min(nreps, group_by_weight[w].size)
        result_args = np.argsort(-pauliprobs[group_by_weight[w]])[:ninclude]
        leading_by_weight[w] = group_by_weight[w][result_args]
    operator_labels = {
        w: qec.PauliOperatorToSymbol(qcode.PauliOperatorsLST[leading_by_weight[w]])
        for w in range(max_weight)
    }
    print(
        "\033[2mLeading by weight:\n{}\nOperator labels:\n{}\033[0m".format(
            leading_by_weight, operator_labels
        )
    )
    for w in range(3):
        print(
            "\033[2mpauliprobs[leading_by_weight[{}]] = {}\033[0m".format(
                w, pauliprobs[leading_by_weight[w]]
            )
        )
    plotfname = fn.PauliDistribution(outdir, channel)
    with PdfPages(plotfname) as pdf:
        fig = plt.figure(figsize=gv.canvas_size)
        current = 0
        for w in range(1, max_weight):
            if leading_by_weight[w].size > 0:
                plt.bar(
                    np.arange(current, current + leading_by_weight[w].size),
                    pauliprobs[leading_by_weight[w]],
                    color=gv.Colors[w % gv.n_Colors],
                    width=0.5,
                    label="w = %d" % (w),
                    alpha=0.6,
                )
                current = current + leading_by_weight[w].size
        # Principal axes labels
        ax = plt.gca()
        # ax.set_xlabel("Errors", fontsize=gv.axes_labels_fontsize)
        # locs, labels = xticks()
        # xticks(np.arange(0, 1, step=0.2))
        plt.xticks(
            np.arange(sum([len(operator_labels[w]) for w in range(1, max_weight)])),
            [label for w in range(1, max_weight) for label in operator_labels[w]],
            rotation=45,
        )
        ax.set_ylabel("Probabilities", fontsize=gv.axes_labels_fontsize)
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
        ax.legend(
            numpoints=1,
            loc=1,
            shadow=True,
            fontsize=gv.legend_fontsize,
            markerscale=gv.legend_marker_scale,
        )
        # Save the plot
        pdf.savefig(fig)
        plt.close()
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = "Pauli distribution of errors."
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None
