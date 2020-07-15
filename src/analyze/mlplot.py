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
                label=ml.Metrics[pmet]["latex"],
                color=ml.Metrics[pmet]["color"],
                marker=ml.Metrics[pmet]["marker"],
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
                    % (l, ml.Metrics[lmet]["latex"].replace("$", ""))
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
            AddBinVariancePlot(bins, l, lmet, pmet, ml.Metrics[pmet]["latex"], pdf)
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
                ml.Metrics[pmet]["name"],
                ", ".join(list(map(lambda str: "%d" % str, range(1, 1 + dbs.levels)))),
                ml.Metrics[lmet]["name"],
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
                label=ml.Metrics[pmet]["latex"],
                color=ml.Metrics[pmet]["color"],
                marker=ml.Metrics[pmet]["marker"],
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
                    % (l, ml.Metrics[lmet]["latex"].replace("$", ""))
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
            AddBinVariancePlot(bins, l, lmet, pmet, ml.Metrics[pmet]["latex"], pdf)
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
                ml.Metrics[pmet]["name"],
                ", ".join(list(map(lambda str: "%d" % str, range(1, 1 + dbs.levels)))),
                ml.Metrics[lmet]["name"],
                dbs.channels,
            )
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None
