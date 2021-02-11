# Critical packages
import os
import sys
import datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import colors, ticker, cm
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset
from scipy.interpolate import griddata

# Functions from other modules
from define.utils import IsNumber
from define.metrics import Metrics
from define import globalvars as gv
from analyze.load import LoadPhysicalErrorRates
from analyze.utils import DisplayForm, RealNoise, scientific_float, latex_float
from define.fnames import MCStatsPlotFile, PhysicalErrorRates, RunningAverageCh


def GetChannelPosition(rates, samples, available):
    """
    Compute the channel index corresponding to a rate and sample.
    """
    pos = np.zeros((rates.shape[0], samples.shape[0]), dtype=np.int)
    for r in range(rates.shape[0]):
        for s in range(samples.shape[0]):
            pos[r, s] = np.nonzero(
                list(
                    map(
                        lambda element: np.allclose(
                            np.concatenate((rates[r, :], [samples[s]])), element
                        ),
                        available,
                    )
                )
            )[0][0]
    return pos


def MaxFluctuationFilter(filter, rates, samples, running_averages, cutoff, xinclude):
    """
    Compute an inficator array to include only the leading fluctuations for every rate.
    """
    maximums = np.zeros((rates.shape[0], samples.shape[0]), dtype=np.int)
    for r in range(rates.shape[0]):
        for s in range(samples.shape[0]):
            filter[r, s] = 1 - np.any(
                running_averages[r, samples[s], xinclude] < 1 / cutoff
            )
    for r in range(rates.shape[0]):
        max_fluctuation = 0
        max_s = 0
        for s in range(samples.shape[0]):
            if filter[r, s] == 1:
                fluctuation = np.max(
                    running_averages[r, samples[s], xinclude]
                ) / np.min(running_averages[r, samples[s], xinclude])
                if max_fluctuation < fluctuation:
                    max_s = s
                    max_fluctuation = fluctuation
        if filter[r, max_s] == 1:
            maximums[r, max_s] = 1
    return maximums


def SetZones(plt, stats, running_averages, cutoff):
    """
    Set the filling for green, yellow and red zones.
    """
    # Green zone
    plt.fill_between(
        stats,
        10 * np.max(running_averages),
        10 / cutoff,
        interpolate=True,
        color="green",
        alpha=0.35,
        label="$Y >$%s" % DisplayForm(1 / cutoff, base=10),
    )
    # Yellow zone
    plt.fill_between(
        stats,
        10 / cutoff,
        1 / cutoff,
        interpolate=True,
        color="yellow",
        alpha=0.35,
        label="$Y \sim$%s" % DisplayForm(1 / cutoff, base=10),
    )
    # Red zone
    plt.fill_between(
        stats,
        1 / cutoff,
        np.min(running_averages) / 10,
        interpolate=True,
        color="red",
        alpha=0.35,
        label="$Y <$%s" % DisplayForm(1 / cutoff, base=10),
    )
    plt.axvline(x=cutoff, linestyle="--", color="k", linewidth=gv.line_width)
    return None


def LoadRunningAverages(dbs, lmet, rates, samples):
    """
    Load the running averages array.
    """
    running_averages = np.zeros(
        (rates.shape[0], samples.size, dbs.stats.size), dtype=np.double
    )
    for r in range(rates.shape[0]):
        for s in range(samples.size):
            running_averages[r, s, :] = np.load(
                RunningAverageCh(dbs, rates[r, :], samples[s], lmet)
            )
    return running_averages


def MCStatsPlot(dbses, lmet, pmet, rates, samples=None, cutoff=1e3):
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
    if samples is None:
        samples = np.arange(dbs.samps)
    
    if len(dbses) > 1:
        return MCompare(dbses, pmet, lmet, rates, samples, cutoff)
    else:
        dbs = dbses[0]
    min_stats = 0
    settings = {}
    LoadPhysicalErrorRates(dbs, pmet, settings, dbs.levels)
    phyerrs = settings["xaxis"]

    plotfname = MCStatsPlotFile(dbs, lmet, pmet)
    running_averages = LoadRunningAverages(dbs, lmet, rates, samples)

    # Compute the position of rate, sample in the availaibe list of channels.
    pos = GetChannelPosition(rates, samples, dbs.available)

    filter = np.ones((rates.shape[0], samples.shape[0]), dtype=np.int)
    with PdfPages(plotfname) as pdf:
        for case in range(2):
            # case 0 is the regular stat plot.
            # case 1 corresponds to a focused region plot: only the region in the green zone to the right of the stats line. Green zone is greater than 1/Stats.
            if case == 0:
                xindices = np.nonzero(dbs.stats >= min_stats)[0]
            else:
                xindices = np.nonzero(dbs.stats >= cutoff)[0]
                filter = MaxFluctuationFilter(
                    filter, rates, samples, running_averages, cutoff, xindices
                )
            fig = plt.figure(figsize=gv.canvas_size)
            ax = plt.gca()
            for r in range(rates.shape[0]):
                for s in range(samples.shape[0]):
                    if filter[r, s] == 0:
                        continue
                    # label = scientific_float(phyerrs[pos[r, s]])
                    xaxis = dbs.stats[xindices]
                    yaxis = running_averages[r, samples[s], xindices]
                    # print("xaxis\n{}\nyaxis\n{}".format(xaxis, yaxis))
                    label = None
                    if case == 0:
                        linestyle = gv.line_styles[s % len(gv.line_styles)]
                        if s == 0:
                            label = "$1 - F = %s$" % (
                                scientific_float(phyerrs[pos[r, s]]).replace("$", "")
                            )
                            # label = "$e = %s$" % (",".join(list(map(lambda x: "%g" % x, rates[r]))))
                    else:
                        label = "$1 - F = %s, \\lambda=%.1f$" % (
                            latex_float(phyerrs[pos[r, s]]).replace("$", ""),
                            np.max(yaxis) / np.min(yaxis),
                        )
                        linestyle = "-"
                    plt.plot(
                        xaxis,
                        yaxis,
                        linewidth=gv.line_width,
                        label=label,
                        color=gv.Colors[r % len(gv.Colors)],
                        linestyle=linestyle,
                        # marker="o",
                        # markersize=gv.marker_size * 0.75,
                    )
            # if case == 0:
            #     SetZones(plt, dbs.stats, running_averages, cutoff)

            # Axes limits and scaling
            ax.set_xlim(np.min(xaxis), np.max(xaxis))
            if case == 0:
                # ax.set_ylim(None, 10 * np.max(running_averages))
                ax.set_yscale("log")
                ax.set_xscale("log")
            else:
                ax.set_xscale("log")
                # ax.set_yscale("log")
                # ax.set_ylim(50 / cutoff, 55 / cutoff)
                # ax.set_ylim(1 / cutoff, 50 / cutoff)

            # Axes labels
            ax.set_xlabel(
                "$N$", fontsize=gv.axes_labels_fontsize, labelpad=gv.axes_labelpad
            )
            ax.set_ylabel(
                "$\\overline{%s_{%d}}$"
                % (Metrics[lmet]["latex"].replace("$", ""), dbs.levels),
                fontsize=gv.axes_labels_fontsize,
                labelpad=gv.axes_labelpad,
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
            # Legend
            plt.legend(
                numpoints=1,
                loc="upper center",
                ncol=4 if case == 0 else 3,
                bbox_to_anchor=(0.5, 1.17),
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
            % (Metrics[lmet]["name"], str(pmet), dbs.levels)
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None


def MCompare(dbses_input, pmet, lmet, rates, samples=None, cutoff=1e6):
    """
    Compare the running averages from many datasets.
    """
    __, uniques = np.unique([int(dbs.decoder_fraction * 4 ** (dbs.eccs[0].N)) for dbs in dbses_input], return_index=True)
    # print("uniques = {}".format(uniques))
    dbses = [dbses_input[d] for d in uniques]
    min_stats = 0
    # sampling_methods = {0: "Direct sampling", 1: "Importance sampling"}
    plotfname = MCStatsPlotFile(dbses[0], lmet, pmet)
    if samples is None:
        samples = np.arange(dbses[0].samps)
    with PdfPages(plotfname) as pdf:
        fig = plt.figure(figsize=gv.canvas_size)
        ax = plt.gca()
        dset_lines = []
        dset_labels = []
        for (d, dbs) in enumerate(dbses):
            settings = {}
            LoadPhysicalErrorRates(dbs, pmet, settings, dbs.levels)
            phyerrs = settings["xaxis"]
            running_averages = LoadRunningAverages(dbs, lmet, rates, samples)
            # Compute the position of rate, sample in the availaibe list of channels.
            pos = GetChannelPosition(rates, samples, dbs.available)
            xindices = np.nonzero(dbs.stats >= min_stats)[0]
            xaxis = dbs.stats[xindices]

            # Adding a line style for the dataset to the legend
            dset_lines.append(
                Line2D(
                    [0],
                    [0],
                    color="k",
                    linewidth=3,
                    linestyle=gv.line_styles[d % len(gv.line_styles)],
                )
            )
            # dset_labels.append(dbs.plotsettings["name"]) ORIGINAL
            for r in range(rates.shape[0]):
                for s in range(samples.shape[0]):
                    yaxis = running_averages[r, s, xindices]
                    # print("xaxis\n{}\nyaxis\n{}".format(xaxis, yaxis))
                    linestyle = gv.line_styles[d % len(gv.line_styles)]
                    label = "$%s = %s$" % (
                        Metrics[pmet]["latex"].replace("$", ""),
                        latex_float(phyerrs[pos[r, s]]),
                    )
                    plt.plot(
                        xaxis,
                        yaxis,
                        linewidth=gv.line_width,
                        # label=label if d == 0 else None, ORIGINAL
                        # label="%d" % int(dbs.decoder_fraction * 4 ** (dbs.eccs[0].N)),
                        label="%d,%s_%s" % (int(dbs.decoder_fraction * 4 ** (dbs.eccs[0].N)), dbs.timestamp[:2], dbs.timestamp[-2:]),
                        # color=gv.Colors[(r * samples.size + s) % len(gv.Colors)], ORIGINAL
                        color=gv.Colors[d % len(gv.Colors)],
                        # linestyle=linestyle, ORIGINAL
                        linestyle="-"
                        # marker="o",
                        # markersize=gv.marker_size * 0.75,
                    )
        # Axes limits and scaling
        ax.set_yscale("log")
        ax.set_xscale("log")
        # Axes labels
        ax.set_xlabel(
            "$N$", fontsize=gv.axes_labels_fontsize, labelpad=gv.axes_labelpad
        )
        ax.set_ylabel(
            "$\\overline{%s_{%d}}$"
            % (Metrics[lmet]["latex"].replace("$", ""), dbs.levels),
            fontsize=gv.axes_labels_fontsize,
            labelpad=gv.axes_labelpad,
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
        # Legend
        mc_handles, mc_labels = ax.get_legend_handles_labels()

        plt.legend(
            handles=mc_handles + dset_lines,
            labels=mc_labels + dset_labels,
            numpoints=1,
            loc="upper center",
            # ncol=4, ORIGINAL
            ncol=6,
            bbox_to_anchor=(0.5, 1.15),
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
            % (Metrics[lmet]["name"], str(pmet), dbs.levels)
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()

        # Vertical line at cutoff
        plt.axvline(x=cutoff, linestyle="--", color="k", linewidth=gv.line_width)

    return None
