import datetime as dt

try:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colors, ticker, cm
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
except:
    pass
from define import globalvars as gv
from define import fnames as fn
from define import metrics as ml


def epsilon(alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    return np.sqrt(np.tanh(np.power(np.abs(alpha), 2.0)))


def r(alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    return (
        (np.power(epsilon(alpha), 4.0) - 1)
        * np.power(np.abs(alpha), 2.0)
        / np.power(epsilon(alpha), 2.0)
    )


def q(alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    return (
        (np.power(epsilon(alpha), 4.0) + 1)
        * np.power(np.abs(alpha), 2.0)
        / np.power(epsilon(alpha), 2.0)
    )


def p(alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    return (
        np.power((np.power(epsilon(alpha), 2.0) + 1), 2.0)
        * np.power(np.abs(alpha), 2.0)
        / (2 * np.power(epsilon(alpha), 2.0))
    )


def m(alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    return (
        np.power((np.power(epsilon(alpha), 2.0) - 1), 2.0)
        * np.power(np.abs(alpha), 2.0)
        / (2 * np.power(epsilon(alpha), 2.0))
    )


def R(gamma, alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    return (Q(gamma, alpha) - 1) * r(alpha) / q(alpha)


def Q(gamma, alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    return np.power(1 - gamma, q(alpha))


def P(gamma, alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    return np.power(1 - gamma, p(alpha))


def M(gamma, alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    return np.power(1 - gamma, m(alpha))


def Fp(gamma, alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    return np.sqrt(
        np.power(M(gamma, alpha) + P(gamma, alpha), 2.0)
        + np.power(R(gamma, alpha), 2.0)
    )


def Fm(gamma, alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    return np.sqrt(
        np.power(M(gamma, alpha) - P(gamma, alpha), 2.0)
        + np.power(R(gamma, alpha), 2.0)
    )


def F1(gamma, alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    return np.sqrt(
        Fm(gamma, alpha) / (M(gamma, alpha) - P(gamma, alpha) + Fm(gamma, alpha))
    )


def F2(gamma, alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    # print("M(%g, %g) = %g, P(%g, %g) = %g, Fm(%g, %g) = %g." % (gamma, alpha, M(gamma, alpha), gamma, alpha, P(gamma, alpha), gamma, alpha, Fm(gamma, alpha)))
    return np.sqrt(
        Fm(gamma, alpha) / (-M(gamma, alpha) + P(gamma, alpha) + Fm(gamma, alpha))
    )


def F3(gamma, alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    return np.sqrt(
        Fp(gamma, alpha) * (M(gamma, alpha) + P(gamma, alpha) + Fp(gamma, alpha))
    ) / np.longdouble(R(gamma, alpha))


def F4(gamma, alpha):
    # coefficients for simplifying the expression for the Krauss operators.
    return np.sqrt(
        Fp(gamma, alpha) / (M(gamma, alpha) + P(gamma, alpha) + Fp(gamma, alpha))
    )


def KraussCoefficients(gamma, alpha):
    # Return the four coefficients that are multiplied with the four Krauss operators.
    coeffs = np.zeros(4, dtype=np.longdouble)
    coeffs[0] = (1 - Q(gamma, alpha) - Fm(gamma, alpha)) / np.longdouble(2)
    coeffs[1] = (1 - Q(gamma, alpha) + Fm(gamma, alpha)) / np.longdouble(2)
    coeffs[2] = (1 + Q(gamma, alpha) - Fp(gamma, alpha)) / np.longdouble(2)
    coeffs[3] = (1 + Q(gamma, alpha) + Fp(gamma, alpha)) / np.longdouble(2)
    return coeffs


def K1(gamma, alpha):
    # return the expression for the first Krauss operator
    krop = np.zeros((2, 2), dtype=np.complex128)
    krop[0, 1] = (
        1j
        * (-M(gamma, alpha) + P(gamma, alpha) - R(gamma, alpha) + Fm(gamma, alpha))
        / (2 * R(gamma, alpha) * F1(gamma, alpha))
    )
    krop[1, 0] = (
        1j
        * (-M(gamma, alpha) + P(gamma, alpha) + R(gamma, alpha) + Fm(gamma, alpha))
        / (2 * R(gamma, alpha) * F1(gamma, alpha))
    )
    return krop


def K2(gamma, alpha):
    # return the expression for the second Krauss operator
    krop = np.zeros((2, 2), dtype=np.complex128)
    krop[0, 1] = (
        -1j
        * (M(gamma, alpha) - P(gamma, alpha) + R(gamma, alpha) + Fm(gamma, alpha))
        / (2 * R(gamma, alpha) * F2(gamma, alpha))
    )
    krop[1, 0] = (
        1j
        * (-M(gamma, alpha) + P(gamma, alpha) + R(gamma, alpha) - Fm(gamma, alpha))
        / (2 * R(gamma, alpha) * F2(gamma, alpha))
    )
    return krop


def K3(gamma, alpha):
    # return the expression for the third Krauss operator
    krop = np.zeros((2, 2), dtype=np.complex128)
    krop[0, 0] = (
        -M(gamma, alpha) - P(gamma, alpha) + R(gamma, alpha) - Fp(gamma, alpha)
    ) / (2 * R(gamma, alpha) * F3(gamma, alpha))
    krop[1, 1] = (
        M(gamma, alpha) + P(gamma, alpha) + R(gamma, alpha) + Fp(gamma, alpha)
    ) / (2 * R(gamma, alpha) * F3(gamma, alpha))
    return krop


def K4(gamma, alpha):
    # return the expression for the fourth Krauss operator
    krop = np.zeros((2, 2), dtype=np.complex128)
    krop[0, 0] = (
        -M(gamma, alpha) - P(gamma, alpha) + R(gamma, alpha) + Fp(gamma, alpha)
    ) / (2 * R(gamma, alpha) * F4(gamma, alpha))
    krop[1, 1] = (
        M(gamma, alpha) + P(gamma, alpha) + R(gamma, alpha) - Fp(gamma, alpha)
    ) / (2 * R(gamma, alpha) * F4(gamma, alpha))
    return krop


def PLKrauss(gamma, alpha):
    # Return the Krauss operators for the Photon Loss channel
    krauss = np.zeros((4, 2, 2), dtype=np.complex128)
    coeffs = KraussCoefficients(gamma, alpha)
    krauss[0, :, :] = np.sqrt(coeffs[0]) * K1(gamma, alpha)
    krauss[1, :, :] = np.sqrt(coeffs[1]) * K2(gamma, alpha)
    krauss[2, :, :] = np.sqrt(coeffs[2]) * K3(gamma, alpha)
    krauss[3, :, :] = np.sqrt(coeffs[3]) * K4(gamma, alpha)
    return krauss


def PLKraussOld(gamma, alpha):
    # Return the Krauss operators for the Photon loss channel.
    krauss = np.zeros((2, 2, 2), dtype=np.complex128)
    krauss[0, :, :] = oldK1(gamma, alpha)
    krauss[1, :, :] = oldK2(gamma, alpha)
    return krauss


def oldK1(gamma, alpha):
    # Return the (old version of) Krauss operators corresponding to the idetity term in the decomposition.
    krop = np.zeros((2, 2), dtype=np.complex128)
    krop[[0, 1], [0, 1]] = np.sqrt(
        np.power((1 - gamma), np.power(alpha, 2.0))
        + np.power((1 - gamma), (-1) * np.power(alpha, 2.0))
    ) / np.sqrt(2.0)
    return krop


def oldK2(gamma, alpha):
    # Return the (old version of) Krauss operators corresponding to the Photon loss term in the decomposition.
    krop = np.zeros((2, 2), dtype=np.complex128)
    krop[0, 1] = np.sqrt(
        (
            np.power((1 - gamma), (-1) * np.power(alpha, 2.0))
            - np.power((1 - gamma), np.power(alpha, 2.0))
        )
        / (np.longdouble(2.0) * np.tanh(np.power(alpha, 2.0)))
    )
    krop[1, 0] = krop[0, 1] * np.tanh(np.power(alpha, 2.0))
    return krop


def PLThreshPlots(submit, logmet):
    # Plot logical error rates vs physical error rates, for the photon loss channel.
    # The X-axis is gamma, the Y-axis is the selected logical metric.
    # There will be one curve for every concatenation level and a new plot for every alpha.
    # The logical error rates are ordered according to the cartesian product: (gamma values x alpha values).
    logErrorRates = np.load(fn.LogicalErrorRates(submit, logmet, fmt="npy"))
    xaxis = np.power(2 / np.longdouble(3), submit.noiserange[0])
    plotfname = fn.ThreshPlot(submit, "gamma", logmet)
    with PdfPages(plotfname) as pdf:
        for a in range(submit.noiserange[1].shape[0]):
            fig = plt.figure(figsize=gv.canvas_size)
            nqubits = 1
            dist = 1
            for l in range(submit.levels):
                nqubits = nqubits * submit.eccs[l].N
                dist = dist * submit.eccs[l].D
                yaxis = logErrorRates[
                    np.arange(submit.noiserange[0].shape[0])
                    * submit.noiserange[1].shape[0]
                    + a,
                    l,
                ]
                plt.plot(
                    xaxis,
                    yaxis,
                    label=("N = %d, d = %d" % (nqubits, dist)),
                    color=ml.Metrics[ml.Metrics.keys()[l]][3],
                    marker=ml.Metrics[ml.Metrics.keys()[l]][2],
                    markersize=gv.marker_size,
                    linestyle="--",
                    linewidth=gv.line_width,
                )
            # Title
            plt.title(
                "$\\alpha = %g$" % (submit.noiserange[1][a]),
                fontsize=gv.title_fontsize,
                y=1.03,
            )
            # Legend
            plt.legend(
                numpoints=1,
                loc=4,
                shadow=True,
                fontsize=gv.legend_fontsize,
                markerscale=gv.legend_marker_scale,
            )
            # Axes labels
            ax = plt.gca()
            ax.set_xlabel("$\\gamma$", fontsize=gv.axes_labels_fontsize)
            ax.set_xscale("log")
            ax.set_ylabel(ml.Metrics[logmet][1], fontsize=gv.axes_labels_fontsize)
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
            # Save the plot
            pdf.savefig(fig)
            plt.close()
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = "%s at levels %s, with gamma for %d channels." % (
            ml.Metrics[logmet][0],
            ", ".join(map(str, range(1, 1 + submit.levels))),
            submit.channels,
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None


def PLPerfPlots(submit, logmet):
    # Plot logical error rates vs physical error rates, for the photon loss channel.
    # The X-axis is gamma, the Y-axis is the selected logical metric.
    # There will be one curve for every alpha and a new plot for every concatenation level.
    # The logical error rates are ordered according to the cartesian product: (gamma values x alpha values).
    logErrorRates = np.load(fn.LogicalErrorRates(submit, logmet, fmt="npy"))
    xaxis = np.power(2 / np.longdouble(3), submit.noiserange[0])
    plotfname = fn.LevelWise(submit, "gamma", logmet)
    with PdfPages(plotfname) as pdf:
        nqubits = 1
        dist = 1
        for l in range(submit.levels):
            nqubits = nqubits * submit.eccs[l].N
            dist = dist * submit.eccs[l].D
            fig = plt.figure(figsize=gv.canvas_size)
            for a in range(submit.noiserange[1].shape[0]):
                yaxis = logErrorRates[
                    np.arange(submit.noiserange[0].shape[0])
                    * submit.noiserange[1].shape[0]
                    + a,
                    l,
                ]
                plt.plot(
                    xaxis,
                    yaxis,
                    label="$\\alpha = %g$" % (submit.noiserange[1][a]),
                    color=ml.Metrics[ml.Metrics.keys()[a]][3],
                    marker=ml.Metrics[ml.Metrics.keys()[a]][2],
                    markersize=gv.marker_size,
                    linestyle="--",
                    linewidth=gv.line_width,
                )
            # Title
            plt.title(
                "N = %d, d = %d" % (nqubits, dist), fontsize=gv.title_fontsize, y=1.03
            )
            # Legend
            plt.legend(
                numpoints=1,
                loc=4,
                shadow=True,
                fontsize=gv.legend_fontsize,
                markerscale=gv.legend_marker_scale,
            )
            # Axes labels
            ax = plt.gca()
            ax.set_xlabel("$\\gamma$", fontsize=gv.axes_labels_fontsize)
            ax.set_xscale("log")
            ax.set_ylabel(ml.Metrics[logmet][1], fontsize=gv.axes_labels_fontsize)
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
            # Save the plot
            pdf.savefig(fig)
            plt.close()
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = "%s at levels %s, with gamma for %d channels." % (
            ml.Metrics[logmet][0],
            ", ".join(map(str, range(1, 1 + submit.levels))),
            submit.channels,
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None


def PLPerfPlots2D(submit, logmet):
    # Plot performance contours for the logical error rates for every concatenation level, with repect to the dephasing and relaxation rates.
    # Each plot will be a contour plot or a color density plot indicating the logical error, with the x-axis as the dephasing rate and the y-axis as the relaxation rate.
    # There will be one plot for every concatenation level.
    logErr = np.load(fn.LogicalErrorRates(submit, logmet, fmt="npy"))
    (meshX, meshY) = np.meshgrid(
        np.linspace(
            submit.noiserates[:, 0].min(),
            submit.noiserates[:, 0].max(),
            max(100, submit.noiserates.shape[0]),
        ),
        np.linspace(
            submit.noiserates[:, 1].min(),
            submit.noiserates[:, 1].max(),
            max(100, submit.noiserates.shape[0]),
        ),
    )
    # print("meshX\n%s\nmeshY\n%s\nlogErr\n%s" % (np.array_str(meshX), np.array_str(meshY), np.array_str(logErr)))
    plotfname = fn.LevelWise(submit, "gamma_alpha", logmet)
    with PdfPages(plotfname) as pdf:
        nqubits = 1
        dist = 1
        for l in range(submit.levels):
            nqubits = nqubits * submit.eccs[l].N
            dist = dist * submit.eccs[l].D
            fig = plt.figure(figsize=gv.canvas_size)
            meshZ = griddata(
                (submit.noiserates[:, 0], submit.noiserates[:, 1]),
                logErr[:, l + 1],
                (meshX, meshY),
                method="cubic",
            )
            # print("Z\n%s" % (np.array_str(meshZ)))
            cplot = plt.contourf(
                meshX,
                meshY,
                meshZ,
                cmap=cm.bwr,
                locator=ticker.LogLocator(),
                linestyles=gv.contour_linestyle,
                pad=gv.ticks_pad,
            )
            plt.scatter(
                submit.noiserates[:, 0], submit.noiserates[:, 1], marker="o", color="k"
            )
            # Title
            plt.title(
                "N = %d, d = %d" % (nqubits, dist), fontsize=gv.title_fontsize, y=1.03
            )
            # Axes labels
            ax = plt.gca()
            ax.set_xlabel("$\\gamma$", fontsize=gv.axes_labels_fontsize)
            # ax.set_xscale('log')
            ax.set_ylabel("$\\alpha$", fontsize=gv.axes_labels_fontsize)
            # ax.set_yscale('log')
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
            cbar = plt.colorbar(cplot)
            cbar.ax.set_xlabel(ml.Metrics[logmet][1], fontsize=gv.colorbar_fontsize)
            cbar.ax.tick_params(labelsize=gv.legend_fontsize)
            cbar.ax.xaxis.labelpad = gv.ticks_pad
            # Save the plot
            pdf.savefig(fig)
            plt.close()
        # Set PDF attributes
        pdfInfo = pdf.infodict()
        pdfInfo["Title"] = "%s at levels %s, for %d Photon loss channels." % (
            ml.Metrics[logmet][0],
            ", ".join(map(str, range(1, 1 + submit.levels))),
            submit.channels,
        )
        pdfInfo["Author"] = "Pavithran Iyer"
        pdfInfo["ModDate"] = dt.datetime.today()
    return None


if __name__ == "__main__":
    import chanreps as crep

    # test the properties of the Photon Loss channel
    gamma = 0.8
    alpha = 5
    plkrauss = PLKraussOld(gamma, alpha)
    print(
        "The Krauss operators for the Photon Loss channel with alpha = %g, gamma = %g are the following.\n%s"
        % (alpha, gamma, np.array_str(plkrauss))
    )
    plprocess = crep.ConvertRepresentations(plkrauss, "krauss", "choi")
    print(
        "The choi matrix for the Photon Loss channel with alpha = %g, gamma = %g is the following.\n%s"
        % (alpha, gamma, np.array_str(plprocess))
    )
