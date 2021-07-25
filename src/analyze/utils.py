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
from define import qchans as qc


def RoundOrder(number):
    # Round a number to the nearest order: a * 10^-b where a is a multiple of 5 and b is an integer.
    (base, exponent) = GetBaseExponent(number)
    base = 5 * np.round(base/5)
    return (base, exponent)


def OrderOfMagnitude(number):
    # Compute the order of magnitude
    if (number < 0):
        return np.int(np.floor(np.log10(-1 * number)))
    return np.int(np.ceil(np.log10(number)))


def GetBaseExponent(number):
    # Separate the base and exponent.
    float_str = "{0:.1e}".format(number)
    (base, exponent) = float_str.split("e")
    return (float(base), float(exponent))


def SetTickLabels(axis, scale="log", interval=1):
    # Set the positions of the axis ticks and the corresponding axis labels.
    # lower = np.floor(np.log10(np.min(axis)))
    # upper = np.ceil(np.log10(np.max(axis)))
    lower = OrderOfMagnitude(np.min(axis))
    upper = OrderOfMagnitude(np.max(axis))
    # print("X axis\nmin = {}, max = {}\nupper = {}, lower = {}".format(np.min(axis), np.max(axis), upper, lower))
    if (abs(upper - lower) > 3):
        interval = max(1, interval)
        orders = np.arange(lower, upper + interval, interval)
        ticks = np.power(0.1, -1 * orders)
    elif (abs(upper - lower) == 0):
        interval = min(0.5, interval)
        (left_base, left_exponent) = GetBaseExponent(np.min(axis))
        (right_base, right_exponent) = GetBaseExponent(np.max(axis))
        ticks = np.arange(left_base, right_base + interval, interval) * np.power(0.1, -1 * left_exponent)
    else:
        lower -= 1
        (min_base, min_exponent) = RoundOrder(np.min(axis))
        # upper += 1
        (max_base, max_exponent) = RoundOrder(np.max(axis))

        interval = min(0.5, interval)
        orders = np.arange(lower, upper + 1)
        # print("orders\n{}".format(orders))
        ticks = np.sort(np.concatenate((np.power(0.1, -1 * orders), interval * 10 * np.power(0.1, -1 * orders[:-1]))))
        # print("ticks\n{}".format(ticks))
        if (min_base >= 5):
            ticks = ticks[1:]
        if (max_base < 5):
            ticks = ticks[:-2]

    tick_labels = list(map(lambda x: "$%s$" % latex_float(x), ticks))
    return (ticks, tick_labels)


def ExtractPDFPages(information, save_folder, save_fname):
    r"""
    Extract pages from a PDF and save it into a new PDF.
    https://stackoverflow.com/questions/51567750/extract-specific-pages-of-pdf-and-save-it-with-python
    """
    for f in range(len(information)):
        pdfobj = open("%s" % information[f]["fname"], "rb")
        pdfReader = pp.PdfFileReader(pdfobj)
        pdf_writer = pp.PdfFileWriter()
        from_page = information[f]["start"]
        to_page = information[f]["end"]
        for p in range(from_page, to_page + 1):
            pdf_writer.addPage(pdfReader.getPage(from_page - 1))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        pdfname = "%s/pg_%d%s" % (save_folder, from_page, save_fname)
        if os.path.isfile(pdfname):
            print("\033[3m!!!Warning, overwriting %s.\033[0m" % pdfname)
            os.system("trash %s" % (pdfname))
        with open(pdfname, "wb") as out:
            pdf_writer.write(out)
        # Convert the PDF to PNG
        fname = "%s/pg_%d%s" % (
            save_folder,
            from_page,
            os.path.os.path.splitext(save_fname)[0],
        )
        print(
            "\033[2mPDF file written to %s/pg_%d_%s.\033[0m"
            % (save_folder, from_page, save_fname)
        )
        os.system("sips -Z 800 -s format png %s.pdf --out %s.png" % (fname, fname))
        print("\033[2mPNG file written to %s.\033[0m" % (fname))
    return None


def latex_float(f):
    # Function taken from: https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python
    float_str = "{0:.1e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        if abs(int(exponent)) <= 3:
            return ("%g" % f)
        if (abs(float(base) - 1) <= 1E-10):
            return r"10^{{{0}}}".format(int(exponent))
        if float(base).is_integer():
            return r"{0} \times 10^{{{1}}}".format(int(float(base)), int(exponent))
        return r"%g \times 10^{%d}" % (float(base), int(exponent))
    else:
        return float_str


def scientific_float(f):
    # Function taken from: https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python
    float_str = "{0:.1e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0}e{1}".format(base, int(exponent))
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
    if (sign == "") and (factor == 1):
        numstr = "$%g^{%d}$" % (base, exponent)
    else:
        numstr = "$%s%g \\times %g^{%d}$" % (sign, factor, base, exponent)
    # if (number < 0):
    # 	numstr = ("-%s" % (numstr))
    return numstr


def GetNKDString(dbs, l):
    r"""
    Compute N,K,D for the concatenated code and return it as a string.
    """
    if l == 0:
        return ("Physical channel: %s") % (qc.Channels[dbs.channel]["name"])
    return ("N = %d, D = %d, %s") % (
        np.prod([dbs.eccs[j].N for j in range(l)]),
        np.prod([dbs.eccs[j].D for j in range(l)]),
        qc.Channels[dbs.channel]["name"],
    )


def RealNoise(scales, ratesexp):
    # Output the set of noise rates given the noise rate specifiers
    rates = np.zeros(ratesexp.shape[0], dtype=np.longdouble)
    for i in range(ratesexp.shape[0]):
        if scales[i] == 1:
            rates[i] = ratesexp[i]
        else:
            rates[i] = np.power(scales[i], ratesexp[i])
    return rates
