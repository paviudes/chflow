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
from define.QECCLfid import uncorrectable as uc

# Force the module scripts to run locally -- https://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path
# import inspect as ins
# current = os.path.realpath(os.path.abspath(os.path.dirname(ins.getfile(ins.currentframe()))))
# if (not (current in sys.path)):
# 	sys.path.insert(0, current)

from define import qchans as qc
from define import qcode as qec
from define import fnames as fn
from define import globalvars as gv
from define import metrics as ml
from define import submission as sub
from analyze import collect as cl


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
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
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
