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
from scipy.interpolate import griddata
import PyPDF2 as pp

from analyze.load import LoadPhysicalErrorRates
from analyze.bins import ComputeBinVariance
from define import fnames as fn
from define import metrics as ml
from define import globalvars as gv

def DecoderCompare(phymet, logmet, dbses, nbins = 10, thresholds={"y": 10e-16, "x": 10e-16}):
	# Compare performance of various decoders.
	ndb = len(dbses)
	plotfname = fn.DecodersPlot(dbses[0], phymet, logmet)
	nlevels = max([db.levels for db in dbses])
	with PdfPages(plotfname) as pdf:
		for l in range(1, nlevels + 1):
			fig = plt.figure(figsize=gv.canvas_size)
			ax1 = plt.gca()
			for d in range(ndb):
				settings = {
				    "xaxis": None,
				    "xlabel": None,
				    "yaxis": np.load(fn.LogicalErrorRates(dbses[d], logmet))[:, l],
				    "ylabel": "$\\overline{%s_{%d}}$" % (ml.Metrics[logmet]["latex"].replace("$",""), l),
				    "color": gv.Colors[d % gv.n_Colors]
				    if ndb > 1
				    else ml.Metrics[phylist[p]]["color"],
				    "marker": gv.Markers[d % gv.n_Markers]
				    if ndb > 1
				    else ml.Metrics[phylist[p]]["marker"],
				    "linestyle": "",
				}
				if dbses[d].decoders[l-1] == 0:
					decoder_info = "Maximum likelihood"
				elif dbses[d].decoders[l-1] == 1:
					decoder_info = "Minimum weight"
				else:
					decoder_info = "$\\alpha = %g$" % (dbses[d].decoder_fraction)
				LoadPhysicalErrorRates(dbses[d], phymet, settings, l)
				bins = ComputeBinVariance(settings["xaxis"], settings["yaxis"], nbins=nbins, space="log")
				# Plotting
				plotobj = ax1.plot(
				    (bins[:,0] + bins[:,1])/2,
				    bins[:, 6],
				    color=settings["color"],
				    alpha=0.75,
				    marker="o", # settings["marker"]
				    markersize=gv.marker_size,
				    linestyle=settings["linestyle"],
				    linewidth=gv.line_width,
				    label=decoder_info
				)
			# Axes labels
			ax1.set_xlabel(settings["xlabel"], fontsize=gv.axes_labels_fontsize * 0.8, labelpad = gv.axes_labelpad)
			ax1.set_xscale("log")
			ax1.set_ylabel(settings["ylabel"], fontsize=gv.axes_labels_fontsize, labelpad = gv.axes_labelpad)
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
			ax1.legend(
			    numpoints=1,
			    loc="lower right",
			    shadow=True,
			    fontsize=gv.legend_fontsize,
			    markerscale=gv.legend_marker_scale,
			)
			# Save the plot
			pdf.savefig(fig)
			plt.close()
		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Comparing different decoders"
		pdfInfo["Author"] = "Pavithran Iyer"
		pdfInfo["ModDate"] = dt.datetime.today()
	return None