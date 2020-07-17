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

try:
    import PyPDF2 as pp
except ImportError:
    print("PyPDF2 is not available, cannot manipulate PDFs.")

from analyze.load import LoadPhysicalErrorRates
from define import fnames as fn
from define import metrics as ml
from define import globalvars as gv


def RelativeImprovement(xaxis, yaxes, plt, ax1, xlabel, only_points, annotations=None):
	"""
	Plot relative improvement from RC, in an inset plot.
	We will compute the difference: (second row - first row)/(second row)
	The first row in yaxes refers to RC data while the second row refers to non RC data.
	https://scipython.com/blog/inset-plots-in-matplotlib/
	"""
	atol = 10e-8
	degrading_indices = (yaxes[0, :] - yaxes[1, :]) > atol
	print(
		"Logical error rates:\n RC: {}\n no RC: {}".format(
			yaxes[0, degrading_indices], yaxes[1, degrading_indices]
		)
	)
	ax2 = plt.axes([0, 0, 1, 1])
	# Manually set the position and relative size of the inset axes within ax1
	ip = InsetPosition(ax1, [0.1, 0.67, 0.33, 0.3])
	ax2.set_axes_locator(ip)
	# Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
	mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none")
	for i in only_points:
		ax2.plot(
			xaxis[i],
			yaxes[1, i]/yaxes[0, i],
			color=gv.Colors[i % gv.n_Colors],
			marker="o",
			markersize=gv.marker_size,
		)
		if annotations is not None:
			ax2.annotate(
				annotations[i],
				(0.92 * xaxis[i], 0.89 * (yaxes[1, i] - yaxes[0, i]) / yaxes[1, i]),
				color=gv.Colors[i % gv.n_Colors],
				fontsize=gv.ticks_fontsize * 0.75,
			)
	# Draw a horizontal line at Y=0 to show the break-even point RC and no RC.
	ax2.axhline(y=1, linestyle="--")
	ax2.set_xlabel(xlabel, fontsize=gv.axes_labels_fontsize / 2)
	ax2.set_ylabel("Relative improvement", fontsize=gv.axes_labels_fontsize / 2)
	ax2.set_xscale("log")
	# ax2.set_yscale("log")
	ax2.tick_params(
		axis="both",
		which="both",
		pad=gv.ticks_pad,
		direction="inout",
		length=gv.ticks_length,
		width=gv.ticks_width,
		labelsize=gv.ticks_fontsize * 0.75,
	)
	# ax2.legend(loc=0, fontsize=gv.legend_fontsize / 2)
	return None


def ChannelWisePlot(phymet, logmet, dbses, thresholds={"y": 10e-16, "x": 10e-16}, include_input=None):
	# Plot each channel in the database with a different color.
	# Channels of similar type in different databases will be distinguished using different markers.
	ndb = len(dbses)
	plotfname = fn.ChannelWise(dbses[0], phymet, logmet)
	maxlevel = max([db.levels for db in dbses])
	annotations = None
	select_count = min(10, dbses[0].channels)
	if dbses[0].channels < 7:
		annotations = [
			("$\\mathcal{U}_{%d}$" % (i + 1)) for i in range(dbses[0].channels)
		]
	with PdfPages(plotfname) as pdf:
		for l in range(maxlevel, 0, -1):
			fig, ax1 = plt.subplots(figsize=gv.canvas_size)
			# ax1 = plt.gca()
			# plt.axvline(x=0.06, linestyle="--")
			logerrs = np.zeros((len(dbses), dbses[0].channels), dtype=np.double)
			# phyerrs = np.zeros((len(dbses), dbses[0].channels), dtype=np.double)
			settings = [{} for __ in range(ndb)]
			for d in range(ndb):
				logerrs[d, :] = np.load(fn.LogicalErrorRates(dbses[d], logmet))[:, l]
			include_RC = np.nonzero(logerrs[0, :] > thresholds["y"])[0]
			include_nonRC = np.nonzero(logerrs[1, :] > thresholds["y"])[0]
			include_both = np.intersect1d(include_RC, include_nonRC)
			if include_input is None:
				include = np.random.choice(include_both, min(select_count, include_both.shape[0]))
			else:
				include = np.intersect1d(include_input, include_both)
			print("l: {} and include = {}".format(l, include))
			if logmet == "infid":
				ylabel = "$\\overline{%s_{%d}}$" % (ml.Metrics[logmet]["latex"].replace("$",""), l)
			else:
				ylabel = ml.Metrics[logmet]["latex"].replace("\\mathcal{E}", "\\overline{\\mathcal{E}}_{%d}" % l)
			for d in range(ndb):
				ax1.plot(
					[],
					[],
					marker=gv.Markers[d % gv.n_Markers],
					color="k",
					label=dbses[d].plotsettings["name"],
					markersize=gv.marker_size,
				)
				settings[d] = {
					"xaxis": None,
					"xlabel": None,
					"yaxis": np.load(fn.LogicalErrorRates(dbses[d], logmet))[:, l],
					"ylabel": ylabel,
					"color": "",
					"marker": "",
					"linestyle": "",
				}
				LoadPhysicalErrorRates(dbses[d], phymet, settings[d], d == 0)

				for i, ch in enumerate(include):
					ax1.plot(
						settings[d]["xaxis"][ch],
						settings[d]["yaxis"][ch],
						color=gv.Colors[i % gv.n_Colors],
						marker=gv.Markers[d % gv.n_Markers],
						markersize=2 * gv.marker_size,
					)
					if annotations is not None:
						ax1.annotate(
							annotations[ch],
							(1.05 * settings[d]["xaxis"][ch], settings[d]["yaxis"][ch]),
							color=gv.Colors[i % gv.n_Colors],
							fontsize=gv.ticks_fontsize,
						)
			for i in include:
				# Draw lines between the corresponding channels in databases 0 and 1
				ax1.plot(
					[settings[0]["xaxis"][i], settings[1]["xaxis"][i]],
					[settings[0]["yaxis"][i], settings[1]["yaxis"][i]],
					color="slategrey",
					linestyle="--",
				)
			# Plot the relative improvements in an inset plot
			RelativeImprovement(
				settings[1]["xaxis"],
				np.concatenate(
					(
						settings[0]["yaxis"][np.newaxis, :],
						settings[1]["yaxis"][np.newaxis, :],
					),
					axis=0,
				),
				plt,
				ax1,
				settings[1]["xlabel"],
				include,
				annotations,
			)

			# Principal axes labels
			ax1.set_xlabel(settings[d]["xlabel"], fontsize=gv.axes_labels_fontsize)
			ax1.set_xscale("log")
			ax1.set_ylabel(settings[d]["ylabel"], fontsize=gv.axes_labels_fontsize)
			ax1.set_yscale("log")
			# ax1.set_ylim([10e-9, None])
			ax1.tick_params(
				axis="both",
				which="both",
				pad=gv.ticks_pad,
				direction="inout",
				length=gv.ticks_length,
				width=gv.ticks_width,
				labelsize=gv.ticks_fontsize,
			)
			# Legend
			ax1.legend(
				numpoints=1,
				loc=4,
				shadow=True,
				fontsize=gv.legend_fontsize * 0.75,
				markerscale=gv.legend_marker_scale,
			)
			# Save the plot
			pdf.savefig(fig)
			plt.close()
	return None
