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

# Optional packages
try:
	import PyPDF2 as pp
except ImportError:
	pass

# Functions from other modules
from define import metrics as ml
from define import globalvars as gv
from analyze.load import LoadPhysicalErrorRates
from define.fnames import ChannelWise, LogicalErrorRates


def RelativeImprovement(xaxis, yaxes, plt, ax1, xlabel, only_points, l, annotations=None):
	"""
	Plot relative improvement from RC, in an inset plot.
	We will compute the difference: (second row - first row)/(second row)
	The first row in yaxes refers to RC data while the second row refers to non RC data.
	https://scipython.com/blog/inset-plots-in-matplotlib/
	"""
	atol = 10e-8
	tol = 0.1
	degrading_indices = (yaxes[0, :] - yaxes[1, :]) > atol
	print(
		"Logical error rates:\n RC: {}\n no RC: {}".format(
			yaxes[0, degrading_indices], yaxes[1, degrading_indices]
		)
	)
	if ax1 is not None:
		ax2 = plt.axes([0, 0, 1, 1])
		# Manually set the position and relative size of the inset axes within ax1
		ip = InsetPosition(ax1, [0.1, 0.67, 0.33, 0.3])
		ax2.set_axes_locator(ip)
		# Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
		mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none")
		axes_label_fontsize = gv.axes_labels_fontsize/2
	else:
		ax2 = plt.axes()
		axes_label_fontsize = gv.axes_labels_fontsize

	unsettled_indices, = np.nonzero(np.abs(yaxes[1, only_points] - yaxes[0, only_points]) / yaxes[1, only_points] < tol)
	improvement_indices, = np.nonzero(yaxes[1, only_points] / yaxes[0, only_points] > 1)
	degradation_indices, = np.nonzero(yaxes[1, only_points] / yaxes[0, only_points] < 1)
	
	unsettled = only_points[unsettled_indices]
	improvements = only_points[np.setdiff1d(improvement_indices, unsettled_indices)]
	degradations = only_points[np.setdiff1d(degradation_indices, unsettled_indices)]

	# print("only_points\n{}Non RC\n{}\nRC\n{}\nimprovements\n{}\ndegradations\n{}\nunsettled\n{}".format(only_points, yaxes[1, only_points], yaxes[0, only_points], improvements, degradations, unsettled))

	if (improvements.size > 0):
		ax2.plot(xaxis[improvements], yaxes[1, improvements]/yaxes[0, improvements], color=gv.QB_GREEN, marker="o", alpha = 0.8, markersize=gv.marker_size, linestyle = "None")
	if (degradations.size > 0):
		ax2.plot(xaxis[degradations], yaxes[1, degradations]/yaxes[0, degradations], color="red", marker="o", alpha = 0.8, markersize=gv.marker_size, linestyle = "None")
	if (unsettled.size > 0):
		ax2.plot(xaxis[unsettled], yaxes[1, unsettled]/yaxes[0, unsettled], color="0.5", marker="o", alpha = 0.8, markersize=gv.marker_size, linestyle = "None")

	# for i in only_points:
	# 	if np.abs((yaxes[1, i] - yaxes[0, i])/yaxes[1, i]) > tol:
	# 		if yaxes[1, i]/yaxes[0, i] < 1 :
	# 			color = "red"
	# 		else:
	# 			color = gv.QB_GREEN
	# 	else:
	# 		color = "0.5"
	# 	ax2.plot(
	# 		xaxis[i],
	# 		yaxes[1, i]/yaxes[0, i],
	# 		color=color,
	# 		marker="o",
	# 		alpha = 0.8,
	# 		markersize=gv.marker_size,
	# 	)
	# 	if annotations is not None:
	# 		ax2.annotate(
	# 			annotations[i],
	# 			(0.92 * xaxis[i], 0.89 * (yaxes[1, i] - yaxes[0, i]) / yaxes[1, i]),
	# 			color=gv.Colors[i % gv.n_Colors],
	# 			fontsize=gv.ticks_fontsize,
	# 		)
	# Draw a horizontal line at Y=0 to show the break-even point RC and no RC.
	ax2.axhline(y=1, linestyle="--")
	ax2.set_xlabel(xlabel, fontsize=axes_label_fontsize )
	ax2.set_ylabel("$\\delta_{%d}$" %(l), fontsize=axes_label_fontsize)
	ax2.set_xscale("log")
	ax2.set_yscale("log")
	ax2.tick_params(
		axis="both",
		which="both",
		pad=gv.ticks_pad,
		direction="inout",
		length=gv.ticks_length,
		width=gv.ticks_width,
		labelsize=gv.ticks_fontsize,
	)
	ax2.plot(
		[],
		[],
		marker="o",
		color=gv.QB_GREEN,
		label="$\\delta_{%d} \\geq 1$"%(l),
		markersize=gv.marker_size,
		linestyle = "None"
	)
	ax2.plot(
		[],
		[],
		marker="o",
		color="red",
		label="$\\delta_{%d}<1$"%(l),
		markersize=gv.marker_size,
		linestyle = "None"
	)
	ax2.legend(
		numpoints=1,
		loc="best",
		shadow=True,
		fontsize=gv.legend_fontsize*1.5,
		markerscale=gv.legend_marker_scale,
	)
	return None


def ChannelWisePlot(phymet, logmet, dbses, thresholds={"y": 10e-16, "x": 10e-16}, include_input=None, only_inset = False):
	# Plot each channel in the database with a different color.
	# Channels of similar type in different databases will be distinguished using different markers.
	only_inset = True
	ndb = len(dbses)
	plotfname = ChannelWise(dbses[0], phymet, logmet)
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
				logerrs[d, :] = np.load(LogicalErrorRates(dbses[d], logmet))[:, l]
			include_RC = np.nonzero(logerrs[0, :] > thresholds["y"])[0]
			include_nonRC = np.nonzero(logerrs[1, :] > thresholds["y"])[0]
			include_both = np.intersect1d(include_RC, include_nonRC)
			if include_input is None:
				# include = np.random.choice(include_both, min(select_count, include_both.shape[0]))
				include = include_both
			else:
				include = np.intersect1d(include_input, include_both)
			print("l: {} and include = {}".format(l, include))
			if logmet == "infid":
				ylabel = "$\\overline{%s_{%d}}$" % (ml.Metrics[logmet]["latex"].replace("$",""), l)
			else:
				ylabel = ml.Metrics[logmet]["latex"].replace("\\mathcal{E}", "\\overline{\\mathcal{E}}_{%d}" % l)
			for d in range(ndb):
				if only_inset == False:
					ax1.plot(
						[],
						[],
						marker=gv.Markers[d % gv.n_Markers],
						color="k",
						label=dbses[d].plotsettings["name"],
						markersize=gv.marker_size/10,
					)
				settings[d] = {
					"xaxis": None,
					"xlabel": None,
					"yaxis": np.load(LogicalErrorRates(dbses[d], logmet))[:, l],
					"ylabel": ylabel,
					"color": "",
					"marker": "",
					"linestyle": "",
				}
				LoadPhysicalErrorRates(dbses[d], phymet, settings[d], d == 0)

				if only_inset == False:
					for i, ch in enumerate(include):
						ax1.plot(
							settings[d]["xaxis"][ch],
							settings[d]["yaxis"][ch],
							color=gv.Colors[i % gv.n_Colors],
							marker=gv.Markers[d % gv.n_Markers],
							markersize= gv.marker_size*0.7,
						)
						if annotations is not None:
							ax1.annotate(
								annotations[ch],
								(1.05 * settings[d]["xaxis"][ch], settings[d]["yaxis"][ch]),
								color=gv.Colors[i % gv.n_Colors],
								fontsize=gv.ticks_fontsize,
							)
					# for i in include:
					# 	# Draw lines between the corresponding channels in databases 0 and 1
					# 	ax1.plot(
					# 		[settings[0]["xaxis"][i], settings[1]["xaxis"][i]],
					# 		[settings[0]["yaxis"][i], settings[1]["yaxis"][i]],
					# 		color="slategrey",
					# 		linestyle="--",
					# 	)
			if only_inset == True:
				ax1 = None # Disable main plot axis

			# Code to find worst deviation channel
			yaxes = np.concatenate(
				(
					settings[0]["yaxis"][np.newaxis, :],
					settings[1]["yaxis"][np.newaxis, :],
				),
				axis=0,
			)
			deltas = yaxes[1, include]/yaxes[0, include]
			min_delta_index = include[np.argmin(deltas)]
			print("numerator = {}, denominator = {}".format(yaxes[1, min_delta_index], yaxes[0, min_delta_index]))
			print("Minimum delta for level {}: {}, is achieved for channel {}.".format(l, np.min(deltas), min_delta_index))
			print("Noise rate , sample : {}".format(dbses[d].available[min_delta_index, :]))
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
				l,
				annotations,
			)

			if only_inset == False:
				# Principal axes labels
				ax1.set_xlabel(settings[d]["xlabel"], fontsize=gv.axes_labels_fontsize)
				ax1.set_xscale("log")
				ax1.set_ylabel(settings[d]["ylabel"], fontsize=gv.axes_labels_fontsize)
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
