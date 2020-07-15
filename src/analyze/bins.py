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
	from matplotlib.ticker import LogLocator
	from scipy.interpolate import griddata
	import PyPDF2 as pp
except:
	pass
from define import fnames as fn
from define import globalvars as gv
from define import metrics as ml
from analyze.utils import scientific_float


def BinsPlot(dbs, lmet, pvals):
	# Plot the number of syndromes at each level for a given probability and conditional logical error (infidelity, etc).
	# If no noise rate is specified, one plot for each channel in the database.
	# For each channel, the bins array is formatted as: bins[level, synd prob, metric val].
	npoints = 6
	nchans = 0
	plotfname = fn.SyndromeBinsPlot(dbs, lmet, pvals)
	with PdfPages(plotfname) as pdf:
		for i in range(dbs.channels):
			if pvals == -1:
				cont = 1
			else:
				cont = 0
				if np.all(dbs.available[i, :] == pvals):
					cont = 1
			if cont == 1:
				# print("p = %s" % (np.array_str(dbs.available[i, :])))
				nchans = nchans + 1
				bins = np.load(
					fn.SyndromeBins(
						dbs, dbs.available[i, :-1], dbs.available[i, -1], lmet
					)
				)
				for l in range(1, dbs.levels):
					fig = plt.figure(
						figsize=(gv.canvas_size[0] * 2, gv.canvas_size[1] * 3)
					)
					# print("l = %d\n%d non zero rows\n%s\n%d non zero columns\n%s" % (l + 1, np.count_nonzero(~np.all(bins[1 + l, :, :] == 0, axis=1)), np.array_str(np.nonzero(~np.all(bins[1 + l, :, :] == 0, axis=1))[0]), np.count_nonzero(~np.all(bins[1 + l, :, :] == 0, axis=0)), np.array_str(np.nonzero(~np.all(bins[1 + l, :, :] == 0, axis=0))[0])))
					nzrows = np.nonzero(~np.all(bins[1 + l, :, :] == 0, axis=1))[0]
					nzcols = np.nonzero(~np.all(bins[1 + l, :, :] == 0, axis=0))[0]
					(meshX, meshY) = np.meshgrid(nzrows, nzcols, indexing="ij")
					meshZ = bins[1 + l, nzrows, :][:, nzcols]
					meshZ = meshZ / np.max(meshZ)
					# print("meshX: (%d, %d) \n%s\nmeshY: (%d, %d)\n%s\nmeshZ: (%d, %d)\n%s" % (meshX.shape[0], meshX.shape[1], np.array_str(meshX), meshY.shape[0], meshY.shape[1], np.array_str(meshY), meshZ.shape[0], meshZ.shape[1], np.array_str(meshZ)))
					plt.pcolor(
						meshY,
						meshX,
						meshZ,
						cmap="binary",
						norm=LogNorm(
							vmin=np.min(meshZ[np.nonzero(meshZ)]), vmax=np.max(meshZ)
						),
					)
					# Title
					# plt.title("p = %s, s = %d, l = %d" % (np.array_str(dbs.available[i, :-1]), int(dbs.available[i, -1]), l + 1), fontsize = gv.title_fontsize, y = 1.03)
					# Axes labels
					ax = plt.gca()
					ax.set_ylabel(
						"$- $log${}_{10}\\mathsf{Pr}(s)$",
						fontsize=gv.axes_labels_fontsize + 144,
					)
					ax.set_ylim([0, 50])
					# yticks = nzrows[np.linspace(0, nzrows.shape[0] - 1, npoints, dtype = np.int)]
					# ax.set_yticks(yticks)
					# ax.set_yticklabels(["%d" % (tc) for tc in yticks])
					ax.set_xlabel(
						"$- $log${}_{10}\\mathcal{N}(\\mathcal{E}^{\\thickspace s}_{%d})$"
						% (l + 1),
						fontsize=gv.axes_labels_fontsize + 144,
					)
					# xticks = nzcols[np.linspace(0, nzcols.shape[0] - 1, npoints, dtype = np.int)]
					# ax.set_xticks(xticks)
					# ax.set_xticklabels(["%d" % (tc) for tc in xticks])
					ax.tick_params(
						axis="both",
						which="both",
						pad=gv.ticks_pad + 50,
						direction="inout",
						length=gv.ticks_length,
						width=gv.ticks_width,
						labelsize=gv.ticks_fontsize + 120,
					)
					# Legend
					cbar = plt.colorbar(spacing="proportional", drawedges=False)
					# cbar.ax.set_xlabel("$\\mathcal{T}_{%d}$" % (l + 1), fontsize = gv.colorbar_fontsize)
					cbar.ax.tick_params(
						labelsize=gv.legend_fontsize + 120,
						pad=gv.ticks_pad,
						length=gv.ticks_length,
						width=gv.ticks_width,
					)
					cbar.ax.xaxis.labelpad = gv.ticks_pad
					# Save the plot
					pdf.savefig(fig)
					plt.close()
		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Syndrome bins for %d channels and %d levels." % (
			nchans,
			dbs.levels,
		)
		pdfInfo["Author"] = "Pavithran Iyer"
		pdfInfo["ModDate"] = dt.datetime.today()
	return None


def PlotBinVarianceMetrics(ax1, dbs, level, lmet, pmets, nbins, include_info):
	# Compare scatter for different physical metrics
	min_bin_fraction = 0.1

	ax2 = plt.axes([0, 0, 1, 1])
	# Manually set the position and relative size of the inset axes within ax1
	# 0.1, 0.65, 0.33, 0.3
	# 0.6, 0.25, 0.33, 0.3
	ip = InsetPosition(ax1, [0.1, 0.6, 0.33, 0.3])
	ax2.set_axes_locator(ip)
	# Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
	mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none")
	phyerrs = np.zeros((len(pmets), dbs.channels), dtype=np.double)
	logerrs = np.load(fn.LogicalErrorRates(dbs, lmet))[:, level]
	plotfname = fn.CompareScatters(dbs, lmet, pmets, mode="metrics")
	collapsed_bins = {mt: None for mt in pmets}
	fig = plt.figure(figsize=gv.canvas_size)
	for p in range(len(pmets)):
		if pmets[p] == "uncorr":
			phyerrs[p, :] = np.load(fn.PhysicalErrorRates(dbs, pmets[p]))[:, level]
		else:
			phyerrs[p, :] = np.load(fn.PhysicalErrorRates(dbs, pmets[p]))

		# print("Computing bins for {} at level {}".format(pmets[p], level))

		include = include_info[pmets[p]]

		bins = ComputeBinVariance(phyerrs[p, include], logerrs[include], nbins=nbins)
		collapsed_bins[pmets[p]] = CollapseBins(
			bins, min_bin_fraction * dbs.channels / nbins
		)

		# print(
		#     "\033[2mnbins for level {}, metric {} = {}\nPoints in bins = {}\nAverage points in a bin = {}\nthreshold: {}\n----\033[0m".format(
		#         level,
		#         pmets[p],
		#         collapsed_bins[pmets[p]].shape[0],
		#         collapsed_bins[pmets[p]][:, 2],
		#         np.mean(bins[:, 2]),
		#         min_bin_fraction * dbs.channels / nbins,
		#     )
		# )

		xaxis = np.arange(collapsed_bins[pmets[p]].shape[0])
		# xaxis = (bins[pmets[p]][:, 0] + bins[pmets[p]][:, 1]) / 2
		yaxis = collapsed_bins[pmets[p]][:, 3]
		# print("yaxis = {}".format(yaxis))
		ax2.plot(
			xaxis,
			yaxis,
			marker=ml.Metrics[pmets[p]]["marker"],
			color=ml.Metrics[pmets[p]]["color"],
			linestyle="-",
			linewidth=gv.line_width,
			markersize=gv.marker_size,
			alpha=0.75,
			label=ml.Metrics[pmets[p]]["latex"],
		)
	# Axes
	# ax2.set_xlabel(ml.Metrics[pmets[0]]["phys"], fontsize=gv.axes_labels_fontsize * 0.6)
	# ax2.set_xlabel("Bins", fontsize=gv.axes_labels_fontsize * 0.6)
	# ax.set_xscale("log")
	ax2.set_ylabel("$\\Delta$", fontsize=gv.axes_labels_fontsize)
	# ax.set_ylim([10e-9, None])
	# ax2.set_yscale("log")
	# Axes ticks
	# loc = LogLocator(base=10, numticks=10) # this locator puts ticks at regular intervals
	# ax2.yaxis.set_major_locator(loc)
	# print("nonzero_bins.size = {}".format(nonzero_bins.size))
	ax2.set_xticks(np.arange(collapsed_bins[pmets[0]].shape[0], dtype=np.int))
	ax2.set_xticklabels(
		list(
			map(
				lambda num: "%s" % scientific_float(num),
				(collapsed_bins[pmets[0]][:, 0] + collapsed_bins[pmets[0]][:, 1]) / 2,
			)
		),
		rotation=45,
		color=ml.Metrics[pmets[0]]["color"],
	)
	ax2.tick_params(
		axis="both",
		which="both",
		pad=gv.ticks_pad,
		direction="inout",
		length=gv.ticks_length,
		width=gv.ticks_width,
		labelsize=gv.ticks_fontsize * 0.75,
	)
	if len(pmets) > 1:
		ax2T = ax2.twiny()
		ax2T.set_axes_locator(ip)
		ax2T.tick_params(
			axis="both",
			which="both",
			# pad=gv.ticks_pad,
			direction="inout",
			length=gv.ticks_length,
			width=gv.ticks_width,
			labelsize=gv.ticks_fontsize * 0.75,
		)
		ax2T.set_xticks(np.arange(collapsed_bins[pmets[1]].shape[0], dtype=np.int))
		ax2T.set_xticklabels(
			list(
				map(
					lambda num: "%s" % scientific_float(num),
					(collapsed_bins[pmets[1]][:, 0] + collapsed_bins[pmets[1]][:, 1])
					/ 2,
				)
			),
			rotation=45,
			color=ml.Metrics[pmets[1]]["color"],
		)
	return None


def PlotBinVarianceDataSets(ax1, dbses, level, lmet, phymets, nbins, include_info):
	# Compare scatter for different physical metrics
	min_bin_fraction = 0.1
	ax2 = plt.axes([0, 0, 1, 1])
	# Manually set the position and relative size of the inset axes within ax1
	ip = InsetPosition(ax1, [0.1, 0.6, 0.33, 0.3])
	# if len(phymets) > 1:
	#     ip = InsetPosition(ax1, [0.1, 0.6, 0.33, 0.3])
	# else:
	#     ip = InsetPosition(ax1, [0.1, 0.65, 0.33, 0.3])
	ax2.set_axes_locator(ip)
	# Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
	mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none")
	# Broadcast the physical error metric if only one is given.
	if len(phymets) == 1:
		pmets = [phymets[0] for __ in range(len(dbses))]
	else:
		pmets = phymets
	ndb = len(dbses)
	phyerrs = np.zeros((ndb, dbses[0].channels), dtype=np.double)
	logerrs = np.zeros((ndb, dbses[0].channels), dtype=np.double)
	collapsed_bins = [None for d in range(ndb)]
	for d in range(ndb):
		if pmets[d] == "uncorr":
			phyerrs[d, :] = np.load(fn.PhysicalErrorRates(dbses[d], pmets[d]))[:, level]
		else:
			phyerrs[d, :] = np.load(fn.PhysicalErrorRates(dbses[d], pmets[d]))
		logerrs[d, :] = np.load(fn.LogicalErrorRates(dbses[d], lmet))[:, level]

		include = include_info[pmets[d]]

		bins = ComputeBinVariance(
			phyerrs[d, include], logerrs[d, include], space="log", nbins=nbins
		)
		collapsed_bins[d] = CollapseBins(
			bins, min_bin_fraction * dbses[d].channels / nbins
		)

		# print(
		#     "\033[2mnbins for level {} = {}\nPoints in bins = {}\nAverage points in a bin = {}\nthreshold: {}\n----\033[0m".format(
		#         level,
		#         collapsed_bins[d].shape[0],
		#         collapsed_bins[d][:, 2],
		#         np.mean(bins[:, 2]),
		#         min_bin_fraction * dbses[d].channels / nbins,
		#     )
		# )

		xaxis = np.arange(collapsed_bins[d].shape[0])
		# xaxis = (bins[pmets[p]][:, 0] + bins[pmets[p]][:, 1]) / 2
		yaxis = collapsed_bins[d][:, 3]
		# print("Bin plot for (d={}) {}: {}".format(d, names[d], gv.Colors[d]))
		if (d == 0):
			ax2T = ax2.twiny()
			ax2T.set_axes_locator(ip)
			ax2T.plot(
			xaxis,
			yaxis,
			marker=gv.Markers[d % gv.n_Markers],
			color=gv.Colors[d],
			linestyle="-",
			linewidth=gv.line_width,
			markersize=gv.marker_size,
			alpha=0.75)
		else:
			ax2.plot(
				xaxis,
				yaxis,
				marker=gv.Markers[d % gv.n_Markers],
				color=gv.Colors[d],
				linestyle="-",
				linewidth=gv.line_width,
				markersize=gv.marker_size,
				alpha=0.75,
			)
		# print("dbs name: {}".format(dbses[d].name))
	# Axes
	# ax2.set_xlabel("Bins", fontsize=gv.axes_labels_fontsize * 0.6)
	ax2.set_ylabel("$\\Delta$", fontsize=gv.axes_labels_fontsize)
	# ax.set_ylim([10e-9, None])
	ax2.set_yscale("log")
	ax2.tick_params(
		axis="both",
		which="both",
		# pad=gv.ticks_pad,
		direction="inout",
		length=gv.ticks_length,
		width=gv.ticks_width,
		labelsize=gv.ticks_fontsize * 0.75,
	)
	ax2.set_xticks(np.arange(0, collapsed_bins[1].shape[0], dtype=np.int))
	ax2.set_xticklabels(
		list(
			map(
				lambda num: "%s" % scientific_float(num),
				(collapsed_bins[1][:, 0] + collapsed_bins[1][:, 1]) / 2,
			)
		),
		rotation=45,
		color=gv.Colors[1],
	)
	# if len(phymets) > 1:
	ax2T.tick_params(
		axis="both",
		which="both",
		# pad=gv.ticks_pad,
		direction="inout",
		length=gv.ticks_length,
		width=gv.ticks_width,
		labelsize=gv.ticks_fontsize * 0.75,
	)
	ax2T.set_xticks(np.arange(0, collapsed_bins[0].shape[0], dtype=np.int))
	ax2T.set_xticklabels(
		list(
			map(
				lambda num: "%s" % scientific_float(num),
				(collapsed_bins[0][:, 0] + collapsed_bins[0][:, 1]) / 2,
			)
		),
		rotation=45,
		color=gv.Colors[0],
	)
	return None


def CollapseBins(bins, min_bin_size):
	"""
	Merge bins into one if either have less than a threshold number of points.
	"""
	# return bins
	collapse = np.zeros(bins.shape[0], dtype=np.int)
	i = 0
	while i < bins.shape[0] - 1:
		if bins[i, 2] < min_bin_size:
			collapse[i] = 1
			i += 1
		i += 1

	if np.sum(collapse) == 0:
		if bins[-1, 2] < min_bin_size:
			bins[-2, 1] = bins[-1, 1]
			bins[-2, 2] += bins[-1, 2]
			bins[-2, 3] = max(bins[-2, 5], bins[-1, 5]) / (
				min(bins[-2, 4], bins[-1, 4]) * bins[-2, 2]
			)
			bins[-2, 4] = min(bins[-2, 4], bins[-1, 4])
			bins[-2, 5] = max(bins[-2, 5], bins[-1, 5])
			# Scale all the scatter metrics by the average number of points in a bin.
			average_population = np.mean(bins[:, 2])
			bins[:, 3] = bins[:, 3] * average_population
			return bins[:-1, :]
		# Scale all the scatter metrics by the average number of points in a bin.
		average_population = np.mean(bins[:, 2])
		bins[:, 3] = bins[:, 3] * average_population
		return bins

	collapsed_bins = np.zeros((bins.shape[0] - np.sum(collapse), 6), dtype=np.double)
	i = 0
	j = 0
	while i < collapse.shape[0]:
		if collapse[i] == 0:
			collapsed_bins[j, :] = bins[i, :]
		else:
			collapsed_bins[j, 0] = bins[i, 0]
			collapsed_bins[j, 1] = bins[i + 1, 1]
			collapsed_bins[j, 2] = bins[i, 2] + bins[i + 1, 2]
			collapsed_bins[j, 3] = max(bins[i, 5], bins[i + 1, 5]) / (
				min(bins[i, 4], bins[i + 1, 4]) * collapsed_bins[j, 2]
			)
			collapsed_bins[j, 4] = min(bins[i, 4], bins[i + 1, 4])
			collapsed_bins[j, 5] = max(bins[i, 5], bins[i + 1, 5])
			i += 1
		j += 1
		i += 1
	# Scale all the scatter metrics by the average number of points in a bin.
	# average_population = np.mean(collapsed_bins[:, 2])
	# collapsed_bins[:, 3] = collapsed_bins[:, 3] * average_population
	return CollapseBins(collapsed_bins, min_bin_size)


def GetXCutOff(xdata, ydata, ythreshold, nbins=10, space="log"):
	# Get the X-value for which all the points have their Y-value to be at least the threshold.
	atol = 10e-12

	# print(
	#     "xdata: {} to {}".format(
	#         np.min(np.where(xdata <= atol, atol, xdata)), np.max(xdata)
	#     )
	# )

	bins = np.zeros((nbins - 1, 4), dtype=np.longdouble)
	if space == "log":
		window = np.logspace(
			np.log10(np.min(np.where(xdata <= atol, atol, xdata))),
			np.log10(np.max(xdata)),
			num=nbins,
			base=10,
		)
	else:
		window = np.linspace(np.min(xdata), np.max(xdata), nbins)

	# print("windows = {}".format(window))

	bins[:, 0] = window[:-1]
	bins[:, 1] = window[1:]
	xcutoff = {"left": np.min(xdata), "right": np.max(xdata)}
	found_left = 0
	found_right = 0
	for i in range(nbins - 1):
		points = np.nonzero(
			np.logical_and(
				np.logical_and(
					np.logical_and(xdata >= bins[i, 0], xdata < bins[i, 1]),
					ydata > atol,
				),
				np.logical_not(np.isnan(ydata)),
			)
		)[0]
		bins[i, 2] = np.double(points.shape[0])
		# Variance of the Y axis points in the bin
		# print(
		#     "bin {}: [{}, {}]\n{} points\n{}\nydata\n{}".format(
		#         i + 1, bins[i, 0], bins[i, 1], points.shape, points, ydata[points]
		#     )
		# )
		# print(
		#     "bin {}: [{}, {}]\n{} points".format(
		#         i + 1, bins[i, 0], bins[i, 1], points.shape
		#     )
		# )
		if len(points) > 10:
			if found_left == 0:
				if np.min(ydata[points]) >= ythreshold["lower"]:
					xcutoff["left"] = bins[i, 0]
					found_left = 1
			if found_right == 0:
				if np.max(ydata[points]) >= ythreshold["upper"]:
					xcutoff["right"] = bins[i, 0]
					found_right = 1
	# We want the bin index for which the minimum Y is greater than the threshold.
	return xcutoff


def ComputeBinVariance(xdata, ydata, nbins=10, space="log", binfile=None, submit=None):
	# Compute the amount of scater of data in a plot.
	# Divide the X axis range into bins and compute the variance of Y-data in each bin.
	# The bins must divide the axes on a linear scale -- because they signify the confidence interval in measuring the values of the parameters.
	# The variance of a dataset of values {x1, x2, ..., xn} is just \sum_i (xi - xmean)^2.
	# the Output is formatted as:
	# 	bins: 2D array with N rows and 4 columns
	# 			Each row represents a channel
	# 			bins[i] = [low, high, npoints, var]
	# 			where low and high are the physical error rates that specify the bin.
	# 			npoints is the number of physical error rates in the bin
	# 			var is the variance of logical error rates in the bin.

	atol = 10e-12
	bins = np.zeros((nbins - 1, 7), dtype=np.longdouble)
	if space == "log":
		window = np.logspace(
			np.log10(np.min(xdata)), np.log10(np.max(xdata)), num=nbins, base=10
		)
		# window = np.power(
		#     base,
		#     np.linspace(
		#         np.log10(np.max(xdata)) / np.log10(base),
		#         np.log10(np.min(xdata)) / np.log10(base),
		#         nbins,
		#     ),
		# )[::-1]
	else:
		window = np.linspace(np.min(xdata), np.max(xdata), nbins)
	bins[:, 0] = window[:-1]
	bins[:, 1] = window[1:]
	# print("xdata\n{} to {}".format(np.min(xdata), np.max(xdata)))
	# print("bins\n{}".format(bins[:, :2]))

	if binfile is not None:
		bf = open(binfile, "w")
		bf.write(
			"# index from to npoints var min chan nrate samp max chan nrate samp\n"
		)
		representatives = []
	for i in range(nbins - 1):
		points = np.nonzero(
			np.logical_and(
				np.logical_and(
					np.logical_and(xdata >= bins[i, 0], xdata < bins[i, 1]),
					ydata > atol,
				),
				np.logical_not(np.isnan(ydata)),
			)
		)[0]
		bins[i, 2] = np.double(points.shape[0])
		if len(points) > 0:
			# print(
			#     "{}: ydata max = {}, ydata min = {}".format(
			#         i, np.max(ydata[points]), np.min(ydata[points])
			#     )
			# )
			bins[i, 4] = np.min(ydata[points])
			bins[i, 5] = np.max(ydata[points])
			bins[i, 3] = bins[i, 5] / (bins[i, 4] * bins[i, 2])
			bins[i, 6] = np.mean(ydata[points])
			# bins[i, 4] = np.mean(-np.log10(ydata[points]))
			# bins[i, 5] = np.std(-np.log10(ydata[points]))
			# bins[i, 3] = bins[i, 5]

	# average_population = np.mean(bins[:, 2])
	# print("Average population: {}".format(average_population))
	# Scale all the scatter metrics by the average number of points in a bin.
	# bins[:, 3] = bins[:, 3] * average_population

	for i in range(nbins - 1):
		# print(
		#     "bin %d: %d points\n\t[%g, %g] -- max = %g and min = %g, U = %g, D = %g."
		#     % (
		#         i,
		#         bins[i, 2],
		#         bins[i, 0],
		#         bins[i, 1],
		#         np.max(ydata[points]) if len(points) > 0 else 0,
		#         np.min(ydata[points]) if len(points) > 0 else 0,
		#         np.mean(ydata[points]) if len(points) > 0 else 0,
		#         bins[i, 3],
		#     )
		# )
		if binfile is not None:
			minchan = points[np.argmin(ydata[points])]
			maxchan = points[np.argmax(ydata[points])]
			# print(
			#     "points = {}, minchan = {}, maxchan = {}".format(
			#         points, minchan, maxchan
			#     )
			# )
			bf.write(
				"%d %g %g %d %g %g %d %s %d %g %d %s %d\n"
				% (
					i,
					bins[i, 0],
					bins[i, 1],
					bins[i, 2],
					bins[i, 3],
					np.min(ydata[points]),
					minchan,
					" ".join(
						list(
							map(lambda num: "%g" % num, submit.available[minchan, :-1])
						)
					),
					submit.available[minchan, -1],
					np.max(ydata[points]),
					maxchan,
					" ".join(
						list(
							map(lambda num: "%g" % num, submit.available[maxchan, :-1])
						)
					),
					submit.available[maxchan, -1],
				)
			)
			representatives.append(
				[
					minchan,
					" ".join(
						list(
							map(lambda num: "%g" % num, submit.available[minchan, :-1])
						)
					),
					submit.available[minchan, -1],
				]
			)
			representatives.append(
				[
					maxchan,
					" ".join(
						list(
							map(lambda num: "%g" % num, submit.available[maxchan, :-1])
						)
					),
					submit.available[maxchan, -1],
				]
			)
	print(
		"\033[2mTotal: %d points and average scatter = %g and maximum scatter = %g.\033[0m"
		% (np.sum(bins[:, 2], dtype=int), np.mean(bins[:, 3]), np.max(bins[:, 3]))
	)
	if binfile is not None:
		bf.write("\n\n")
		bf.write("# Representatives from each bin\n")
		bf.write("# chan nrate samp\n")
		for i in range(len(representatives)):
			bf.write(
				"%d %s %d\n"
				% (representatives[i][0], representatives[i][1], representatives[i][2])
			)
		bf.close()
	# exit()
	return bins


def ComputeNDimBinVariance(xdata, ydata, nbins=3, space="linear"):
	# Divide a N-dimensional space into bins and classify xdata points into bins.
	base = 10.0
	ndim = xdata.shape[1]
	# Divide each axis of the d-dimensional space into intervals
	window = np.zeros((ndim, nbins), dtype=np.longdouble)
	for i in range(ndim):
		if space == "log":
			window[i, :] = np.power(
				base,
				np.linspace(
					np.log10(np.max(xdata[:, i])) / np.log10(base),
					np.log10(np.min(xdata[:, i])) / np.log10(base),
					nbins,
				),
			)[::-1]
		else:
			window[i, :] = np.linspace(np.min(xdata[:, i]), np.max(xdata[:, i]), nbins)

	# print("xdata\n%s" % (np.array_str(xdata)))
	# print("window\n%s" % (np.array_str(window)))

	# For every point in xdata, determine its address in terms of windows, in the n-dim space.
	address = np.zeros((xdata.shape[0], ndim), dtype=np.int)
	binindex = np.zeros(xdata.shape[0], dtype=np.int)
	for i in range(xdata.shape[0]):
		for j in range(ndim):
			# which window does xdata[i, j] fall into ?
			# print("xdata[i, j] = %g, window[j, :] = %s" % (xdata[i, j], np.array_str(window[j, :])))
			# print "np.logical_and(xdata[i, j] >= window[j, :-1], xdata[i, j] < window[j, 1:])"
			# print np.logical_and(xdata[i, j] >= window[j, :-1], xdata[i, j] < window[j, 1:])
			# address[i, j] = np.nonzero(np.logical_and(xdata[i, j] >= window[j, :-1], xdata[i, j] < window[j, 1:]).astype(np.int))[0][0]
			for k in range(nbins):
				if xdata[i, j] > window[j, k]:
					address[i, j] = k

		# Interprett the address as an encoding of the bin index in base-b alphabet, where b is the number of bins in any axis.
		binindex[i] = np.sum(
			np.multiply(
				address[i, :], np.power(nbins - 1, np.linspace(ndim - 1, 0, ndim))
			),
			dtype=np.int,
		)

		# print("address[i, :]\n%s\nnp.power(nbins - 1, np.linspace(ndim - 1, 0, ndim))\n%s" % (np.array_str(address[i, :]), np.array_str(np.power(nbins - 1, np.linspace(ndim - 1, 0, ndim)))))

	# print("address\n%s" % (np.array_str(address)))
	# print("binindex\n%s" % (np.array_str(binindex)))

	# Count the number of xdata points with a fixed bin index and record that information in bins.
	bins = np.zeros((np.power(nbins - 1, ndim, dtype=np.int), 4), dtype=np.longdouble)
	isempty = np.zeros(bins.shape[0], dtype=np.int)
	for i in range(bins.shape[0]):
		if np.count_nonzero(binindex == i) == 0:
			isempty[i] = 1
		else:
			points = np.nonzero(binindex == i)[0]
			bins[i, 2] = points.shape[0]
			bins[i, 3] = np.var(ydata[points])
			# print("Bin %d: points = %s\nydata = %s\nvariance = %g and mean = %g." % (i, np.array_str(points), np.array_str(ydata[points]), bins[i, 3], np.mean(ydata[points])))

	# print("Bins\n%s" % (np.array_str(bins[:, 2:])))

	print(
		"\033[2mTotal: %d points and average variance = %g and maximum variance = %g.\033[0m"
		% (np.sum(bins[:, 2], dtype=int), np.mean(bins[:, 3]), np.max(bins[:, 3]))
	)
	return bins


def AddBinVariancePlot(
	bins, level, lmet, pmet, pmetname, pdf=None, plotfname="unknown"
):
	# Plot the variance in each bin with respect to the bin along with producing a table of those values.
	# If a PdfPages object is specified, the plot is simply added to the PDF.
	# Else, it is a separate plot. In this case, the name of the file to which the plot must be stored, must be specified.
	if sub.IsNumber(pmet) == 1:
		color = ml.Metrics[
			list(ml.Metrics.keys())[len(list(ml.Metrics.keys())) % (1 + pmet)]
		]["color"]
	else:
		color = ml.Metrics[pmet]["color"]
	if pdf is None:
		pdfobj = PdfPages(plotfname)
	fig = plt.figure(figsize=((gv.canvas_size[0] * 1.5, gv.canvas_size[1] * 1.5)))
	plt.title(
		"level %d: Average scatter = %g, Maximum scatter = %g"
		% (level, np.mean(bins[:, 3]), np.max(bins[:, 3])),
		fontsize=gv.title_fontsize,
		y=1.03,
	)
	# print("widths\n%s" % (np.array_str(bins[:, 1] - bins[:, 0])))
	barplot = plt.bar(
		bins[:, 0],
		bins[:, 3],
		width=bins[:, 1] - bins[:, 0],
		bottom=0,
		align="edge",
		color=color,
		linewidth=0,
	)
	# Axes
	ax = plt.gca()
	ax.set_xlabel(
		"$-\\log10(\\mathcal{N}_{0}: %s)$" % (pmetname.replace("$", "")),
		fontsize=gv.axes_labels_fontsize,
	)
	# ax.set_xscale('log')
	ax.set_ylabel(
		(
			"Variance in $\\log\\mathcal{N}_{%d}$  $\\left(%s\\right)$"
			% (level, ml.Metrics[lmet]["latex"].replace("$", ""))
		),
		fontsize=gv.axes_labels_fontsize,
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
	# Attach a text label above each bar, indicating the numerical value of the variance.
	for rect in barplot:
		(height, width) = (rect.get_height(), rect.get_width())
		# print("height = %g, width = %d" % (height, width))
		ax.text(
			rect.get_x() + width / float(2),
			1.01 * height,
			"%g" % (height),
			ha="center",
			va="bottom",
			fontsize=gv.ticks_fontsize,
		)
	if pdf is None:
		# Save the plot
		pdfobj.savefig(fig)
		plt.close()
		# Set PDF attributes
		pdfInfo = pdfobj.infodict()
		pdfInfo["Title"] = "Binwise variance of %s with respect to %s." % (
			lmet,
			str(pmet),
		)
		pdfInfo["Author"] = "Pavithran Iyer"
		pdfInfo["ModDate"] = dt.datetime.today()
		pdfobj.close()
	else:
		# Save the plot
		pdf.savefig(fig)
		plt.close()
	return None
