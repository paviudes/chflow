# Critical packages
import os
import sys
import datetime as dt
import numpy as np
from scipy.special import comb
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
	from adjustText import adjust_text
	ADJUST = 1
except ImportError:
	ADJUST = 0

# Functions from other modules
from define import metrics as ml
from define import globalvars as gv
from define.decoder import GetTotalErrorBudget, GetLeadingPaulis
from analyze.utils import latex_float, scientific_float, OrderOfMagnitude
from analyze.bins import ComputeBinVariance
from analyze.statplot import IsConverged
from analyze.load import LoadPhysicalErrorRates
from define.fnames import DecodersPlot, DecodersInstancePlot, LogicalErrorRates, PhysicalErrorRates, NRWeightsFile, RawPhysicalChannel, IsConvergedFile

def DecoderCompare(
	phymet, logmet, dbses, nbins=10, thresholds={"y": 10e-16, "x": 10e-16}
):
	# Compare performance of various decoders.
	ndb = len(dbses)
	plotfname = DecodersPlot(dbses[0], phymet, logmet)
	nlevels = max([db.levels for db in dbses])
	with PdfPages(plotfname) as pdf:
		for l in range(1, nlevels + 1):
			fig = plt.figure(figsize=gv.canvas_size)
			ax1 = plt.gca()
			for d in range(ndb):
				settings = {
					"xaxis": None,
					"xlabel": None,
					"yaxis": np.load(LogicalErrorRates(dbses[d], logmet))[:, l],
					"ylabel": "$\\overline{%s_{%d}}$"
					% (ml.Metrics[logmet]["latex"].replace("$", ""), l),
					"color": gv.Colors[d % gv.n_Colors]
					if ndb > 1
					else ml.Metrics[phymet]["color"],
					"marker": gv.Markers[d % gv.n_Markers]
					if ndb > 1
					else ml.Metrics[phymet]["marker"],
					"linestyle": "",
				}
				if dbses[d].decoders[l - 1] == 0:
					decoder_info = "Maximum likelihood"
				elif dbses[d].decoders[l - 1] == 1:
					decoder_info = "Minimum weight"
				else:
					decoder_info = "$\\alpha = %g$" % (dbses[d].decoder_fraction)
				LoadPhysicalErrorRates(dbses[d], phymet, settings, l)
				bins = ComputeBinVariance(
					settings["xaxis"], settings["yaxis"], nbins=nbins, space="log"
				)
				# Plotting
				plotobj = ax1.plot(
					(bins[:, 0] + bins[:, 1]) / 2,
					bins[:, 6],
					color=settings["color"],
					alpha=0.75,
					marker="o",  # settings["marker"]
					markersize=gv.marker_size,
					linestyle=settings["linestyle"],
					linewidth=gv.line_width,
					label=decoder_info,
				)
			# Axes labels
			ax1.set_xlabel(
				settings["xlabel"],
				fontsize=gv.axes_labels_fontsize * 0.8,
				labelpad=gv.axes_labelpad,
			)
			ax1.set_xscale("log")
			ax1.set_ylabel(
				settings["ylabel"],
				fontsize=gv.axes_labels_fontsize,
				labelpad=gv.axes_labelpad,
			)
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


def DecoderInstanceCompare(phymet, logmet, dbses_input, chids = [0], thresholds=None):
	# Compare performance of various decoders.
	# Only show alpha values that correspond to distinct number of Pauli error rates.
	if thresholds is None:
		thresholds={"y": 10e-16, "x": 10e-16}
	qcode = dbses_input[0].eccs[0]
	max_weight = 1 + qcode.N//2
	noise = dbses_input[0].available[chids[0], :-1]
	sample = int(dbses_input[0].available[chids[0], -1])
	(budgets, uniques) = np.unique(np.array([GetTotalErrorBudget(dbs, noise, sample) for dbs in dbses_input[1:]], dtype=np.int64), return_index=True)
	# Add the entries for the minimum weight decoder.
	if (np.prod(dbses_input[0].decoders) == 1):
		budgets = np.concatenate(([0], budgets))
		uniques = np.concatenate(([0], 1 + uniques))

	# print("budgets = {}".format(budgets))

	dbses = [dbses_input[d] for d in uniques]
	ndb = len(dbses)
	plotfname = DecodersInstancePlot(dbses_input[0], phymet, logmet)
	nlevels = max([db.levels for db in dbses])

	# max_budget = np.sum([comb(dbses[0].eccs[0].N, i) * 3**i for i in range(max_weight + 1)])/4**dbses[0].eccs[0].N

	# The top xticklabels show the Pauli error budget left out in the NR data set.
	alphas = np.array([dbs.decoder_fraction for dbs in dbses], dtype = np.float64)
	nr_weights = np.load(NRWeightsFile(dbses[0], noise))[sample, :]
	chan_probs = np.load(RawPhysicalChannel(dbses_input[0], noise))[sample, :]
	budget_left = np.zeros(alphas.size, dtype = np.double)
	for (i, alpha) in enumerate(alphas):
		(__, __, knownPaulis) = GetLeadingPaulis(alpha, qcode, chan_probs, "weight", nr_weights)
		budget_left[i] = 1 - np.sum(knownPaulis)
		# xticklabels_top[i] = scientific_float(1 - np.sum(knownPaulis))

	# Stretch X axis by enlarging the canvas
	extended = (gv.canvas_size[0], gv.canvas_size[1])

	with PdfPages(plotfname) as pdf:
		for l in range(1, nlevels + 1):
			phyerrs = np.load(PhysicalErrorRates(dbses[0], phymet))[:, l-1]
			print("phyerrs: {}".format(phyerrs))
			# fig = plt.figure(figsize=gv.canvas_size)
			# ax1 = plt.gca()
			(fig, (ax2, ax1)) = plt.subplots(2, 1, sharex=True, figsize=extended, gridspec_kw={'height_ratios': [1, 3]})
			ax1.plot(
				[],
				[],
				color="k",
				linestyle="--",
				label="$%s = %s$"
				% (
					ml.Metrics[phymet]["latex"].replace("$", ""),
					latex_float(phyerrs[chids[0]]),
				),
			)
			minwt_bottom = 1
			for (c, ch) in enumerate(chids):
				settings = {
					"xaxis": [],
					"xlabel": "Remaining fraction of total probability",
					"yaxis": [],
					"ylabel": "$\\overline{%s_{%d}}$"
					% (ml.Metrics[logmet]["latex"].replace("$", ""), l),
					"color": gv.Colors[c % gv.n_Colors],
					"marker": gv.Markers[c % gv.n_Markers],
					"linestyle": "--",
				}
				contains_minwt = 0
				for d in range(ndb - 1, -1, -1):
					if dbses[d].decoders[l - 1] == 0:
						ax1.axhline(
							y=np.load(LogicalErrorRates(dbses[d], logmet))[ch, l],
							linestyle="--",
							linewidth=gv.line_width,
							color="green",
							label="MLD",
						)
					elif dbses[d].decoders[l - 1] == 1:
						contains_minwt = 1
					else:
						settings["xaxis"].append(budget_left[d])
						settings["yaxis"].append(
							np.load(LogicalErrorRates(dbses[d], logmet))[ch, l]
						)
						# print("{} --- {}".format(int(dbses[d].decoder_fraction * (4 ** dbses[0].eccs[0].N)), dbses[d].timestamp))
				sortorder = np.argsort(settings["xaxis"])
				settings["xaxis"] = np.array(settings["xaxis"])[sortorder]
				settings["yaxis"] = np.array(settings["yaxis"])[sortorder]
				# Plotting
				# print("X: {}\nY: {}\nMWD: {}".format(settings["xaxis"], settings["yaxis"], minwt))
				plotobj = ax1.plot(
					settings["xaxis"],
					settings["yaxis"],
					color=settings["color"],
					alpha=0.75,
					marker="o",  # settings["marker"]
					markersize=gv.marker_size,
					linestyle=settings["linestyle"],
					linewidth=gv.line_width,
					label="$%s = %s$" % (ml.Metrics[phymet]["latex"].replace("$", ""), latex_float(phyerrs[ch]))
					# label="$\\mathcal{D}_{\\alpha}$",
				)
				if (contains_minwt == 1):
					minwt_perf = np.load(LogicalErrorRates(dbses_input[0], logmet))[ch, l]
					# Add an empty plot for a legend entry.
					# temporarily muting the legend - debug
					# ax1.plot([], [], linestyle="--", linewidth=gv.line_width, color="red", label="MWD")
					ax2.plot(
						settings["xaxis"],
						[minwt_perf] * len(settings["xaxis"]),
						alpha = 0.75,
						linestyle="--",
						linewidth=gv.line_width,
						color="red",
						label="MWD",
					)
					if (minwt_perf <= minwt_bottom):
						minwt_bottom = minwt_perf

				# print("X axis for dcplot\n{}".format(settings["xaxis"]))
				texts = []
				for i in range(len(settings["xaxis"])):
					texts.append(ax1.text(settings["xaxis"][i], settings["yaxis"][i], "%d" % (budgets[-(i + 1)]), fontsize=gv.ticks_fontsize * 0.75))

			# Set xticks and labels
			ax1.invert_xaxis()

			# Set the y axis limits
			(__, minwt_top) = ax2.get_ylim()
			ax2.set_ylim(minwt_bottom/5, 5 * minwt_top)

			# Fuse ax1 and ax2 on top of each other such that they share the same X axis
			# https://matplotlib.org/examples/pylab_examples/broken_axis.html
			# hide the spines between ax and ax2
			ax1.spines['top'].set_visible(False)
			ax2.spines['bottom'].set_visible(False)
			ax2.xaxis.tick_top()
			ax1.xaxis.tick_bottom()

			# Axes labels
			ax1.set_xlabel(
				settings["xlabel"],
				fontsize=gv.axes_labels_fontsize * 0.8,
				labelpad=gv.axes_labelpad,
			)
			ax1.set_ylabel(
				settings["ylabel"],
				fontsize=gv.axes_labels_fontsize,
				labelpad=gv.axes_labelpad,
			)
			ax1.tick_params(
				axis="both",
				which="both",
				pad=gv.ticks_pad,
				direction="inout",
				length=gv.ticks_length,
				width=gv.ticks_width,
				labelsize=gv.ticks_fontsize,
			)
			ax2.tick_params(
				axis="both",
				which="both",
				pad=gv.ticks_pad,
				direction="inout",
				length=gv.ticks_length,
				width=gv.ticks_width,
				labelsize=gv.ticks_fontsize,
			)
			ax2.tick_params(
				axis="x",
				labeltop=False
			)
			# temporarily muting the legend
			ax1.legend(
				numpoints=1,
				loc="upper center",
				ncol=4,
				shadow=True,
				fontsize=gv.legend_fontsize,
				markerscale=gv.legend_marker_scale,
			)
			locmaj = ticker.LogLocator(base=10,numticks=1)
			ax1.yaxis.set_major_locator(locmaj)
			ax1.grid(axis='y',which='both')
			AddKinkLine(ax2, ax1)

			ax1.set_xscale("log")
			ax1.set_yscale("log")
			ax2.set_xscale("log")
			ax2.set_yscale("log")

			# Make non overlapping annotations
			# https://stackoverflow.com/questions/19073683/matplotlib-overlapping-annotations-text
			if (ADJUST == 1):
				adjust_text(texts, only_move={'points':'y', 'texts':'y'}, expand_points=(1, 2), precision=0.05, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

			# Save the plot
			pdf.savefig(fig)
			plt.close()
		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Comparing different decoders"
		pdfInfo["Author"] = "Pavithran Iyer"
		pdfInfo["ModDate"] = dt.datetime.today()
	return None


def AllocateBins(values, threshold_width=10):
	# Arrange the values into bins, each of which have a threshold (max) width.
	# Output is a dictionary with bin index "g" refering to the indices in the array that belong to the bin g.
	sort_index = np.argsort(values)
	bins = {}
	bin_count = 0
	current_bin_min = values[sort_index[0]]
	for i in sort_index:
		if(values[i] > threshold_width * current_bin_min):
			bin_count += 1
		if bin_count in bins:
			bins[bin_count].append(i)
		else:
			bins[bin_count] = [i]
			current_bin_min = values[i]
	print("values\n{}\nbins\n{}".format(values[sort_index], bins))
	return bins


def SelectAlphas(ndb, nbins, logerrs, bin_phyerr):
	# If the number of points excluded is more than 50%, do not plot the curve.
	selected_indices = [np.zeros(ndb - 1, dtype = np.double) for b in range(nbins)]
	count_selected = 0
	for b in range(nbins):
		selected = np.zeros(ndb - 1, dtype = np.int64)
		for d in range(ndb - 1):
			filtered_dataset = [x for x in bin_phyerr[b] if (logerrs[d][x] != -1)]
			if (len(filtered_dataset)/len(bin_phyerr[b]) >= 0.5):
				selected[d] = 1
				count_selected += len(filtered_dataset)
		selected_indices[b], = np.nonzero(selected)
	return selected_indices
				

def FilterLogicalErrorRates(dbses, chids, logmet, level):
	# Load the logical error rates for plotting.
	# Include only those logical error rate estimates that have converged.
	ndb = len(dbses)
	rates = []
	samples = []
	for (c, ch) in enumerate(chids):
		noise = dbses[0].available[chids[ch], :-1]
		rates.append(np.argmin(np.sum(np.abs(dbses[0].noiserates - noise), axis=1)))
		samples.append(int(dbses[0].available[chids[ch], -1]))
	is_converged = np.zeros((ndb, len(rates), len(samples)), dtype = np.int64)
	for d in range(ndb):
		if os.path.isfile(IsConvergedFile(dbses[d], logmet)):
			is_converged[d, :, :] = np.load(IsConvergedFile(dbses[d], logmet))
		else:
			is_converged[d, :, :] = IsConverged(dbses[d], logmet, rates, samples, threshold = 10)
			np.save(IsConvergedFile(dbses[d], logmet), is_converged[d, :, :])

	# Load the logical error rates
	logerrs = {d: -1 * np.ones(len(chids), dtype = np.double) for d in range(ndb - 1)}
	minalpha_perfs = np.zeros(len(chids), dtype = np.double)
	for (c, ch) in enumerate(chids):
		noise = dbses[0].available[ch, :-1]
		rate_index = np.argmin(np.sum(np.abs(dbses[0].noiserates - noise), axis=1))
		sample_index = int(dbses[0].available[ch, -1])
		# Load the minimum alpha performance.
		minalpha_perfs[c] = np.load(LogicalErrorRates(dbses[1], logmet))[ch, level]
		for d in range(1, ndb): # Start from d = 2 when taking the ratio between the first alpha and the rest.
			if (is_converged[d, rate_index, sample_index] == 1):
				# Normalize with the performance of the lowest alpha (ideally, RB).
				logerrs[d - 1][c] = np.load(LogicalErrorRates(dbses[d], logmet))[ch, level]/minalpha_perfs[ch]
	return (logerrs, minalpha_perfs)


def BinPhysErrs(phyerrs, logerrs, bin_width, ndb):
	# Bin the physical error rates and average logical error rates in a bin.
	print("phyerrs\n{}".format(phyerrs))
	bins = AllocateBins(phyerrs, bin_width)
	nbins = len(bins)
	logerrs_binned = np.zeros((3, ndb - 1, nbins), dtype = np.double)
	filtered = {d: [] for d in range(ndb - 1)}
	for b in range(nbins):
		for d in range(ndb - 1):
			filtered_dataset = [x for x in bins[b] if (logerrs[d][x] != -1)]
			filtered[d].extend(filtered_dataset)
			median = np.median(logerrs[d][filtered_dataset])
			logerrs_binned[0, d, b] = median
			# Compute the lower and upper error bars.
			logerrs_binned[1, d, b] = median - np.percentile(logerrs[d][filtered_dataset], 25)
			logerrs_binned[2, d, b] = np.percentile(logerrs[d][filtered_dataset], 75) - median
	return (bins, logerrs_binned, filtered)
			

def GetTVD(dbses, chids, bins, logerrs):
	# Extract the TVDs for each alpha, and plot as an inset.
	# Get the TVD for each decoder knowledge with the full Pauli error distribution. These will be in the annotations.
	ndb = len(dbses)
	nbins = len(bins)
	alphas = [dbs.decoder_fraction for dbs in dbses]
	tvds = np.zeros((ndb - 1, len(chids)), dtype = np.double)
	for d in range(1, len(alphas)):
		tvds[d - 1, :] = np.load(PhysicalErrorRates(dbses[d], "dctvd"))[chids]
		# print("TVDs for alpha = {}\n{}".format(alphas[d], tvds[d - 1, :]))
	# Bin the TVDs
	tvds_filtered = np.zeros((3, ndb, nbins), dtype = np.double)
	for b in range(nbins):
		for d in range(ndb - 1):
			filtered_dataset = [x for x in bins[b] if (logerrs[d][x] != -1)]
			median = np.median(tvds[d, filtered_dataset])
			tvds_filtered[0, d, b] = median
			# Compute the upper and lower error bars.
			tvds_filtered[1, d, b] = median - np.percentile(tvds[d, filtered_dataset], 25)
			tvds_filtered[2, d, b] = np.percentile(tvds[d, filtered_dataset], 75) - median
	return tvds_filtered


def GetBudgets(dbses, chids):
	# Compute the budgets (X-axis): number of Pauli error rates in NR.
	ndb = len(dbses)
	budgets = np.zeros((ndb - 1, len(chids)), dtype = np.double)
	budget_left = np.zeros((ndb - 1, len(chids)), dtype = np.double)
	for (c, ch) in enumerate(chids):
		noise = dbses[0].available[chids[ch], :-1]
		sample = int(dbses[0].available[chids[ch], -1])
		nr_weights = np.load(NRWeightsFile(dbses[0], noise))[sample, :]
		budgets[:, c] = np.array([GetTotalErrorBudget(dbs, noise, sample) for dbs in dbses[1:]], dtype=np.int64)
		# chan_probs = np.load(RawPhysicalChannel(dbses[0], noise))[sample, :]
		# for d in range(1, alphas.size):
		# 	(__, __, knownPaulis) = GetLeadingPaulis(alphas[d], qcode, chan_probs, "weight", nr_weights)
		# 	budget_left[d - 1, c] = 1 - np.sum(knownPaulis)
	return budgets

def SetInsetTVD(ax_principal, xaxes, yaxes, selected, bins):
	# Plot the TVDs for the various alphas in the inset plot.
	# Inset axes
	nbins = len(bins)
	ax_inset = plt.axes([0, 0, 1, 1])
	# Position and relative size of the inset axes within ax_principal
	ip = InsetPosition(ax_principal, [0.17, 0.2, 0.32, 0.3]) # Positon: bottom right
	ax_inset.set_axes_locator(ip)
	# Mark the region corresponding to the inset axes on ax_principal and draw lines in grey linking the two axes.
	mark_inset(ax_principal, ax_inset, loc1=2, loc2=4, fc="none")
	
	for b in range(nbins):
		# If the number of points in a bin in less than 5, do not plot.
		if (len(bins[b]) < 5):
			continue
		# Plot
		# print("TVD for bin {}\n{}".format(b, yaxes[0, selected[b], b]))
		ax_inset.errorbar(
			xaxes[b, selected[b]],
			yaxes[0, selected[b], b],
			yerr=yaxes[1:, selected[b], b],
			color=gv.Colors[b % gv.n_Colors],
			alpha=0.75,
			marker="o",
			markersize=gv.marker_size,
			linestyle="--",
			linewidth=gv.line_width
			# label="$\\langle %s\\rangle = %s$" % (ml.Metrics[phymet]["latex"].replace("$", ""), latex_float(average_phymet))
		)

	# Gridlines
	ax_inset.grid(which="both", axis="both", color="0.85")# Gridlines
	ax_inset.grid(which="both", axis="both", color="0.85")

	# Scales
	ax_inset.set_xscale("log")
	ax_inset.set_yscale("log")
	
	# Axes labels
	ax_inset.set_xlabel("$K$", fontsize=gv.axes_labels_fontsize * 0.8, labelpad=gv.axes_labelpad)
	ax_inset.set_ylabel("TVD ($\\delta$)", fontsize=gv.axes_labels_fontsize * 0.75, labelpad=gv.axes_labelpad)
	# Axes ticks
	ax_inset.tick_params(axis="both", which="both", pad=gv.ticks_pad, direction="inout", length=gv.ticks_length, width=gv.ticks_width, labelsize=gv.ticks_fontsize * 0.75)
	return None


def RelativeDecoderInstanceCompare(phymet, logmet, dbses, chids = [0], thresholds=None):
	# Compare performance of various decoders.
	# Only show alpha values that correspond to distinct number of Pauli error rates.
	
	if thresholds is None:
		thresholds={"y": 10e-16, "x": 10e-16}
	
	qcode = dbses[0].eccs[0]
	max_weight = 1 + qcode.N//2
	ndb = len(dbses)
	plotfname = DecodersInstancePlot(dbses[0], phymet, logmet)
	nlevels = max([db.levels for db in dbses])
	alphas = np.array([dbs.decoder_fraction for dbs in dbses], dtype = np.float64)

	# Sort the alphas.
	sort_order = np.argsort(alphas)
	alphas = alphas[sort_order]
	dbses = [dbses[i] for i in sort_order]

	# print("channels: {}".format(chids))

	phyerrs = np.load(PhysicalErrorRates(dbses[0], phymet))[chids]
	
	bin_width = 2
	with PdfPages(plotfname) as pdf:
		for l in range(nlevels, nlevels + 1):
			fig = plt.figure(figsize=(gv.canvas_size[0] * 1.3, gv.canvas_size[1]))
			ax = plt.gca()
			
			# Load the logical error rates that have converged well.
			(yaxes, minalpha_perfs) = FilterLogicalErrorRates(dbses, chids, logmet, l)
			print("l: {}\nminalpha_perfs\n{}\nYaxes\n{}".format(l, minalpha_perfs, yaxes))

			# Bin the physical and logical error rates.
			(bins, yaxes_binned, filtered) = BinPhysErrs(phyerrs, yaxes, bin_width, ndb)
			nbins = len(bins)
			# print("bin_width = {}\nbin_sizes\n{}".format(bin_width, [len(bins[b]) for b in bins]))
			
			# Print the performance of the minimum alpha (RB) for each bin.
			for b in range(nbins):
				if (len(bins[b]) >= 5):
					print("Bin {}, Logical performance with RB data: {}".format(b, np.mean(minalpha_perfs[bins[b]])))
			
			# Compute the TVDs for the decoders and bin them.
			tvds_filtered = GetTVD(dbses, chids, bins, yaxes)

			# Compute the number of Pauli error rates in NR.
			budgets = GetBudgets(dbses, chids)
			xaxes = np.zeros((nbins, ndb - 1), dtype = np.double)
			for b in range(nbins):
				xaxes[b, :] = np.array([np.mean(budgets[d, filtered[d]]) for d in range(ndb - 1)])
			
			# If the number of points excluded is more than 50% of the bin, ignore the bin in the plot.
			selected = SelectAlphas(ndb, nbins, yaxes, bins)

			# Bin the physical error rates and average logical error rates in a bin.
			max_y = 0
			min_y = np.max(np.array(list(yaxes.values())))
			plots = []
			labels = []
			empty_plots = []
			rb_perf_labels = []
			for b in range(nbins):
				average_phymet = np.median(phyerrs[bins[b]])
				#################
				# Exclude bins
				# If the number of points in a bin in less than 5, do not plot.
				# print("Average number of channels selected in bin %d = %.2f." % (b, selected.size/(ndb - 1)))
				if (len(bins[b]) < 5):
					continue
				#################
				# Plotting
				pl = ax.errorbar(
					xaxes[b, selected[b]],
					yaxes_binned[0, selected[b], b],
					yerr=yaxes_binned[1:, selected[b], b],
					color=gv.Colors[b % gv.n_Colors],
					alpha=0.75,
					marker="o",
					markersize=gv.marker_size,
					linestyle="--",
					linewidth=gv.line_width
				)
				plots.append(pl)
				lab = "$\\langle %s\\rangle = %s$" % (ml.Metrics[phymet]["latex"].replace("$", ""), latex_float(average_phymet))
				labels.append(lab)
				
				# Add an empty plot for the RB logical error rate labels.
				pl, = ax.plot(
					[], [],
					color=gv.Colors[b % gv.n_Colors],
					alpha=0.75,
					marker="o",
					markersize=gv.marker_size,
					linestyle="--",
					linewidth=gv.line_width,
				)
				empty_plots.append(pl)
				lab = "$\\langle \\overline{%s}^{RB}_{%d}\\rangle = %s$" % (ml.Metrics[logmet]["latex"].replace("$", ""), l, latex_float(np.mean(minalpha_perfs[bins[b]])))
				rb_perf_labels.append(lab)

				# Compute the max y value for designing the axes limits
				if (max_y < np.max(yaxes_binned[0, :, b])):
					max_y = np.max(yaxes_binned[0, :, b])
				if (min_y > np.min(yaxes_binned[0, :, b])):
					min_y = np.min(yaxes_binned[0, :, b])

				"""
				# Annotate each point with the TVD
				texts = []
				for d in range(ndb - 1):
					if (selected[d] == 1):
						texts.append(ax.text(xaxes[d], yaxes_binned[d, b] * 0.4, "$%g$" % np.round(tvds_filtered[d, b], 4), fontsize=gv.ticks_fontsize * 0.75, rotation=-20))
				"""

			# Set the inset plots to show TVD.
			SetInsetTVD(ax, xaxes, tvds_filtered, selected, bins)

			# Axes limits
			# ax.set_ylim([min_y / 5, max_y * 5])

			# Gridlines
			ax.grid(which="both", axis="both", color="0.85")

			# Axes labels
			ax.set_xlabel(
				"Number of Pauli decay rates $(K)$",
				fontsize=gv.axes_labels_fontsize,
				labelpad=gv.axes_labelpad,
			)
			ax.set_ylabel(
				"Gain over RB ($\\Delta_{%d}$)" % (l),
				fontsize=gv.axes_labels_fontsize,
				labelpad=gv.axes_labelpad,
			)
			# Axes ticks
			ax.tick_params(
				axis="both",
				which="both",
				pad=gv.ticks_pad,
				direction="inout",
				length=gv.ticks_length,
				width=gv.ticks_width,
				labelsize=gv.ticks_fontsize,
			)
			# Lengend with curve labels of average physical infidelity
			physinfid_legend = ax.legend(
				plots,
				labels,
				numpoints=1,
				# loc="lower left",
				loc="upper center",
				ncol=2,
				bbox_to_anchor=(0.45, 1.3),
				shadow=True,
				fontsize=gv.legend_fontsize * 1.4,
				markerscale=gv.legend_marker_scale,
			)
			ax.add_artist(physinfid_legend)
			rb_perf_legend = ax.legend(
				empty_plots,
				rb_perf_labels,
				numpoints=1,
				# loc="lower left",
				loc="center left",
				ncol=1,
				bbox_to_anchor=(1, 0.6),
				shadow=True,
				fontsize=gv.legend_fontsize * 1.4,
				markerscale=gv.legend_marker_scale,
			)
			ax.set_xscale("log")
			ax.set_yscale("log")

			# Axes ticks
			# print("max_y = {} and min_y = {}".format(max_y, min_y))
			yticks = np.arange(OrderOfMagnitude(min_y/5), OrderOfMagnitude(max_y * 5))
			ax.set_yticks(np.power(10.0, yticks), minor=True)
			# print("Y ticks\n{}".format(yticks))

			# Make non overlapping annotations
			# https://stackoverflow.com/questions/19073683/matplotlib-overlapping-annotations-text
			# if (ADJUST == 1):
			# 	adjust_text(texts, only_move={'points':'y', 'texts':'y'}, expand_points=(1, 2), precision=0.05, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

			# Save the plot
			plt.tight_layout(pad=30)
			pdf.savefig(fig)
			plt.close()

		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Comparing different decoders"
		pdfInfo["Author"] = "Pavithran Iyer"
		pdfInfo["ModDate"] = dt.datetime.today()
	return None


def AddKinkLine(ax, ax2):
	# Add the diagonal lines in a broken y-axis plot.
	# Code copied from: https://matplotlib.org/examples/pylab_examples/broken_axis.html
	# "ax" is the top plot.
	d = 0.01  # how big to make the diagonal lines in axes coordinates
	# arguments to pass to plot, just so we don't keep repeating them
	kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
	ax.plot((-d, +d), (0, 0), linewidth=gv.line_width, **kwargs)        # top-left diagonal
	ax.plot((1 - d, 1 + d), (0, 0), linewidth=gv.line_width, **kwargs)  # top-right diagonal

	kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
	ax2.plot((-d, +d), (1, 1), linewidth=gv.line_width, **kwargs)  # bottom-left diagonal
	ax2.plot((1 - d, 1 + d), (1, 1), linewidth=gv.line_width, **kwargs)  # bottom-right diagonal

	# What's cool about this is that now if we vary the distance between
	# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
	# the diagonal lines will move accordingly, and stay right at the tips
	# of the spines they are 'breaking'
	return None
