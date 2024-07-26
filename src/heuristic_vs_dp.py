import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import datetime as dt
from scipy.special import comb

from define import globalvars as gv
from define import qcode as qc
from define.randchans import CreateIIDPauli
from define.decoder import GetLeadingPaulis
from define.decoder import CompleteDecoderKnowledge

def compare_depolarizing_heuristic(qcode, leading_fraction, chan_probs):
	# Use the Depolarizing Amsatz to compute the probability distribution of errors: D(E)
	# Use the heuristic to compute the probability distribution of errors: H(E)
	# Plot D(E) and H(E) for errors ordered according to their weights.
	npauli = np.power(4, qcode.N, dtype=int)
	
	# Filling using the Heuristic
	(heuristic, __) = CompleteDecoderKnowledge(leading_fraction, chan_probs, qcode, ["full","split"])

	# Filling using the depolarizing ansatz
	(depolarizing, __) = CompleteDecoderKnowledge(leading_fraction, chan_probs, qcode, ["full","dp"])
	
	# Order errors by weights
	weight_ordering = np.concatenate([qcode.group_by_weight[w] for w in range(qcode.N+1)])
	nerrors_upto_weight = np.cumsum([comb(qcode.N, i) * 3**i for i in range(qcode.N+1)])
	
	# Plotting Depolarizing Vs. Heuristic.
	plotfname = "depolarizing_vs_heuristic.pdf"
	with PdfPages(plotfname) as pdf:
		fig = plt.figure(figsize=(gv.canvas_size[0] * 1.5, gv.canvas_size[1] * 1.2))
		plt.plot(np.arange(npauli, dtype=int), depolarizing[weight_ordering], linestyle="None", marker="o", markersize=0.2 * gv.marker_size, color="blue", label="Depolarizing")
		plt.plot(np.arange(npauli, dtype=int), heuristic[weight_ordering], linestyle="None", marker="s", markersize=0.2 * gv.marker_size, color="red", label="Heuristic")

		plt.plot(np.arange(npauli, dtype=int), chan_probs[weight_ordering], linestyle="None", marker="d", markersize=0.2 * gv.marker_size, color="k", label="True Channel")

		# Vertical times to demarcate weights
		plt.axvline(x=1, color="k", linestyle="dashed", linewidth=gv.line_width) # weight = 0 errors
		for w in range(qcode.N+1):
			plt.axvline(x=nerrors_upto_weight[w], color=gv.Colors[w % gv.n_Colors], linestyle="dashed", linewidth=gv.line_width, label="Weight $w=%d$" % (w)) # weight = w errors
		plt.yscale("log")
		
		plt.xlabel("Error count", fontsize=gv.axes_labels_fontsize, labelpad=gv.axes_labelpad)
		plt.ylabel("Probabilities", fontsize=gv.axes_labels_fontsize, labelpad=gv.axes_labelpad)

		# Axes ticks
		ax=plt.gca()
		ax.tick_params(
			axis="both",
			which="both",
			pad=gv.ticks_pad,
			direction="inout",
			length=gv.ticks_length,
			width=gv.ticks_width,
			labelsize=gv.ticks_fontsize,
		)

		plt.legend(numpoints=1, loc="best", shadow=True, fontsize=gv.legend_fontsize * 1.2, markerscale=gv.legend_marker_scale)

		# Save figure
		pdf.savefig(fig)
		plt.close()

		# Set PDF attributes
		pdfInfo = pdf.infodict()
		pdfInfo["Title"] = "Comparing different decoder ansatz"
		pdfInfo["Author"] = "Pavithran Iyer and Aditya Jain"
		pdfInfo["ModDate"] = dt.datetime.today()

	print("TVDs for r = {}".format(1 - chan_probs[0]))
	print("|original - heuristic| = {}".format(np.linalg.norm(heuristic - chan_probs)))
	print("|original - depolarizing| = {}".format(np.linalg.norm(depolarizing - chan_probs)))
	print("|depolarizing - heuristic| = {}".format(np.linalg.norm(heuristic - depolarizing)))
	
	return None

if __name__ == '__main__':
	# Load the code
	qecc = qc.QuantumErrorCorrectingCode("Steane")
	qc.Load(qecc)
	qc.PrepareSyndromeLookUp(qecc)
	
	# Load the channel
	chan_probs_samples = np.load("/home/pavi/Documents/IQC/chbank/cg1d/split/cg1d_lam3_high_dc_0.01/physical/raw_cg1d_0.55_4_12_2.5.npy")
	chan_probs = chan_probs_samples[0, :]

	# Compare the distributions
	lead_frac = 0.01
	compare_depolarizing_heuristic(qecc, lead_frac, chan_probs)