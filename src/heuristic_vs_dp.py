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
	
	print("1 qubit error probabilities in the original channel\n{}".format(chan_probs[qcode.group_by_weight[1]]))

	print("2 qubit error probabilities in the original channel\n{}".format(chan_probs[qcode.group_by_weight[2]]))

	print("3 qubit error probabilities in the original channel\n{}".format(chan_probs[qcode.group_by_weight[3]]))

	# Filling using the Heuristic
	(heuristic, __) = CompleteDecoderKnowledge(leading_fraction, chan_probs, qcode, ["knr","split"])
	# print("3 qubit error probabilities in the heuristic\n{}".format(heuristic[qcode.group_by_weight[3]]))

	# Filling using the depolarizing ansatz
	(depolarizing, __) = CompleteDecoderKnowledge(leading_fraction, chan_probs, qcode, ["knr","dp"])
	# print("3 qubit error probabilities in the depolarizing channel\n{}".format(depolarizing[qcode.group_by_weight[3]]))
	
	# Order errors by weights
	weight_ordering = np.concatenate([qcode.group_by_weight[w] for w in range(qcode.N+1)])
	nerrors_upto_weight = np.cumsum([comb(qcode.N, i) * 3**i for i in range(qcode.N+1)])
	
	# Plotting Depolarizing Vs. Heuristic.
	plotfname = "depolarizing_vs_heuristic.pdf"
	with PdfPages(plotfname) as pdf:
		fig = plt.figure(figsize=(gv.canvas_size[0] * 1.5, gv.canvas_size[1] * 1.2))
		# plt.plot(np.arange(1, 1 + npauli, dtype=int), depolarizing[weight_ordering], linestyle="None", marker="o", markersize=0.8 * gv.marker_size, color="blue", label="Depolarizing")
		# plt.plot(np.arange(1, 1 + npauli, dtype=int), heuristic[weight_ordering], linestyle="None", marker="s", markersize=0.5 * gv.marker_size, color="red", label="Heuristic")

		plt.plot(np.arange(1, 1 + npauli, dtype=int), chan_probs[weight_ordering], linestyle="None", marker="d", markersize=0.3 * gv.marker_size, color="k", label="True Channel")

		# Vertical times to demarcate weights
		plt.axvline(x=1, color="k", linestyle="dashed", linewidth=gv.line_width) # weight = 0 errors
		for w in range(qcode.N+1):
			plt.axvline(x=nerrors_upto_weight[w], color=gv.Colors[w % gv.n_Colors], linestyle="dashed", linewidth=gv.line_width, label="Weight $w=%d$" % (w)) # weight = w errors
		plt.xscale("log")
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

	# LST ordering
	# test_ops = [0, 100, 200, 300, 400, 500]
	# print("LST ordering of operators:\n{}".format(qecc.PauliOperatorsLST[test_ops, :]))

	# test_ops = [0, 10, 20, 30, 40, 50]
	# (ls_ops, phases) = qc.GetOperatorsForTLSIndex(qecc, range(2**(qecc.N + qecc.K)))
	# print("LS Operators\n{}\nPhases\n{}".format(ls_ops[test_ops], phases[test_ops]))
	
	# Load the channel
	chan_probs_samples = np.load("/home/pavi/Documents/IQC/chbank/unitary/linear/unitary_intensive_dc_0/physical/raw_corr_unitary_0.3_4_9_1.25.npy")
	chan_probs = chan_probs_samples[0, :]

	print("p = {}".format((1 - np.power(chan_probs[0], 1/7)) / 3))
	# for w in range(4):
	# 	print("Probabilities of weight - {} errors".format(w))
	# 	print(np.real(chan_probs[qecc.group_by_weight[w]]))

	# print("non zero entries: {}".format(np.count_nonzero(np.real(chan_probs))))

	# Compare the distributions
	lead_frac = -2
	compare_depolarizing_heuristic(qecc, lead_frac, chan_probs)