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
from define.heuristic import AssignErrorProbs

def compare_depolarizing_heuristic(qcode, known_paulis, known_probs, infid):
	# Use the Depolarizing Amsatz to compute the probability distribution of errors: D(E)
	# Use the heuristic to compute the probability distribution of errors: H(E)
	# Plot D(E) and H(E) for errors ordered according to their weights.
	npauli = np.power(4, qcode.N, dtype=int)
	infid_qubit = 1 - np.power(1 - infid, 1/qcode.N)
	
	# Filling using the Heuristic
	heuristic = AssignErrorProbs(known_paulis.astype(np.uint64), known_probs.astype(np.float64), qcode.PauliOperatorsLST.astype(np.uint8), np.float64(infid_qubit))

	# Filling using the depolarizing ansatz
	depolarizing = CreateIIDPauli(infid_qubit, qcode)
	depolarizing[known_paulis] = known_probs
	
	# Normalizing the total probability of unknown errors
	total_unknown = 1 - np.sum(known_probs)
	# # https://stackoverflow.com/questions/27824075/accessing-numpy-array-elements-not-in-a-given-index-list
	mask = np.ones(depolarizing.shape[0], dtype=bool)
	mask[known_paulis] = False
	depolarizing[mask] = total_unknown * depolarizing[mask] / np.sum(depolarizing[mask])
	heuristic[mask] = total_unknown * heuristic[mask] / np.sum(heuristic[mask])

	# Order errors by weights
	weight_ordering = np.concatenate([qcode.group_by_weight[w] for w in range(qcode.N+1)])
	nerrors_upto_weight = np.cumsum([comb(qcode.N, i) * 3**i for i in range(qcode.N+1)])
	
	# Plotting Depolarizing Vs. Heuristic.
	plotfname = "depolarizing_vs_heuristic.pdf"
	with PdfPages(plotfname) as pdf:
		fig = plt.figure(figsize=(gv.canvas_size[0] * 1.5, gv.canvas_size[1] * 1.2))
		plt.plot(np.arange(npauli, dtype=int), depolarizing[weight_ordering], linestyle="None", marker="o", markersize=0.2 * gv.marker_size, color="blue", label="Depolarizing")
		plt.plot(weight_ordering, heuristic[weight_ordering], linestyle="None", marker="s", markersize=0.2 * gv.marker_size, color="red", label="Heuristic")

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
		
	return None

if __name__ == '__main__':
	# Load the code
	qecc = qc.QuantumErrorCorrectingCode("Steane")
	qc.Load(qecc)
	qc.PrepareSyndromeLookUp(qecc)
	
	# Load the channel
	chan_probs = np.load("/home/pavi/Documents/IQC/chbank/cg1d/fill/cg1d_lam3_dc_0.01_fill/physical/raw_cg1d_0.35_4_12_3.npy")
	
	# Get the leading probs
	lead_frac = 0.01
	infid, known_paulis, known_probs = GetLeadingPaulis(lead_frac, qecc, chan_probs, "full")

	# Compare the distributions
	compare_depolarizing_heuristic(qecc, known_paulis, known_probs, infid)