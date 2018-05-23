import os
import sys
try:
	import numpy as np
except:
	pass
from define import fnames as fn
# Force the module scripts to run locally -- https://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path
# import inspect as ins
# current = os.path.realpath(os.path.abspath(os.path.dirname(ins.getfile(ins.currentframe()))))
# if (not (current in sys.path)):
# 	sys.path.insert(0, current)

def IsComplete(submit):
	# Determine the noise rates and samples for which simulation output data is available.
	if (submit.complete == -1):
		chcount = 0
		for i in range(submit.noiserates.shape[0]):
			for j in range(submit.samps):
				if (os.path.isfile(fn.LogicalChannel(submit, submit.noiserates[i], j))):
					chcount = chcount + 1
		submit.channels = chcount
		if (chcount > 0):
			submit.available = np.zeros((submit.channels, 1 + submit.noiserates.shape[1]), dtype = np.float)
			chcount = 0
			for i in range(submit.noiserates.shape[0]):
				for j in range(submit.samps):
					if (os.path.isfile(fn.LogicalChannel(submit, submit.noiserates[i], j))):
						submit.available[chcount, :-1] = submit.noiserates[i, :]
						submit.available[chcount, -1] = j
						chcount = chcount + 1
		submit.complete = (100 * chcount)/np.float(submit.noiserates.shape[0] * submit.samps)
		print("\033[2mSimulation data is available for %d%% of the channels.\033[0m" % (submit.complete))
	return submit.complete


def GatherLogErrData(submit):
	# Gather the logical error rates data from all completed simulations and save as a 2D array in a file.
	for m in range(len(submit.metrics)):
		logErr = np.zeros((submit.channels, submit.levels + 1), dtype = np.longdouble)
		for i in range(submit.channels):
			fname = fn.LogicalErrorRate(submit, submit.available[i, :-1], submit.available[i, -1], submit.metrics[m])
			logErr[i, :] = np.load(fname)
		# Save the gathered date to a numpy file.
		fname = fn.LogicalErrorRates(submit, submit.metrics[m], fmt = "npy")
		np.save(fname, logErr)
		# Save the gathered date to a text file.
		fname = fn.LogicalErrorRates(submit, submit.metrics[m], fmt = "txt")
		with open(fname, "w") as fp:
			fp.write("# Channel Noise Sample %s\n" % (" ".join([("L%d" % l) for l in range(1 + submit.levels)])))
			for i in range(submit.channels):
				fp.write("%d %s %d" % (i, " ".join(map(lambda num: ("%g" % num), submit.available[i, :-1])), submit.available[i, -1]))
				for l in range(submit.levels + 1):
					fp.write(" %g" % (logErr[i, l]))
				fp.write("\n")
	return None


def ComputeBestFitLine(xydata, line):
	# Compute the best fit line for a X, Y (two columns) dataset, in the log-scale.
	# Use linear regression: https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line
	tol = 10E-20
	xave = 0.0
	yave = 0.0
	for i in range(xydata.shape[0]):
		if (xydata[i, 1] > tol):
			xave = xave + xydata[i, 0]
			yave = yave + np.log(xydata[i, 1])
	xave = xave/np.float(xydata.shape[0])
	yave = yave/np.float(xydata.shape[0])
	covxy = 0.0
	varx = 0.0
	for i in range(xydata.shape[0]):
		if (xydata[i, 1] > tol):
			covxy = covxy + (xydata[i, 0] - xave) * (np.log(xydata[i, 1]) - yave);
			varx = varx + (xydata[i, 0] - xave) * (xydata[i, 0] - xave)
	# Best fit line
	line[0] = covxy/np.float(varx);
	line[1] = yave - line[0] * xave;
	return None


def ComputeThreshold(dbs, logicalmetric):
	# Compute the threshold of the code using the logical error rates data, with respect to the noise parameter.
	# For every physical noise rate, determine if it is above or below threshold.
	# Above threshold iff the logical error rate decreases with the level of concatenation and below otherwise.
	# The threshold is the average of the smallest physical noise rate that is above threshold and the largest physical noise rate that is below threshold.
	logErr = np.load(fn.GatheredLogErrData(dbs, logicalmetric))
	regime = np.zeros(dbs.noiserates.shape[0], dtype = np.int8)
	bestfit = np.zeros(2, dtype = np.float)
	for i in range(dbs.channels):
		ComputeBestFitLine(np.concatenate((np.arange(dbs.levels + 1)[:, np.newaxis], logErr[i, :, np.newaxis]), axis = 1), bestfit)
		regime[dbs.available[i, 0]] = np.power(-1, int(bestfit[0] < 0))
	thresholds = np.zeros(2, dtype = np.float)
	for i in range(dbs.noiserates.shape[0]):
		if (regime[i] == 1):
			threshold[0] = dbs.noiserates[i][0]
			break
	for i in range(dbs.noiserates.shape[0]):
		if (regime[dbs.noiserates.shape[0] - i - 1] == -1):
			threshold[1] = dbs.noiserates[i][0]
			break
	threshold = (thresholds[0] + thresholds[1])/np.float(2)
	return threshold

