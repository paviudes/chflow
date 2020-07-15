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


def IsEmptyFolder(dirname):
    """
	Check if a folder is empty.
	"""
    if os.path.isdir(dirname):
        if len([f for f in os.listdir(dirname) if not f.startswith(".")]) == 0:
            return 1
    return 0


def IsComplete(submit):
    # Determine the noise rates and samples for which simulation output data is available.
    if submit.complete == -1:
        chcount = 0
        if not IsEmptyFolder("%s/channels" % (submit.outdir)):
            for i in range(submit.noiserates.shape[0]):
                for j in range(submit.samps):
                    # print("%s" % (fn.LogicalChannel(submit, submit.noiserates[i], j)))
                    if os.path.isfile(
                        fn.LogicalChannel(submit, submit.noiserates[i], j)
                    ):
                        chcount = chcount + 1
        submit.channels = chcount
        if chcount > 0:
            submit.available = np.zeros(
                (submit.channels, 1 + submit.noiserates.shape[1]), dtype=np.float
            )
            chcount = 0
            for i in range(submit.noiserates.shape[0]):
                for j in range(submit.samps):
                    if os.path.isfile(
                        fn.LogicalChannel(submit, submit.noiserates[i], j)
                    ):
                        submit.available[chcount, :-1] = submit.noiserates[i, :]
                        submit.available[chcount, -1] = j
                        chcount = chcount + 1
        submit.complete = (100 * chcount) / np.float(
            submit.noiserates.shape[0] * submit.samps
        )
        print(
            "\033[2mSimulation data is available for %d%% of the channels.\033[0m"
            % (submit.complete)
        )
    return submit.complete


def CollectPhysicalChannels(submit):
    """
	Get the physical channels for a database.
	"""
    if submit.iscorr == 0:
        nparams = 4 ** submit.eccs[0].K * 4 ** submit.eccs[0].K
    else:
        submit.rawchans = np.zeros(
            (submit.noiserates.shape[0], submit.samps, 4 ** submit.eccs[0].N),
            dtype=np.longdouble,
        )
        nparams = 2 ** (submit.eccs[0].N + submit.eccs[0].K)

    submit.phychans = np.zeros(
        (submit.noiserates.shape[0], submit.samps, nparams), dtype=np.longdouble
    )
    for i in range(submit.noiserates.shape[0]):
        (folder, fname) = os.path.split(
            fn.PhysicalChannel(submit, submit.noiserates[i])
        )
        if os.path.isfile("%s/%s" % (folder, fname)) == 1:
            submit.phychans[i, :, :] = np.load("%s/%s" % (folder, fname))
            if submit.iscorr == 1:
                submit.rawchans[i, :, :] = np.load("%s/raw_%s" % (folder, fname))
    return None


def GatherLogErrData(submit, additional=[]):
    # Gather the logical error rates data from all completed simulations and save as a 2D array in a file.
    # If no running averages data is found, a zero array is stored in its place.
    logmetrics = submit.metrics + additional
    for m in range(len(logmetrics)):
        logerr = np.zeros((submit.channels, submit.levels + 1), dtype=np.longdouble)
        runavg = np.zeros((submit.channels, submit.stats.shape[0]), dtype=np.longdouble)
        for i in range(submit.channels):
            for quant in range(2):
                fname = fn.LogicalErrorRate(
                    submit,
                    submit.available[i, :-1],
                    submit.available[i, -1],
                    logmetrics[m],
                    average=quant,
                )
                if os.path.isfile(fname):
                    break
            logerr[i, :] = np.load(fname)
            fname = fn.RunningAverageCh(
                submit, submit.available[i, :-1], submit.available[i, -1], logmetrics[m]
            )
            if os.path.isfile(fname):
                # print("running average for channel {}\n{}".format(i, np.load(fname)))
                runavg[i, :] = np.load(fname)
        # Save the gathered date to a numpy file.
        fname = fn.LogicalErrorRates(submit, logmetrics[m], fmt="npy")
        np.save(fname, logerr)
        # Save the running averages to a file
        fname = fn.RunningAverages(submit, logmetrics[m])
        np.save(fname, runavg)
        # Save the gathered date to a text file.
        fname = fn.LogicalErrorRates(submit, logmetrics[m], fmt="txt")
        with open(fname, "w") as fp:
            fp.write(
                "# Channel Noise Sample %s\n"
                % (" ".join([("L%d" % l) for l in range(1 + submit.levels)]))
            )
            for i in range(submit.channels):
                fp.write(
                    "%d %s %d"
                    % (
                        i,
                        " ".join(
                            map(lambda num: ("%g" % num), submit.available[i, :-1])
                        ),
                        submit.available[i, -1],
                    )
                )
                for l in range(submit.levels + 1):
                    fp.write(" %g" % (logerr[i, l]))
                fp.write("\n")
    return None


def ComputeBestFitLine(xydata):
    # Compute the best fit line for a X, Y (two columns) dataset, in the log-scale.
    # Use linear regression: https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line
    # print("xydata\n%s" % (np.array_str(xydata)))

    line = np.zeros(2, dtype=np.float)
    tol = 10e-20
    xave = 0.0
    yave = 0.0
    for i in range(xydata.shape[0]):
        if xydata[i, 1] > tol:
            xave = xave + xydata[i, 0]
            yave = yave + np.log10(xydata[i, 1])
    xave = xave / np.float(xydata.shape[0])
    yave = yave / np.float(xydata.shape[0])

    # print("xave = %g, yave = %g." % (xave, yave))

    covxy = 0.0
    varx = 0.0
    for i in range(xydata.shape[0]):
        if xydata[i, 1] > tol:
            covxy = covxy + (xydata[i, 0] - xave) * (np.log10(xydata[i, 1]) - yave)
            varx = varx + (xydata[i, 0] - xave) * (xydata[i, 0] - xave)
    # print("covxy = %g, varx = %g." % (covxy, varx))

    # Best fit line
    line[0] = covxy / np.float(varx)
    line[1] = yave - line[0] * xave
    return line


def ComputeThreshold(dbs, lmet):
    # Compute the threshold of the code using the logical error rates data, with respect to the noise parameter.
    # For every physical noise rate, determine if it is above or below threshold.
    # Above threshold iff the logical error rate decreases with the level of concatenation and below otherwise.
    # The threshold is the average of the smallest physical noise rate that is above threshold and the largest physical noise rate that is below threshold.
    logErr = np.load(fn.LogicalErrorRates(dbs, lmet, fmt="npy"))
    regime = np.zeros(dbs.noiserates.shape[0], dtype=np.int8)
    bestfit = np.zeros(2, dtype=np.float)
    for i in range(dbs.channels):
        bestfit = ComputeBestFitLine(
            np.concatenate(
                (np.arange(dbs.levels + 1)[:, np.newaxis], logErr[i, :, np.newaxis]),
                axis=1,
            )
        )
        regime[dbs.available[i, 0]] = np.power(-1, int(bestfit[0] < 0))
    thresholds = np.zeros(2, dtype=np.float)
    for i in range(dbs.noiserates.shape[0]):
        if regime[i] == 1:
            threshold[0] = dbs.noiserates[i][0]
            break
    for i in range(dbs.noiserates.shape[0]):
        if regime[dbs.noiserates.shape[0] - i - 1] == -1:
            threshold[1] = dbs.noiserates[i][0]
            break
    threshold = (thresholds[0] + thresholds[1]) / np.float(2)
    return threshold


def AddPhysicalRates(dbs):
    # Only for backward compatibility
    # Add the physical error rate to the logical error rates data
    # Read the physical error rates.
    phyerr = np.zeros((len(dbs.metrics), dbs.channels), dtype=np.longdouble)
    for m in range(len(dbs.metrics)):
        phyerr[m, :] = np.load(fn.PhysicalErrorRates(dbs, dbs.metrics[m]))
        for i in range(dbs.channels):
            fname = fn.LogicalErrorRate(
                dbs, dbs.available[i, :-1], dbs.available[i, -1], dbs.metrics[m]
            )
            logerr = np.load(fname)
            logerr[0] = phyerr[m, i]
            np.save(fname, logerr)
            print(
                "\r\033[2m%d channels remaining.\033[0m"
                % (dbs.channels * len(dbs.metrics) - (m * dbs.channels + i + 1))
            ),
    print("")
    return None
