def DecoderKnowledgeFile(dbs, noise):
    # File containing the decoder knowledge.
    noisedes = "_".join(list(map(lambda p: ("%g" % p), noise)))
    fname = "%s/physical/dc_%s_%s.npy" % (dbs.outdir, dbs.channel, noisedes)
    return fname


def DeviationPlotFile(dbs, pmet, lmet):
    # File name containing scatter bins.
    fname = "%s/results/dvplot_%s_%s.pdf" % (dbs.outdir, pmet, lmet)
    return fname

def DecodersPlot(dbs, pmet, lmet):
    # File name containing scatter bins.
    fname = "%s/results/dcplot_%s_%s.pdf" % (dbs.outdir, pmet, lmet)
    return fname


def DecodersInstancePlot(dbs, pmet, lmet):
    # File name containing scatter bins.
    fname = "%s/results/dciplot_%s_%s.pdf" % (dbs.outdir, pmet, lmet)
    return fname


def HammerPlot(dbs, lmet, pmets):
    # File name containing scatter bins.
    fname = "%s/results/hammer_%s_%s.pdf" % (dbs.outdir, "_".join(pmets), lmet)
    return fname


def CompareScatters(dbs, lmet, pmets, mode="metrics"):
    # File name containing scatter bins.
    fname = "%s/results/scatbins_%s_%s.pdf" % (dbs.outdir, "_".join(pmets), lmet)
    return fname


def PauliDistribution(outdir, channel):
    # Name of the file containing the probability distribution of Pauli errors.
    fname = "%s/results/paulidist_%s.pdf" % (outdir, channel)
    return fname


def BinSummary(submit, phymet, logmet, level):
    # Name of the file containing the summary of bins
    fname = "%s/results/bins_summary_%s_%s_l%d.txt" % (
        submit.outdir,
        phymet,
        logmet,
        level,
    )
    return fname


def SubmissionInputs(timestamp):
    # Name of the file containing the submission input
    fname = "./../input/%s.txt" % (timestamp)
    return fname


def SubmissionSchedule(timestamp):
    # Name of the file containing the submission parameters schedule
    fname = "./../input/schedule_%s.txt" % (timestamp)
    return fname


def DecoderBins(dbs, noise, sample):
    # Name of the file containing the physical channel
    noisedes = "_".join(list(map(lambda p: ("%g" % p), noise)))
    fname = "%s/channels/decbins_%s_s%d.txt" % (dbs.outdir, noisedes, sample)
    return fname


def PhysicalChannel(dbs, noise):
    # Name of the file containing the physical channel
    noisedes = "_".join(list(map(lambda p: ("%g" % p), noise)))
    fname = "%s/physical/%s_%s.npy" % (dbs.outdir, dbs.channel, noisedes)
    return fname


def ChannelInformationFile(dbs, noise):
    # Name of the file containing the physical channel
    noisedes = "_".join(list(map(lambda p: ("%g" % p), noise)))
    fname = "%s/physical/info_%s_%s.txt" % (dbs.outdir, dbs.channel, noisedes)
    return fname


def RawPhysicalChannel(dbs, noise):
    # Name of the file containing the physical channel
    noisedes = "_".join(list(map(lambda p: ("%g" % p), noise)))
    fname = "%s/physical/raw_%s_%s.npy" % (dbs.outdir, dbs.channel, noisedes)
    return fname


def OutputDirectory(path, dbs):
    # Directory containing the simulation results
    dirname = "%s/%s" % (path, dbs.timestamp)
    return dirname


def LogicalChannel(dbs, noise, samp):
    # File containing the logical channels
    noisedes = "_".join(list(map(lambda p: ("%g" % p), noise)))
    fname = "%s/channels/logchan_%s_s%d.npy" % (dbs.outdir, noisedes, samp)
    return fname


def LogChanVariance(dbs, noise, samp):
    # File containing the logical channels
    noisedes = "_".join(list(map(lambda p: ("%g" % p), noise)))
    fname = "%s/channels/chanvar_%s_s%d.npy" % (dbs.outdir, noisedes, samp)
    return fname


def LogicalErrorRate(dbs, noise, samp, metric, average=0):
    # File containing the logical error rates for a given physical channel and logical metric.
    noisedes = "_".join(list(map(lambda p: ("%g" % p), noise)))
    if average == 0:
        fname = "%s/metrics/%s_%s_s%d.npy" % (dbs.outdir, metric, noisedes, samp)
    else:
        fname = "%s/metrics/ave_%s_%s_s%d.npy" % (dbs.outdir, metric, noisedes, samp)
    return fname


def LogErrVariance(dbs, noise, samp, metric):
    # File containing the logical error rates for a given physical channel and logical metric.
    noisedes = "_".join(list(map(lambda p: ("%g" % p), noise)))
    fname = "%s/metrics/var_%s_%s_s%d.npy" % (dbs.outdir, metric, noisedes, samp)
    return fname


def SyndromeBins(dbs, noise, samp, metric):
    # File containing the syndrome-metric bins.
    noisedes = "_".join(list(map(lambda p: ("%g" % p), noise)))
    fname = "%s/metrics/bins_%s_%s_s%d.npy" % (dbs.outdir, metric, noisedes, samp)
    return fname


def LogicalErrorRates(dbs, metric, fmt="npy"):
    # File containing all the logical error rates for the physical channels in the database
    fname = "%s/results/log_%s.%s" % (dbs.outdir, metric, fmt)
    return fname


def PhysicalErrorRates(dbs, metric):
    # File containing all the logical error rates for the physical channels in the database
    fname = "%s/results/phy_%s.npy" % (dbs.outdir, metric)
    return fname


def ThreshPlot(submit, pmet, lmet):
    # File containing the plots of the logical error rate
    fname = "%s/results/thresh_%s_vs_%s.pdf" % (submit.outdir, lmet, pmet)
    return fname


def ChannelWise(submit, pmet, lmet):
    # File containing the plots of the logical error rate
    fname = "%s/results/chan_%s_vs_%s.pdf" % (submit.outdir, lmet, pmet)
    return fname


def LevelWise(submit, pmet, lmet):
    # File containing the plots of the logical error rate
    fname = "%s/results/%s_vs_%s.pdf" % (submit.outdir, lmet, pmet)
    return fname


def CompareLogErrRates(dbses, lmet):
    # File containing the plot of comparing the logical error rates from two (or more) submission records.
    fname = "%s/results/compare_%s_%s.pdf" % (
        dbses[0].outdir,
        "_".join([dbses[i].timestamp for i in range(1, len(dbses))]),
    )
    return fname


def CalibrationData(chname, metric):
    # File containing the data for calibrating a channel with a particular output metric.
    fname = "./../temp/calib_%s_%s.txt" % (chname, metric)
    return fname


def CalibrationPlot(chname, metric):
    # File containing the data for calibrating a channel with a particular output metric.
    fname = "./../temp/calib_%s_%s.pdf" % (chname, metric)
    return fname


def FitPhysRates(dbs, lmet):
    # File containing the fit obtained physical error rates
    fname = "%s/results/pfit_%s.npy" % (dbs.outdir, lmet)
    return fname


def FitWtEnums(dbs, lmet):
    # File containing the fit obtained weight enumerator coefficients (combinatorial factors)
    fname = "%s/results/wefit_%s.npy" % (dbs.outdir, lmet)
    return fname


def FitExpo(dbs, lmet):
    # File containing the fit obtained exponents for the physical error rate
    fname = "%s/results/expofit_%s.npy" % (dbs.outdir, lmet)
    return fname


def AnsatzComparePlot(dbs, lmet, pmet):
    # File containing the plot comparing physical error metric with a fit obtained error metric
    fname = "%s/results/%s_fit_vs_%s.pdf" % (dbs.outdir, pmet, lmet)
    return fname


def TrainingSet(dbs):
    # File containing the training set -- physical channel parameters with the fit error rates
    fname = "%s/results/mltrain.npy" % (dbs.outdir)
    return fname


def TestingSet(dbs):
    # File containing the testing set -- physical channel parameters
    fname = "%s/results/mltest.npy" % (dbs.outdir)
    return fname


def PredictedPhyRates(dbs):
    # File containing the predictions of the physical error rates
    fname = "%s/results/pred.npy" % (dbs.outdir)
    return fname


def PredictComparePlot(dbs, lmet, pmet):
    # File containing the comparisons between the machine learnt metric and a standard metric -- by studying both their correlations with the logical error rate.
    fname = "%s/results/val_%s_vs_%s.pdf" % (dbs.outdir, pmet, lmet)
    return fname


def RFECVRankings(dbs, deg):
    # File containing the Regression rankings
    fname = "%s/results/rfecv_%d.npy" % (dbs.outdir, deg)
    return fname


def MCStatsPlotFile(dbs, lmet, pmet):
    # File containing the plot of average logical error rate with different syndrome samples
    fname = "%s/results/mcplot_%s_%s.pdf" % (dbs.outdir, pmet, lmet)
    return fname


def RunningAverageCh(dbs, noise, samp, metric):
    # File containing the data for running averages for top level metric values for a specific channel.
    noisedes = "_".join(list(map(lambda p: ("%g" % p), noise)))
    fname = "%s/metrics/running_%s_%s_s%d.npy" % (dbs.outdir, metric, noisedes, samp)
    return fname


def RunningAverages(dbs, lmet):
    # File containing the data for running averages of top level metrics for all channels
    fname = "%s/results/runavg_%s.npy" % (dbs.outdir, lmet)
    return fname


def SyndromeBinsPlot(dbs, lmet, pvals):
    # File containing the plot of syndrome bins for specific logical error metric and physical error rate.
    if pvals == -1:
        fname = "%s/results/bplot_%s_all.pdf" % (dbs.outdir, lmet)
    else:
        fname = "%s/results/bplot_%s_%s.pdf" % (
            dbs.outdir,
            lmet,
            "_".join(map(str, pvals)),
        )
    return fname


def VarianceBins(dbs, lmet, pmet):
    # File containing the bar plot of variance in each bin of a scatter plot.
    fname = "%s/results/scatbins_%s_%s.pdf" % (dbs.outdir, str(pmet), lmet)
    return fname


def CompressionMatrix(dbs, lmet, level):
    # File containing the compression matrix.
    fname = "%s/results/compmat_%s_l%d.npy" % (dbs.outdir, lmet, level)
    return fname


def CompressedParams(dbs, lmet, level):
    # File containing the compression matrix.
    fname = "%s/results/compressed_%s_l%d.npy" % (dbs.outdir, lmet, level)
    return fname


# =====================================================================================================================================

# def ChannelsBankFname():
# 	# Name of the file containing the details of all available simulations.
# 	if (HOST == 0):
# 		fname = "/Users/pavithran/Documents/LDPC/Channel_Flow/available.txt"
# 	elif (HOST == 1):
# 		fname = "/Users/poulingroup/Desktop/pavithran/channels/available.txt"
# 	else:
# 		fname = "/home/pavi/Channel_Flow/Bank/available.txt"

# 	return fname

# def LastBackupDateTime(dbsObj):
# 	# File containing the details of the date and time at which the last backup was taken
# 	fname = ("%s/Last.txt" % (dbsObj.backup))
# 	return fname

# def LogicalChannelsFname(dbsObj, noise, sampIndex, isAveraged = 0):
# 	# Assign a name for the file containing the logical channels produced by starting with a specific physical channel
# 	if (isAveraged == 0):
# 		fname = ("%s/channels/%s_DB_%g_%d_%d_s%d.npy" % (dbsObj.dirname, dbsObj.pertType, noise, dbsObj.nStats, dbsObj.nLevels, sampIndex))
# 	else:
# 		fname = ("%s/channels/%s_ave_DB_%g_%d_%d_s%d.npy" % (dbsObj.dirname, dbsObj.pertType, noise, dbsObj.nStats, dbsObj.nLevels, sampIndex))
# 	return fname

# def DatabaseDir(dbsObj):
# 	# Directory containing all the data files related to the database object
# 	if (HOST == 0):
# 		dirname = ("/Users/pavithran/Documents/LDPC/Channel_Flow/%s_%s_%g_%g_%g" % (dbsObj.eccType, dbsObj.baseChType, dbsObj.noiseRange[0], dbsObj.noiseRange[-1], dbsObj.noiseRange[1] - dbsObj.noiseRange[0]))
# 	elif (HOST == 1):
# 		dirname = ("/Users/poulingroup/Desktop/pavithran/channels/%s_%s_%g_%g_%g" % (dbsObj.eccType, dbsObj.baseChType, dbsObj.noiseRange[0], dbsObj.noiseRange[-1], dbsObj.noiseRange[1] - dbsObj.noiseRange[0]))
# 	else:
# 		dirname = ("/home/pavi/Channel_Flow/Bank/%s_%s_%g_%g_%g" % (dbsObj.eccType, dbsObj.baseChType, dbsObj.noiseRange[0], dbsObj.noiseRange[-1], dbsObj.noiseRange[1] - dbsObj.noiseRange[0]))
# 	return dirname

# def BackupDir(dbsObj):
# 	# Directory containing all the data files related to the database object
# 	if (HOST == 0):
# 		dirname = ("/Users/pavithran/Documents/LDPC/Channel_Flow/backups/%s_%s_%g_%g_%g" % (dbsObj.eccType, dbsObj.baseChType, dbsObj.noiseRange[0], dbsObj.noiseRange[-1], dbsObj.noiseRange[1] - dbsObj.noiseRange[0]))
# 	elif (HOST == 1):
# 		dirname = ("/Users/poulingroup/Desktop/pavithran/channels/backups/%s_%s_%g_%g_%g" % (dbsObj.eccType, dbsObj.baseChType, dbsObj.noiseRange[0], dbsObj.noiseRange[-1], dbsObj.noiseRange[1] - dbsObj.noiseRange[0]))
# 	else:
# 		dirname = ("/home/pavi/Channel_Flow/Bank/%s_%s_%g_%g_%g" % (dbsObj.eccType, dbsObj.baseChType, dbsObj.noiseRange[0], dbsObj.noiseRange[-1], dbsObj.noiseRange[1] - dbsObj.noiseRange[0]))
# 	return dirname

# def IndvMetFname(dbsObj, metName, noise, sampIndex, isLogicalMetric = 0, isAveraged = 0):
# 	# File containing metrics for logical channels corresponding to all concatenation levels (averaged over several decoding trials) of a particular physical channels
# 	if (isLogicalMetric == 0):
# 		fname = ("%s/metrics/%s_%s_%g_s%d.txt" % (dbsObj.dirname, dbsObj.pertType, metName, noise, sampIndex))
# 	else:
# 		if (isAveraged == 0):
# 			fname = ("%s/metrics/%s_%s_%g_%d_%d_s%d.npy" % (dbsObj.dirname, dbsObj.pertType, metName, noise, dbsObj.nStats, dbsObj.nLevels, sampIndex))
# 		else:
# 			fname = ("%s/metrics/ave_%s_%s_%g_%d_%d_s%d.npy" % (dbsObj.dirname, dbsObj.pertType, metName, noise, dbsObj.nStats, dbsObj.nLevels, sampIndex))
# 	return fname


# def LogicalErrorRates(dbsObj, logicalMetric):
# 	# File containing the logical error rates of a database
# 	fname = ("logErr_%s_%s_%s_%d_%d_%d.npy" % (dbsObj.eccType, dbsObj.pertType, logicalMetric, dbsObj.nStats, dbsObj.nLevels, dbsObj.nSamps))
# 	return fname

# def ThresholdDataFname(dbsObj, physicalMetric, logicalMetric, profilingMetric, partialStats = 0):
# 	# File containing the data for threhold plots for a combination of metrics to denote physical and logical error rates.
# 	if (partialStats == 0):
# 		partialStats = dbsObj.nStats

# 	fname = ("threholds_%s_%s_%s_%s_%s_%s_%d_%d_%d.npy" % (dbsObj.eccType, dbsObj.baseChType, dbsObj.pertType, physicalMetric, logicalMetric, profilingMetric, partialStats, dbsObj.nLevels, dbsObj.nSamps))
# 	return fname

# def ThresholdPlotFname(dbsObj, physicalMetric, logicalMetric, profilingMetric, partialStats = 0):
# 	# File containing the threhold plots for a combination of metrics to denote physical and logical error rates.
# 	if (partialStats == 0):
# 		partialStats = dbsObj.nStats

# 	fname = ("threholds_%s_%s_%s_%s_%s_%s_%d_%d_%d.pdf" % (dbsObj.eccType, dbsObj.baseChType, dbsObj.pertType, physicalMetric, logicalMetric, profilingMetric, partialStats, dbsObj.nLevels, dbsObj.nSamps))
# 	return fname

# def CalibrationData(chname, metric):
# 	# File containing the values of a particuar metric for different noise parameters of a given channel
# 	fname = ("calibration_%s_%s.npy" % (chname, metric))
# 	return fname

# def CalibrationPlot(chname, metrics):
# 	# File containing the plot showing the values of a particular noise metric for different noise parameters of a given channel
# 	fname = ("calibration_%s_%s.pdf" % (chname, "_".join(metrics)))
# 	return fname

# def DatabaseComparisions(dbs1, dbs2):
# 	# File containing the logical metrics data for two different plots
# 	fname = ("compare_%s_%s.npy" % (dbs1.eccType, dbs2.eccType))
# 	return fname

# def ComparisonsPlotFname(dbs1, dbs2):
# 	# File containing the logical metrics data for two different plots
# 	fname = ("compare_%s_%s_%s_%s_%d.pdf" % (dbs1.eccType, dbs1.chType, dbs2.eccType, dbs2.chType, min(dbs1.nLevels, dbs2.nLevels)))
# 	return fname

# def ExtremeChannels(dbsObj, physicalMetric, logicalMetric):
# 	# File containing the best and worst logical channel, for every value of the physical metric
# 	fname = ("extremes_%s_%s_%g_%g_%s_%s.npy" % (dbsObj.eccType, dbsObj.pertType, dbsObj.noiseRange[0], dbsObj.noiseRange[-1], physicalMetric, logicalMetric))
# 	return fname

# def ProfilingData(dbsObj, profilingMetric, physicalMetric, logicalMetric):
# 	# File containing the profiling of best and worst channels for a given physical noise metric
# 	fname = ("profile_%s_%s_%g_%g_%s_%s_%s.npy" % (dbsObj.eccType, dbsObj.pertType, dbsObj.noiseRange[0], dbsObj.noiseRange[-1], physicalMetric, logicalMetric, profilingMetric))
# 	return fname

# def ProfilingPlot(dbsObj, profilingMetric, physicalMetric, logicalMetric):
# 	# File containing the profiling of best and worst channels for a given physical noise metric
# 	fname = ("profile_%s_%s_%g_%g_%s_%s_%s.pdf" % (dbsObj.eccType, dbsObj.pertType, dbsObj.noiseRange[0], dbsObj.noiseRange[-1], physicalMetric, logicalMetric, profilingMetric))
# 	return fname

# def BestPhysicalNoiseRates(dbsObj, logicalMetric):
# 	# File containing the physical error rates that can best explain the logical error rate (according to a preset formula).
# 	fname = ("bestPhys_%s_%s_%s_%d_%d_%d.npy" % (dbsObj.eccType, dbsObj.pertType, logicalMetric, dbsObj.nStats, dbsObj.nLevels, dbsObj.nSamps))
# 	return fname

# def BestWeightEnumerators(dbsObj, logicalMetric):
# 	# File containing the weight enumerator coefficients for a particular type of error correcting code and a choice of logical metric such that the weight enumerators produce the best approximation to the corresponding logical error rate.
# 	fname = ("weight_enumerators_%s_%s.npy" % (dbsObj.eccType, logicalMetric))
# 	return fname

# def TrainingSet(dbsObj, logicalMetric, metricsToCompute):
# 	# File containing the training set that is used to derive a relation between the standard noise metrics and the physical metric that best describes the logical error rate.
# 	fname = ("training_set_%s_%s_%s_%s_%d_%d_%d.npy" % (dbsObj.eccType, dbsObj.pertType, logicalMetric, "_".join(metricsToCompute), dbsObj.nStats, dbsObj.nLevels, dbsObj.nSamps))
# 	return fname

# def PredictedLogicalError(dbsTest, dbsTrain, logicalMetric, metricsToCompute):
# 	# File containing the predicted logical error rates, at every concatenation level, for a database of physical channels.
# 	fname = ("predictions_%s_%s_%d_%d_%d_%s_%s_%d_%s_%s.npy" % (dbsTest.eccType, dbsTest.pertType, dbsTest.nStats, dbsTest.nLevels, dbsTest.nSamps, dbsTrain.eccType, dbsTrain.chType, dbsTrain.nChannels, logicalMetric, "_".join(metricsToCompute)))
# 	return fname

# def ComparePredictedMeasured(dbsTest, dbsTrain, logicalMetric, metricsToCompute):
# 	# PDF file containing the comparisons between the predicted and the measured logical error rates.
# 	fname = ("comparisons_%s_%s_%d_%d_%d_%s_%s_%d_%s_%s.pdf" % (dbsTest.eccType, dbsTest.pertType, dbsTest.nStats, dbsTest.nLevels, dbsTest.nSamps, dbsTrain.eccType, dbsTrain.chType, dbsTrain.nChannels, logicalMetric, "_".join(metricsToCompute)))
# 	return fname

# def FormulaVerificationPlots(dbsObj, logicalMetric):
# 	# PDF file containing the comparisons between the predicted and the measured logical error rates.
# 	fname = ("formula_%s_%s_%d_%d_%d_%s.pdf" % (dbsObj.eccType, dbsObj.pertType, dbsObj.nStats, dbsObj.nLevels, dbsObj.nSamps, logicalMetric))
# 	return fname

# def AnsatzTesting(dbsObj, logicalMetric):
# 	# File containing the plots of the best fit physical error rate, with the logical error rates, for all channels in a database
# 	fname = ("ansatz_%s_%s_%s_%d_%d_%d.pdf" % (dbsObj.eccType, dbsObj.pertType, logicalMetric, dbsObj.nStats, dbsObj.nLevels, dbsObj.nSamps))
# 	return fname

# # =====================================================================================================================================

# def MetricHistogramData(dbsObj, metName, noiseRateExp, sampIndex, li):
# 	# File containing the metric values of all the logical channels, at a given level.
# 	fname = ("%s/metrics/hist_%s_%s_%g_%d_l%d_s%d.npy" % (dbsObj.dirname, dbsObj.pertType, metName, noiseRateExp, dbsObj.nStats, li, sampIndex))
# 	return fname

# def HistogramPlot(dbsObj, metName, noiseRateExp):
# 	# File containing the Histogram plots (for each of the levels) showing the number of channels for every window of the logical metric
# 	fname = ("hist_%s_%s_%g_%d_%d.pdf" % (dbsObj.pertType, metName, noiseRateExp, dbsObj.nLevels, dbsObj.nStats))
# 	return fname
