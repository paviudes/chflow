import os
import sys
import time
import numpy as np
import setup as st

# Files from the "define" module.
from define.fnames import (
    PhysicalChannel,
    LogicalErrorRates,
    RawPhysicalChannel,
    FitPhysRates,
    FitWtEnums,
    FitExpo,
    CompressedParams,
    PredictedPhyRates,
    HammerPlot,
    LevelWise,
    PauliDistribution,
    ChannelWise,
    MCStatsPlotFile,
    DecodersInstancePlot,
    DeviationPlotFile
)
from define.qcode import (
    QuantumErrorCorrectingCode,
    IsCanonicalBasis,
    ConstructSyndromeProjectors,
    PrepareSyndromeLookUp,
    PrintQEC,
)

from define import globalvars as gv
from define.verifychans import IsQuantumChannel
from define.submission import Submission, PrintSub, MergeSubs, Select
from define.utils import LoadTimeStamp, IsNumber
from define.save import SavePhysicalChannels, Save, Schedule, PrepOutputDir
from define.load import LoadSub
from define.merge import MergeSubs
from define.metrics import Metrics, ComputeNorms, ComputeMetrics  # Calibrate,
from define.qchans import Channels, SaveChan  # Twirl, PrintChan
from define.genchans import PreparePhysicalChannels
from define.chanreps import (
    CreatePauliDistChannels,
    TwirlChannels,
    ConvertRepresentations,
)

from define.QECCLfid.utils import GetErrorProbabilities

from analyze.collect import IsComplete, GatherLogErrData, AddPhysicalRates
from analyze.cplot import ChannelWisePlot
from analyze.dcplot import DecoderCompare, DecoderInstanceCompare
from analyze.dvplot import PlotDeviationYX
from analyze.lplot import LevelWisePlot, LevelWisePlot2D
from analyze.statplot import MCStatsPlot
from analyze.hamplot import DoubleHammerPlot
from analyze.pdplot import PauliDistributionPlot
from analyze.utils import ExtractPDFPages

# from define import chandefs as chdef
# from define import chanapprox as capp
# from define import photonloss as ploss
# from define import gendamp as gdamp
# from define import chanreps as crep
# from define.QECCLfid import uncorrectable as uc
# from define.QECCLfid import utils as ut


def DisplayLogoLicense():
    # Display logo as ascii drawing from http://ascii.mastervb.net with font = xcourb.tiff
    # Display license from the LICENSE file in chflow/
    logo = r"""
	      ###      ###  ###
	       ##     ##     ##
	 ###   ####  #####   ##    ###  ## # ##
	## ##  ## ##  ##     ##   ## ##  # # #
	##     ## ##  ##     ##   ## ##  #####
	## ##  ## ##  ##     ##   ## ##   ####
	 ###   ## ## ####  ######  ###    # #
	"""
    license = r"""
	BSD 3-Clause License
	Copyright (c) 2018, Pavithran S Iyer, Aditya Jain and David Poulin
	All rights reserved.
	"""
    url = "https://github.com/paviudes/chflow/wiki"
    print(
        "%s\n\tWelcome to chflow version \033[1mv2.0\033[0m\n\tCheck out %s for help.\n%s"
        % (logo, url, license)
    )
    return None


def RemoteExecution(timestamp, node):
    # Load a submission record
    from simulate import simulate as sim

    submit = Submission()
    LoadSub(submit, timestamp, 0)

    # Prepare syndrome look-up table for hard decoding.
    if np.any(submit.decoders == 1):
        for l in range(submit.levels):
            if submit.decoders[l] == 1:
                if submit.eccs[l].lookup is None:
                    print(
                        "\033[2mPreparing syndrome lookup table for the %s code.\033[0m"
                        % (submit.eccs[l].name)
                    )
                    PrepareSyndromeLookUp(submit.eccs[l])

    # If no node information is specified, then simulate all nodes in serial.
    # Else simulate only the given node.
    if node > -1:
        print("Running local simulations")
        sim.LocalSimulations(submit, node)
    else:
        for i in range(submit.nodes):
            sim.LocalSimulations(submit, i)
    return None


if __name__ == "__main__":
    # This is the starting point for the "chflow" shell command.
    avchreps = list(
        map(lambda rep: ('"%s"' % (rep)), ["krauss", "choi", "chi", "process", "stine"])
    )
    avmets = list(map(lambda met: '"%s"' % (met), Metrics.keys()))
    avch = list(map(lambda chan: '"%s"' % (chan), Channels.keys()))
    mannual = {
        "qcode": [
            "Load a quantum error correcting code.",
            "qcode s(string)\n\twhere s is the name of the file containing the details of the code.",
        ],
        "qcbasis": [
            "Output the Canonical basis for the quantum error correcting code.",
            "No parameters.",
        ],
        "qcminw": [
            "Prepare the syndrome lookup table for (hard) minimum weight decoding algorithm.",
            "No parameters.",
        ],
        "qcprint": [
            "Print all the details on the underlying quantum error correcting code.",
            "No parameters.",
        ],
        "chan": [
            "Load a channel.",
            "chan s1(string) x1(float),x2(float),...\n\twhere s1 is either the name of a channel or a file name containing the channel information x1,x2,... specify the noise rates.",
        ],
        "chsave": [
            "Save a channel into a file.",
            "chsave s1(string)\n\twhere s1 is the name of the file.",
        ],
        "chrep": [
            "Convert from its current representation to another form.",
            "chrep s1(string)\n\twhere s1 must be one of %s." % (", ".join(avchreps)),
        ],
        "chtwirl": ["Twirl a quantum channel.", "No parameters."],
        "chpa": ["Honest Pauli Approximation of a quantum channel.", "No parameters."],
        "chmetrics": [
            "Compute standard metric(s).",
            "chmetrics s1(string)[,s2(string)]\n\ts1 must be one of %s."
            % (", ".join(avmets)),
        ],
        "chval": [
            "Test if all the properties of a CPTP map are satisfied.",
            "No parameters.",
        ],
        "chprint": [
            "Print a quantum channel in its current representation.",
            "No parameters.",
        ],
        "chcalib": [
            "Measure all the norms for a quantum channel.",
            'calibrate s1(string) x11(float),x12(float),n1(int);x21(float),x22(float),n2(int),... [m1(string),m2(string),m3(string),...] [c(int)]\n\twhere s1 must be one of %s\n\t"xi1", "xi2" and "ni" specify the noise range for calibration: from "xi1" to "xi2", in "ni" steps.\n\teach of m1, m2, m3, ... must be one of %s.\n\tc is a number that specified which noise parameter must be varied in the calibration plots. By default it is the first parameter, i.e, c = 0.'
            % (", ".join(avch), ", ".join(avmets)),
        ],
        "sbprint": [
            "Print details about the currently loaded submission.",
            "No parameters.",
        ],
        "sbload": [
            "Create a new submission of channels to be simulated.",
            "load [s1(string)]\n\twhere s1 either a time stamp specifying a submission or a file containing parameters. If no inputs are given, the user will be prompted on the console.",
        ],
        "sbmerge": [
            "Merge the results from two or more simulations.",
            "merge s(string) s1(string)[,s2(string),s3(string)...]\nwhere s is a name of the merged submission and s1, s2,... are names of the simulations to be merged.",
        ],
        "submit": [
            "Create a simulation input record with the current submission parameters.",
            "No parameters.",
        ],
        "sbsave": ["Save the current simulation parameters.", "No parameters."],
        "pmetrics": [
            "Compute metrics for the physical channels in the current submission.",
            "pmetrics m1(string)[,m2(string),m3(string),...]\nwhere m1,m2,... should be one of %s."
            % (avmets),
        ],
        "lpmetrics": [
            "Compute metrics for the physical channels that dependent on the levels in the current submission.",
            "lpmetrics m1(string)[,m2(string),m3(string),...]\nwhere m1,m2,... should be one of %s."
            % (avmets),
        ],
        "lmetrics": [
            "Compute metrics for the average logical channels (for all levels) in the current submission.",
            "lmetrics m1(string)[,m2(string),m3(string),...]\nwhere m1,m2,... should be one of %s."
            % (avmets),
        ],
        "threshold": [
            "Estimate the threshold of the ECC and quantum channel with the current simulation data.",
            "threshold s1(string)\nwhere s1 is a metric computed at the logical level.",
        ],
        "compare": [
            "Compare the logical error rates from two different simulations.",
            "compare s1(string) s2(string) m(string)\nwhere s1, s2 are names of the respective simulations to be compared and m is a metric that is computed at the logical level. Note: the logical error rates for the simulations should be available and gathered in the results/ folder.",
        ],
        "ecc": [
            "Run error correction simulations, if the simulation data doesn't already exist.",
            "No parameters",
        ],
        "collect": ["Collect results.", "No parameters"],
        "tplot": [
            "Threshold plots for the current database.",
            "tplot s11(string)[,s12(string),s13(string),...] s2(string)\nwhere each of s1i are either metric names or indices of indepenent parameters in the defition of the physical noise model; s2 is the name of the logical metric.",
        ],
        "lplot": [
            "Level-wise plots of the logical error rates vs physical noise rates in the current (and/or other) database(s).",
            "lplot s11(string)[,s12(string),s13(string),...] s2(string) [s31(string),s32(string),...]\nwhere each of s1i are either metric names or indices of indepenent parameters in the defition of the physical noise model; s2 is the name of the logical metric; s3i are time stamps of other simulation records that also need to be plotted alongside.",
        ],
        "lplot2d": [
            "Level-wise 2D (density) plots of the logical error rates vs. a pair of physical noise rates in the current (and/or other) database(s).",
            "lplot2d s11(string),s12(string) s2(string)\nwhere each of s1i are either metric names or indices of indepenent parameters in the defition of the physical noise model; s2 is the name of the logical metric.",
        ],
        "mcplot": [
            "Plot showing running average logical error rate with the number of syndrome samples for current and other databases.",
            "mcplot [s1(string)] [s2(string)] [s31(string),s32(string),...]\nwhere s1 and s2 are the logical and physical error metric names. s2 can either be a name or an index to denote the physical noise parameter. s3i are time stamps of additional databases.",
        ],
        "bplot": [
            "Plot the number of syndromes counted for a particular syndrome probability and a conditional logical error.",
            "bplot [s1(string)] [s2(string)] [s3(string)]\nwhere s1 is the name of a logical metric and s2 are values of the physical metrics for which the syndrome counts have to be ploted.",
        ],
        "varplot": [
            "Plot the variance in a scatter plot (of the logical error rates vs. physical metrics)",
            "varplot [s1(string)] [s2(string)] [i3(int)] [s3(string)]\nwhere s1 and s2 are logical and physical metrics (parameter index), respectively. i3 is the number of bins",
        ],
        "sbfit": [
            "Fit the logical error rates in the database with an ansatz and plot the results. s3 are time stamps of additional databases (separated by a comma) whose scatter plots must be combined with the current submission.",
            "sbfit [s1(string)] [s2(string)] [s31(string),s32(string),...]\nwhere s1 and s2 are physical and logical error metric names respectively; s3i are time stamp of additional simulation datasets that need to be fitted with the same ansatz.",
        ],
        "compress": [
            "Compute a compression matrix for the channel parameters such that compressed parameters vary whenever the logical error rates vary. If the compression matrix exists, simply compute the bin variance.",
            "compress i1(int) [s1(string)] [i2(int)] [i3(int)] [i4(int)]\nwhere i1 is the level whose logical error rates need to be compared for compression. s1 is the logical error metric. i3 is the number of parameters in the compressed version. i4 is the number of bins for dividing the parameter space to compute the variance.",
        ],
        "sblearn": [
            "Derive new noise rates for physical channels using machine learnning.",
            "sblearn s1(string) s2(string) s31(string)[,s32(string),...] [s4(string)] [s5(string)]\nwhere s1 is the name of a testing database; s2 is the name of a logical metric; s3i are the names of physical metrics to be included in the training set; s4 is the name of the machine learning method to be used; s5 is the name of a mask.",
        ],
        "build": [
            "Compile Cython files or C extension files.",
            "build s1(string) s2(string).\nwhere s1 is the directory in which the Cython (or C) files exist and s2 specifies if new C files need to be generated (using cythonize).",
        ],
        "clean": ["Remove compilation and run time files.", "No parameters."],
        "man": ["Mannual", "No parameters."],
        "quit": ["Quit", "No parameters."],
        "exit": ["Quit", "No parameters."],
    }

    # Display the logo and license information
    DisplayLogoLicense()

    # Check if all the packages exist
    st.CheckDependencies()

    # Handle console inputs
    fileinput = 0
    infp = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--":
            # Input commands are supplied through a file.
            inputfname = "./../input/%s" % sys.argv[2]
            case_id = None
            if len(sys.argv) > 3:
                case_id = int(sys.argv[3])
            if os.path.isfile(inputfname):
                infp = open(inputfname, "r")
                fileinput = 1
                if case_id is not None:
                    skip = 0
                    while skip == 0:
                        line = infp.readline().strip("\n").strip(" ").lower()
                        print("\033[2mskipping: {}\033[0m".format(line))
                        if line.lower().replace(" ", "") == ("!!case%d!!" % (case_id)):
                            skip = 1
            else:
                print("\033[2mInput file %s not found.\033[0m" % (inputfname))
        else:
            # The simulations are to be run remotely.
            timestamp = sys.argv[1].strip("\n").strip(" ")
            if len(sys.argv) > 2:
                node = int(sys.argv[2].strip("\n").strip(" "))
            else:
                node = -1
            RemoteExecution(timestamp, node)
            sys.exit(0)

    isquit = 0
    n_empty = 0
    max_empties = 10
    rep = "process"
    channel = np.zeros((4, 4), dtype=np.longdouble)
    qeccode = None
    submit = Submission()

    while isquit == 0:
        if fileinput == 0:
            try:
                print(">>")
                user = list(
                    map(
                        lambda val: val.strip("\n").strip(" "),
                        input().strip(" ").strip("\n").split(" "),
                    )
                )
            except KeyboardInterrupt:
                user = ["quit"]
                print("")
        else:
            command = infp.readline().strip("\n").strip(" ")
            if command[0] == "#":
                continue
            print(">>")
            user = list(map(lambda val: val.strip("\n").strip(" "), command.split(" ")))
            print(command)

        if user[0] == "qcode":
            # define a quantum code
            qeccode = QuantumErrorCorrectingCode(user[1])
            Load(qeccode)

        #####################################################################

        elif user[0] == "qcbasis":
            # display the canonical basis for the code
            IsCanonicalBasis(qeccode.S, qeccode.L, qeccode.T, verbose=1)

        #####################################################################

        elif user[0] == "qcproj":
            # Construct the syndrome projectors
            print("\033[2mConstructing syndrome projectors.\033[0m")
            ConstructSyndromeProjectors(qeccode)
            print("\033[2mDone, saved to code/%s_syndproj.npy.\033[0m" % (qeccode.name))

        #####################################################################

        elif user[0] == "qcminw":
            # prepare a syndrome lookup table for minimum weight decoding
            # Syndrome look-up table for hard decoding.
            print("\033[2mPreparing syndrome lookup table.\033[0m")
            PrepareSyndromeLookUp(qeccode)

        #####################################################################

        elif user[0] == "qcprint":
            # print details of the error correcting code
            PrintQEC(qeccode)

        #####################################################################

        elif user[0] == "chan":
            noiserates = []
            if len(user) > 2:
                noiserates = list(map(np.longdouble, user[2].split(",")))
            for i in range(10):
                channel = ConvertRepresentations(
                    GetKraussForChannel(user[1], *noiserates), "krauss", "process"
                )
                rep = "process"
                print(
                    '\033[2mNote: the current channel is in the "process" representation.\033[0m'
                )
                PrintChan(channel, rep)
                print("\033[2mxxxxxx\033[0m")

        #####################################################################

        elif user[0] == "chsave":
            SaveChan(user[1], channel, rep)

        #####################################################################

        elif user[0] == "chrep":
            channel = np.copy(ConvertRepresentations(channel, rep, user[1]))
            rep = user[1]

        #####################################################################

        elif user[0] == "chtwirl":
            proba = Twirl(channel, rep)
            print(
                "\033[2mTwirled channel\n\tE(R) = %g R + %g X R X + %g Y R Y + %g Z R Z.\033[0m"
                % (proba[0], proba[1], proba[2], proba[3])
            )

        #####################################################################

        elif user[0] == "chpa":
            (proba, proxim) = HonestPauliApproximation(channel, rep)
            print(
                "\033[2mHonest Pauli Approximation\n\tE(R) = %g R + %g X R X + %g Y R Y + %g Z R Z,\n\tand it has a diamond distance of %g from the original channel.\033[0m"
                % (
                    1 - np.sum(proba, dtype=np.float),
                    proba[0],
                    proba[1],
                    proba[2],
                    proxim,
                )
            )

        #####################################################################

        elif user[0] == "chprint":
            PrintChan(channel, rep)
            print("\033[2mxxxxxx\033[0m")

        #####################################################################

        elif user[0] == "chmetrics":
            metrics = user[1].split(",")
            if not (rep == "choi"):
                metvals = ComputeNorms(
                    ConvertRepresentations(channel, rep, "choi"),
                    metrics,
                    {"qcode": qeccode, "chtype": "physical", "corr": 0},
                )
            else:
                metvals = ComputeNorms(
                    channel,
                    metrics,
                    {"qcode": qeccode, "chtype": "physical", "corr": 0},
                )
            print("{:<20} {:<10}".format("Metric", "Value"))
            print("-------------------------------")
            for m in range(len(metrics)):
                print(
                    "{:<20} {:<10}".format(
                        Metrics[metrics[m]]["name"], ("%.2e" % metvals[m])
                    )
                )
            print("xxxxxx")

        #####################################################################

        elif user[0] == "chcalib":
            if len(user) == 3:
                # noise metric, xcol and ycol are not specified.
                out = "fidelity"
                xcol = 0
                ycol = -1
            elif len(user) == 4:
                # xcol and ycol are not provided
                out = user[3]
                xcol = 0
                ycol = -1
            elif len(user) == 5:
                # ycol is not provided
                out = int(user[3])
                xcol = int(user[4])
                ycol = -1
            else:
                out = user[3]
                xcol = int(user[4])
                ycol = int(user[5])
            Calibrate(user[1], user[2], out, xcol=xcol, ycol=ycol)

        #####################################################################

        elif user[0] == "chval":
            IsQuantumChannel(channel, rep)

        #####################################################################

        elif user[0] == "sbprint":
            PrintSub(submit)

        #####################################################################

        elif user[0] == "sbload":
            exists = 0
            if len(user) == 1:
                # no file name or time stamp was provided.
                print("Console input is not set up currently.")
            else:
                submit = Submission()
                exists = LoadSub(submit, user[1], 1, 0)
            if exists == 1:
                if len(submit.scales) == 0:
                    submit.scales = [1 for __ in range(submit.noiserates.shape[1])]
                # Generate new physical channels if needed
                reuse = 1
                for i in range(submit.noiserates.shape[0]):
                    if reuse == 1:
                        if not os.path.isfile(
                            PhysicalChannel(submit, submit.noiserates[i, :])
                        ):
                            reuse = 0
                if reuse == 0:
                    # print("Physical channels not found")
                    # If the simulation is to be run on a cluster, generate input using cluster nodes.
                    if submit.chgen_cluster == 0:
                        PreparePhysicalChannels(submit)
                else:
                    IsComplete(submit)

                submit.nodes = int(
                    np.ceil(
                        submit.noiserates.shape[0]
                        * submit.samps
                        / np.longdouble(submit.cores[0])
                    )
                )
                # else:
                # 	# prepare the set of parameters
                # 	submit.params = np.zeros((submit.noiserates.shape[0] * submit.samps, submit.noiserates.shape[1] + 1), dtype = np.longdouble)
                # 	for i in range(submit.noiserates.shape[0]):
                # 		for j in range(submit.samps):
                # 			submit.params[i * submit.samps + j, :-1] = submit.noiserates[i, :]
                # 			submit.params[i * submit.samps + j, -1] = j
        #####################################################################

        elif user[0] == "chgen":
            PreparePhysicalChannels(submit)

        #####################################################################

        # elif user[0] == "build":
        #     # Compile the modules required for error correction simulations and machine learning
        #     locs = ["simulate", "analyze"]
        #     cython = 0
        #     if len(user) > 2:
        #         if user[2].lower() == "cython":
        #             cython = 1
        #     if len(user) > 1:
        #         if os.path.exists(user[1]) == 1:
        #             locs = [user[1]]
        #     for l in range(len(locs)):
        #         st.BuildExt(locs[l], cython)

        #####################################################################

        elif user[0] == "merge":
            second_submit = Submission()
            exists = LoadSub(second_submit, user[1], 1)
            if exists == 1:
                merged = MergeSubs(submit, second_submit)
                print(
                    "\033[2mMerged dataset is identified with the timestamp %s.\033[0m"
                    % (merged)
                )
            else:
                print(
                    "\033[2mData set with time stamp %s doesn't exist.\033[0m"
                    % (user[1])
                )

        #####################################################################

        elif user[0] == "submit":
            # if not (submit.host in gv.cluster_hosts):
            # Check if physical channels for the existing submission exist.
            old_outdir = submit.outdir
            if len(user) > 1:
                LoadTimeStamp(submit, user[1])
            else:
                LoadTimeStamp(
                    submit,
                    time.strftime("%d/%m/%Y %H:%M:%S")
                    .replace("/", "_")
                    .replace(":", "_")
                    .replace(" ", "_"),
                )
            Save(submit)
            Schedule(submit)
            PrepOutputDir(submit)
            if submit.phychans.size == 0:
                os.system("cp %s/physical/* %s/physical/" % (old_outdir, submit.outdir))
            else:
                # Write the physical channels to folder
                SavePhysicalChannels(submit)
            # Prepare decoder knowledge and save
            # if the decoder needs partial information, then the decoder knowledge needs to be prepared.
            if submit.host == "local":
                print(
                    '\033[2mFor remote execution, run: "./chflow.sh %s"\033[0m'
                    % (submit.timestamp)
                )
            else:
                if submit.host in gv.cluster_info:
                    if not os.path.exists("./../input/%s" % (submit.host)):
                        os.mkdir("./../input/%s" % (submit.host))

                    try:
                        from cluster import cluster as cl

                        # Create pre-processing and post-processing scripts if this is for cluster.
                        if submit.chgen_cluster == 1:
                            cl.CreatePreBatch(submit)
                        cl.CreateLaunchScript(submit)
                        cl.CreatePostBatch(submit)
                        cl.Usage(submit)
                    except:
                        print(
                            "\033[2mError importing cluster.py. Ensure CreateLaunchScript(<submission object>) and Usage(<submission object>) are defined.\033[0m"
                        )
                else:
                    print(
                        '\033[2mNo submission rules for "%s". See wiki for details.\033[0m'
                        % (submit.host)
                    )

        #####################################################################

        elif user[0] == "collect":
            # Collect the logical failure rates into one file.
            if submit.complete > 0:
                GatherLogErrData(submit)

        #####################################################################

        elif user[0] == "tplot":
            # Produce threshold plots for a particular logical metric.
            # Plot the logical error rate with respect to the concatnation layers, with a new curve for every physical noise rate.
            # At the threshold in the physical noise strengh, the curves will have a bifurcation.
            ThresholdPlot(user[1], user[2], submit)

        #####################################################################

        elif user[0] == "lplot":
            # Plot the logical error rate with respect to a physical noise strength, with a new figure for every concatenation layer.
            # One or more simulation data can be plotted in the same figure with a new curve for every dataset.
            # One of more measures of physical noise strength can be plotted on the same figure with a new curve for each definition.

            if len(user) >= 3:
                dbses = [submit]
                thresholds = []
                nbins = 10
                if len(user) > 6:
                    for (i, ts) in enumerate(user[6].split(",")):
                        dbses.append(Submission())
                        LoadSub(dbses[i + 1], ts, 0)
                check = 1
                for d in range(len(dbses)):
                    IsComplete(dbses[d])
                    if dbses[d].complete > 0:
                        if not os.path.isfile(
                            LogicalErrorRates(dbses[d], user[2], fmt="npy")
                        ):
                            GatherLogErrData(dbses[d])
                    else:
                        check = 0
                        break
                if len(user) > 5:
                    nbins = int(user[5])
                if len(user) > 4:
                    upper_cutoffs = np.array(list(map(np.double, user[4].split(","))))
                    upper_cutoffs = np.repeat(
                        upper_cutoffs, submit.levels // upper_cutoffs.shape[0]
                    )
                if len(user) > 3:
                    lower_cutoffs = np.power(
                        0.1, list(map(np.double, user[3].split(",")))
                    )
                    lower_cutoffs = np.repeat(
                        lower_cutoffs, submit.levels // lower_cutoffs.shape[0]
                    )
                for c in range(upper_cutoffs.shape[0]):
                    thresholds.append(
                        {"upper": upper_cutoffs[c], "lower": lower_cutoffs[c]}
                    )
                if len(thresholds) == 0:
                    thresholds = [
                        {"lower": 1e-9, "upper": 1e-2} for __ in submit.levels
                    ]
                if check == 1:
                    phylist = list(map(lambda phy: phy.strip(" "), user[1].split(",")))
                    LevelWisePlot(
                        phylist,
                        user[2],
                        dbses,
                        1,  # inset flag
                        int(("dnorm" in phylist) and (len(dbses) > 1)),  # flow lines
                        nbins,
                        thresholds,
                    )
                else:
                    print(
                        "\033[2mOne of the databases does not have logical error data.\033[0m"
                    )
            else:
                print("\033[2mUsage: %s\033[0m" % mannual["lplot"][1])

        #####################################################################

        elif user[0] == "dcplot":
            # No documentation yet.
            # Compare decoders.
            pmet = user[1]
            lmet = user[2]
            dbses = [submit]
            if len(user) > 3:
                for (i, ts) in enumerate(user[3].split(",")):
                    dbses.append(Submission())
                    LoadSub(dbses[i + 1], ts, 0)
                    IsComplete(dbses[i + 1])
                    if not os.path.isfile(
                        LogicalErrorRates(dbses[i + 1], lmet, fmt="npy")
                    ):
                        GatherLogErrData(dbses[i + 1])
            nbins = dbses[0].noiserates.shape[0]
            if len(user) > 4:
                nbins = int(user[4])
            DecoderCompare(pmet, lmet, dbses, nbins=nbins)

        #####################################################################

        elif user[0] == "dciplot":
            # No documentation yet.
            # Compare decoders.
            pmet = user[1]
            lmet = user[2]
            dbses = [submit]
            if len(user) > 3:
                for (i, ts) in enumerate(user[3].split(",")):
                    dbses.append(Submission())
                    LoadSub(dbses[i + 1], ts, 0, 0)
                    IsComplete(dbses[i + 1])
                    if not os.path.isfile(
                        LogicalErrorRates(dbses[i + 1], lmet, fmt="npy")
                    ):
                        GatherLogErrData(dbses[i + 1])
            chids = [0]
            if len(user) > 4:
                chids = list(map(int, user[4].split(",")))
            DecoderInstanceCompare(pmet, lmet, dbses, chids)

        #####################################################################

        elif user[0] == "dvplot":
            # No documentation provided yet
            phymet = user[1].strip(" ")
            logmet = user[2]
            dbses = [submit]
            if len(user) > 3:
                for (i, ts) in enumerate(user[3].split(",")):
                    dbses.append(Submission())
                    LoadSub(dbses[i + 1], ts, 0)
                    IsComplete(dbses[i + 1])
                    if not os.path.isfile(
                        LogicalErrorRates(dbses[i + 1], logmet, fmt="npy")
                    ):
                        GatherLogErrData(dbses[i + 1])
            thresholds = []
            nbins = 10
            if len(user) > 6:
                nbins = int(user[6])
            if len(user) > 5:
                upper_cutoffs = np.array(list(map(np.double, user[5].split(","))))
                upper_cutoffs = np.repeat(
                    upper_cutoffs, submit.levels // upper_cutoffs.shape[0]
                )
            if len(user) > 4:
                lower_cutoffs = np.power(0.1, list(map(np.double, user[4].split(","))))
                lower_cutoffs = np.repeat(
                    lower_cutoffs, submit.levels // lower_cutoffs.shape[0]
                )
            for c in range(upper_cutoffs.shape[0]):
                thresholds.append(
                    {"upper": upper_cutoffs[c], "lower": lower_cutoffs[c]}
                )
            if len(thresholds) == 0:
                thresholds = [{"lower": 1e-9, "upper": 1e-2} for __ in submit.levels]
            PlotDeviationYX(logmet, phymet, dbses, nbins, thresholds)

        #####################################################################

        elif user[0] == "hamplot":
            # No documentation provided yet
            pmets = user[1].strip(" ").split(",")
            logmet = user[2]
            thresholds = []
            nbins = 10
            if len(user) > 6:
                nbins = int(user[6])
            if len(user) > 5:
                upper_cutoffs = np.array(list(map(np.double, user[5].split(","))))
                upper_cutoffs = np.repeat(
                    upper_cutoffs, submit.levels // upper_cutoffs.shape[0]
                )
            if len(user) > 4:
                lower_cutoffs = np.power(0.1, list(map(np.double, user[4].split(","))))
                lower_cutoffs = np.repeat(
                    lower_cutoffs, submit.levels // lower_cutoffs.shape[0]
                )
            for c in range(upper_cutoffs.shape[0]):
                thresholds.append(
                    {"upper": upper_cutoffs[c], "lower": lower_cutoffs[c]}
                )
            if len(thresholds) == 0:
                thresholds = [{"lower": 1e-9, "upper": 1e-2} for __ in submit.levels]
            # Load the other database
            dbs_other = Submission()
            LoadSub(dbs_other, user[3], 0)
            # Load the logical error rates
            IsComplete(dbs_other)
            if dbs_other.complete > 0:
                if not os.path.isfile(LogicalErrorRates(dbs_other, logmet, fmt="npy")):
                    GatherLogErrData(dbs_other)
            else:
                check = 0
                break
            DoubleHammerPlot(logmet, pmets, [submit, dbs_other], 1, nbins, thresholds)

        #####################################################################

        elif user[0] == "pdplot":
            # No documentation provided yet
            nrate = list(map(np.double, user[1].split(",")))
            nreps = int(user[2])
            max_weight = int(user[3])
            sample = -1
            if len(user) > 4:
                sample = int(user[4])
            if os.path.exists(RawPhysicalChannel(submit, nrate)):
                if sample == -1:
                    chi = np.mean(np.load(RawPhysicalChannel(submit, nrate)), axis=0)
                else:
                    chi = np.load(RawPhysicalChannel(submit, nrate))[sample]
            else:
                CreatePauliDistChannels(submit)
            if submit.iscorr == 0:
                pauliprobs = np.diag(chi.reshape([4, 4]))
            elif submit.iscorr == 1:
                pauliprobs = chi
            else:
                pauliprobs = list(map(np.diag, chi.reshape([submit.eccs[0].N, 4, 4])))
            if submit.eccs[0].PauliOperatorsLST is None:
                PrepareSyndromeLookUp(submit.eccs[0])
            dist = GetErrorProbabilities(
                submit.eccs[0].PauliOperatorsLST, pauliprobs, submit.iscorr
            )
            PauliDistributionPlot(
                submit.eccs[0],
                dist,
                nreps=nreps,
                max_weight=max_weight,
                outdir=submit.outdir,
                channel=submit.channel,
            )

        #####################################################################

        elif user[0] == "cplot":
            # No documentation provided
            thresholds = {"y": 10e-16, "x": 10e-16}
            dbses = [submit]
            if len(user) > 5:
                thresholds["x"] = np.power(0.1, int(user[5]))
            if len(user) > 4:
                thresholds["y"] = np.power(0.1, int(user[4]))

            if len(user) > 3:
                for (i, ts) in enumerate(user[3].split(",")):
                    dbses.append(Submission())
                    LoadSub(dbses[i + 1], ts, 0)
            check = 1
            for d in range(len(dbses)):
                IsComplete(dbses[d])
                if dbses[d].complete > 0:
                    if not os.path.isfile(
                        LogicalErrorRates(dbses[d], user[2], fmt="npy")
                    ):
                        GatherLogErrData(dbses[d])
                else:
                    check = 0
                    break
            if check == 1:
                include = np.sort(
                    np.concatenate(
                        (
                            np.arange(0, dbses[0].channels, dbses[0].samps),
                            np.arange(1, dbses[0].channels, dbses[0].samps),
                        )
                    )
                )
                ChannelWisePlot(
                    user[1],
                    user[2],
                    dbses,
                    thresholds=thresholds,
                    include_input=include,
                )
            else:
                print(
                    "\033[2mOne of the databases does not have logical error data.\033[0m"
                )

        #####################################################################

        elif user[0] == "lplot2d":
            # Plot the logical error rates with respect to two parameters of the physical noise rate.
            # The plot will be a 2D density plot with the logical error rates represented by the density of the colors.
            print("\033[2m"),
            LevelWisePlot2D(user[1], user[2], submit)
            print("\033[0m"),

        #####################################################################

        elif user[0] == "mcplot":
            # Plot the average logical error rate with the number of syndrome samples.
            # This plot is only useful for knowing the rate of convergence of the average as a function of the number of syndrome samples.
            # While specifying more than one database, care must be taken to ensure that all the databases have the same
            # physical noise process
            # number of concatenation levels
            # To use this feature, submit.stats must be a list.
            pmet = user[1]
            lmet = user[2]
            nrates = min(3, submit.noiserates.shape[0])
            nsamps = 10
            dbses = [submit]
            if len(user) > 5:
                for (i, ts) in enumerate(user[5].split(",")):
                    dbses.append(Submission())
                    LoadSub(dbses[i + 1], ts, 0, 0)
                    IsComplete(dbses[i + 1])
                    if not os.path.isfile(
                        LogicalErrorRates(dbses[i + 1], lmet, fmt="npy")
                    ):
                        GatherLogErrData(dbses[i + 1])
            if len(user) > 4:
                nsamps = int(user[4])
            if len(user) > 3:
                rate_range = list(map(int, user[3].split(",")))
            # rates = submit.noiserates[
            #     np.sort(np.random.choice(submit.noiserates.shape[0], nrates))[::-1], :
            # ]
            rates = submit.noiserates[np.arange(*rate_range), :]
            MCStatsPlot(dbses, lmet, pmet, rates, nsamples=nsamps)

        #####################################################################

        elif user[0] == "bplot":
            # Plot the number of syndromes counted for a particular syndrome probability and a conditional logical error.
            # Produce a 2D plot where X axis is the syndrome probability and Y axis the conditional logical error.
            # A plot is produced for every concatenation level.
            # The number of syndromes counted for fixed values of the X and Y values is represented by the intensity of a color.
            # The plot is produced for the currently loaded database.
            # The logical metric should be supplied, else it is taken as the first available metric for the database.
            # If no physical noise rate and sample is specified, a 2D plot as described above is produced for every channel in the database.
            if len(user) > 2:
                pvals = np.array(
                    map(np.longdouble, user[2].split(",")), dtype=np.longdouble
                )
                lmet = user[1]
            else:
                pvals = -1
                if len(user) > 1:
                    lmet = user[1]
                else:
                    lmet = submit.metrics[0]
            BinsPlot(submit, lmet, pvals)

        #####################################################################

        elif user[0] == "varplot":
            # Plot the variance in a scatter plot (of the logical error rates vs. physical metrics)
            # PlotBinVariance(submit, lmet, pmet, nbins = 10)
            # varplot <pmet> <lment> <number of bins> <additional database time-stamps>
            # default number of bins is 10.
            dbses = [submit]
            check = 1
            maxlev = 0
            if len(user) > 4:
                for (i, ts) in enumerate(user[4].split(",")):
                    dbses.append(Submission())
                    LoadSub(dbses[i + 1], ts, 0)
                    IsComplete(dbses[i + 1])
                    if dbses[i + 1].complete > 0:
                        GatherLogErrData(dbses[i + 1])
                    else:
                        check = 0
                        break
                pmet = user[1]
                lmet = user[2]
                nbins = int(user[3])
            else:
                if len(user) > 3:
                    pmet = user[1]
                    lmet = user[2]
                    nbins = int(user[3])
                else:
                    nbins = 10
                    if len(user) > 2:
                        pmet = user[1]
                        lmet = user[2]
                    else:
                        lmet = submit.metrics[0]
                        if len(user) > 1:
                            pmet = user[1]
                        else:
                            pmet = 0
            if check == 1:
                PlotBinVariance(dbses, lmet, pmet, nbins)
            else:
                print(
                    "\033[2mOne of the databases does not have logical error data up to %d levels.\033[0m"
                    % (maxlev)
                )

        #####################################################################

        elif user[0] == "sbselect":
            # No documentation provided
            Select(submit, list(map(int, user[1].split(","))))

        #####################################################################

        elif user[0] == "sbsave":
            Save(submit)
            Schedule(submit)
            PrepOutputDir(submit)
            # Save the physical channels
            if submit.phychans is not None:
                SavePhysicalChannels(submit)

        #####################################################################

        elif user[0] == "sbtwirl":
            TwirlChannels(submit)

        #####################################################################

        elif user[0] == "pmetrics":
            # compute level-0 metrics.
            if len(user) > 1:
                physmetrics = user[1].split(",")
                ComputeMetrics(submit, physmetrics, chtype="physical")
            else:
                print("\033[2mUsage: %s\033[0m" % mannual["pmetrics"][1])
        #####################################################################

        elif user[0] == "lmetrics":
            # compute metrics of the average logical channel.
            if len(user) > 1:
                logmetrics = user[1].split(",")
                ComputeMetrics(submit, logmetrics, chtype="logical")
                # submit.metrics = list(set().union(submit.metrics, logmetrics))
                GatherLogErrData(submit, additional=logmetrics)

            else:
                print("\033[2mUsage: %s\033[0m" % mannual["lmetrics"][1])

        #####################################################################

        elif user[0] == "lpmetrics":
            # compute metrics of the average logical channel.
            if len(user) > 1:
                phylogmetrics = user[1].split(",")
                ComputeMetrics(submit, phylogmetrics, chtype="phylog")
            else:
                print("\033[2mUsage: %s\033[0m" % mannual["lpmetrics"][1])

        #####################################################################

        elif user[0] == "threshold":
            if user[1] in submit.metrics:
                thresh = ComputeThreshold(submit, user[1])
                print(
                    "The estimated threshold for the %s code over the %s channel is %g."
                    % (
                        " X ".join(
                            [submit.eccs[i].name for i in range(len(submit.eccs))]
                        ),
                        submit.channel,
                        thresh,
                    )
                )
            else:
                print(
                    "Only %s are computed at the logical levels."
                    % (", ".join([Metrics[met]["name"] for met in submit.metrics]))
                )

        #####################################################################

        elif user[0] == "compare":
            tocompare = [Submission(), Submission()]
            for i in range(2):
                LoadSub(tocompare[i], user[1 + i])
            if (user[3] in tocompare[0].metrics) and (user[3] in tocompare[1].metrics):
                if os.path.isfile(
                    GatheredLogErrData(tocompare[0], logicalmetric)
                ) and os.path.isfile(GatheredLogErrData(tocompare[1], logicalmetric)):
                    CompareSubs(tocompare, user[3])
                else:
                    print(
                        "Logical error rates data is not available for one of the simulations."
                    )
            else:
                print(
                    "The logical metrics that are available in both simulation data are %s."
                    % (
                        ", ".join(
                            [
                                Metrics[met]["name"]
                                for met in tocompare[0].metrics
                                if met in tocompare[1].metrics
                            ]
                        )
                    )
                )

        #####################################################################

        elif user[0] == "sbfit":
            # fit the logical error rates to an ansatz
            # if there are two outputs, the first is the logical metric and the second is a list of databases
            lmet = submit.metrics[0]
            pmet = lmet
            dbses = [submit]
            check = 1
            if len(user) > 3:
                for (i, ts) in enumerate(user[3].split(",")):
                    refs.append(Submission())
                    LoadSub(dbses[i + 1], ts, 0)
                    IsComplete(dbses[i + 1])
                    if dbses[i + 1].complete > 0:
                        GatherLogErrData(dbses[i + 1])
                    else:
                        check = 0
            if len(user) > 2:
                lmet = user[2]
            if len(user) > 1:
                pmet = user[1]

            if (
                (os.path.isfile(FitPhysRates(submit, lmet)) == 1)
                and (os.path.isfile(FitWtEnums(submit, lmet)) == 1)
                and (os.path.isfile(FitExpo(submit, lmet)) == 1)
            ):
                CompareAnsatzToMetrics(submit, pmet, lmet)
            else:
                if check == 1:
                    # os.system("cd analyze/;python compile.py build_ext --inplace > compiler_output.txt 2>&1;cd ..")
                    FitPhysErr(pmet, lmet, *dbses)
                    CompareAnsatzToMetrics(submit, pmet, lmet)
                else:
                    print(
                        "\033[2mSome of the databases do not have simulation data.\033[0m"
                    )

        #####################################################################

        elif user[0] == "compress":
            # Compute a compression matrix for the channel parameters such that compressed parameters vary whenever the logical error rates vary.
            # If the compression matrix exists, simply compute the bin variance.
            # compress level [lmet] [ncomp] [nbins]
            lmet = submit.metrics[0]
            ncomp = 3
            nbins = 10
            check = int(submit.complete > 0)
            if len(user) > 4:
                level = np.int(user[1])
                lmet = user[2]
                ncomp = np.int(user[3])
                nbins = np.int(user[4])
            else:
                if len(user) > 3:
                    level = np.int(user[1])
                    lmet = user[2]
                    ncomp = np.int(user[3])
                else:
                    if len(user) > 2:
                        level = np.int(user[1])
                        lmet = user[2]
                    else:
                        if len(user) > 1:
                            level = np.int(user[1])
                        else:
                            check = 0

            if check == 1:
                if os.path.isfile(CompressedParams(submit, lmet, level)):
                    # Compute the bin averages
                    # xdata = compressed parameters
                    # ydata = logical error rates
                    xdata = np.load(CompressedParams(submit, lmet, level))
                    ydata = np.load(LogicalErrorRates(submit, lmet))[:, level]
                    ComputeNDimBinVariance(xdata, ydata, nbins, space="linear")
                else:
                    Compress(submit, lmet, level, ncomp)
            else:
                print(
                    "\033[2mUsage: compress level [lmet] [nbins]. Ensure that the logical error data exists for all levels.\033[0m"
                )

        #####################################################################

        elif user[0] == "synclog":
            # Add the physical error rates to the level-0 logical error data.
            # This is only relevant for backward compatibility since earlier the logical error rates at level-0 were not computed and left as 0.
            AddPhysicalRates(submit)

        #####################################################################

        elif user[0] == "sblearn":
            # learn physical noise rates from fit data
            # sblearn <timestamp> <physical metrics> <logical metric> [method] [<mask>]
            sbtest = Submission()
            LoadSub(sbtest, user[1], 0)
            IsComplete(sbtest)
            if sbtest.complete > 0:
                GatherLogErrData(sbtest)

            if os.path.isfile(PredictedPhyRates(sbtest)) == 1:
                ValidatePrediction(sbtest, user[2], user[3])
            else:
                mask = np.zeros((4, 4), dtype=np.int8)
                mask[:, 1:] = 1
                if len(user) > 5:
                    if user[5] in Channels:
                        noiserates = np.random.rand(len(Channels[user[5]]["params"]))
                        mask[
                            np.nonzero(
                                ConvertRepresentations(
                                    GetKraussForChannel(user[5], *noiserates),
                                    "krauss",
                                    "process",
                                )
                                > 10e-10
                            )
                        ] = 1
                        mask[0, 0] = 0
                method = "mplr"
                if len(user) > 4:
                    method = user[4]
                # os.system("cd analyze/;python compile.py build_ext --inplace > compiler_output.txt 2>&1;cd ..")
                # prepare training set
                PrepareMLData(submit, user[2].split(","), user[3], 0, mask)
                # prepare testing set
                PrepareMLData(sbtest, user[2].split(","), user[3], 1, mask)
                # predict using machine learning
                Predict(sbtest, submit, method)
                # validate machine learning predictions using a plot
                ValidatePrediction(sbtest, user[2].split(",")[0], user[3])

        #####################################################################

        elif user[0] == "man":
            if len(user) > 1:
                if user[1] in mannual:
                    print(
                        '\t\033[2m"%s"\n\tDescription: %s\n\tUsage:\n\t%s\033[0m'
                        % (user[1], mannual[user[1]][0], mannual[user[1]][1])
                    )
                elif user[1] in Channels:
                    print(
                        '\t\033[2m"%s"\n\tDescription: %s\n\tParameters: %s\033[0m'
                        % (
                            user[1],
                            Channels[user[1]]["desc"],
                            Channels[user[1]]["params"],
                        )
                    )
                elif user[1] in Metrics:
                    print(
                        '\t\033[2m"%s"\n\tDescription: %s\n\tExpression: %s\033[0m'
                        % (user[1], Metrics[user[1]]["name"], Metrics[user[1]]["latex"])
                    )
                else:
                    # Try to auto complete and ask for possible suggestions.
                    similar = [entry for entry in mannual if user[1] in entry]
                    if len(similar) > 0:
                        print("\033[2mDid you mean %s ?\033[0m" % (", ".join(similar)))
            else:
                print("\tList of commands and thier functions.")
                for (i, item) in enumerate(mannual):
                    print(
                        '\t%d). \033[2m"%s"\n\tDescription: %s\n\tUsage\n\t%s\033[0m'
                        % (i, item, mannual[item][0], mannual[item][1])
                    )
            print("xxxxxx")

        #####################################################################

        elif user[0] == "clean":
            git = 0
            if len(user) > 1:
                if user[1] == "git":
                    git = 1
            Clean(git)

        #####################################################################

        elif user[0] in ["quit", "exit"]:
            isquit = 1

        #####################################################################

        elif user[0] == "notes":
            # Save a page of the plot file as a PDF in a folder.
            # No documentation provided yet
            phymet = user[1]
            logmet = user[2]
            name = user[3]
            plot_option = user[4]
            notes_location = user[5]
            pages = list(map(int, user[6].split(",")))
            if plot_option == "lplot":
                plot_file = LevelWise(submit, phymet.replace(",", "_"), logmet)
            elif plot_option == "cplot":
                plot_file = ChannelWise(submit, phymet, logmet)
            elif plot_option == "pdplot":
                plot_file = PauliDistribution(submit.outdir, submit.channel)
            elif plot_option == "hamplot":
                plot_file = HammerPlot(submit, logmet, phymet.split(","))
            elif plot_option == "dvplot":
                plot_file = DeviationPlotFile(submit, phymet, logmet)
            elif plot_option == "mcplot":
                plot_file = MCStatsPlotFile(submit, logmet, phymet.split(",")[0])
            elif plot_option == "dciplot":
                plot_file = DecodersInstancePlot(submit, phymet.split(",")[0], logmet)
            else:
                print("\033[2mUnknown plot option %s.\033[0m" % (plot_option))
                continue
            # file_location = "%s/%s" % (submit.outdir, plot_file)
            # print(
            #     "pdfseparate -f %d -l %d %s %s_%s_%s_%s.pdf"
            #     % (page, page, file_location, plot_option, name, phymet, logmet)
            # )
            information = [{"fname": plot_file, "start": pg, "end": pg} for pg in pages]
            output_file = "%s_%s_%s_%s.pdf" % (plot_option, name, phymet, logmet)
            # print(
            #     "ExtractPDFPages({}, {}, {})".format(
            #         information, notes_location, output_file
            #     )
            # )
            ExtractPDFPages(information, notes_location, output_file)

        #####################################################################

        else:
            print("\033[2mNo action.\033[0m")
            similar = [entry for entry in mannual if user[0] in entry]
            if len(similar) > 0:
                print("\033[2mDid you mean %s ?\033[0m" % (", ".join(similar)))
            n_empty = n_empty + 1
            if n_empty > max_empties:
                isquit = 1

    if fileinput == 1:
        infp.close()
