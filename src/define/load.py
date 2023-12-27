import os
import numpy as np
import itertools as it
from define import qchans as qc
from define import qcode as qec
from define.utils import LoadTimeStamp, IsNumber
from define import fnames as fn
from define import globalvars as gv


def Update(submit, pname, newvalue, lookup_load=1):
    # Update the parameters to be submitted
    if pname == "timestamp":
        LoadTimeStamp(submit, newvalue)

    elif pname == "ecc":
        submit.isSubmission = 1
        # Read all the Parameters of the error correcting code
        names = newvalue.split(",")
        submit.ecfiles = []
        submit.levels = len(names)
        submit.decoders = np.zeros(submit.levels, dtype=np.int64)
        for i in range(submit.levels):
            submit.eccs.append(qec.QuantumErrorCorrectingCode(names[i]))
            qec.Load(submit.eccs[i], lookup_load)
            submit.ecfiles.append(submit.eccs[i].defnfile)
            submit.decoders[i] = 0

    elif pname == "decoder":
        decoder_info = newvalue.split(",")
        submit.decoders = np.zeros(submit.levels, dtype=np.int64)
        for l in range(len(decoder_info)):
            submit.decoders[l] = int(decoder_info[l])

    elif pname == "dcfraction":
        submit.decoder_fraction = float(newvalue)

    elif pname == "channel":
        submit.channel = newvalue
        submit.iscorr = qc.Channels[submit.channel]["corr"]
        # print("submit.iscorr = {}".format(submit.iscorr))

    elif pname == "chtype":
        qc.Channels[submit.channel]["Pauli"] = newvalue

    elif pname == "rc":
        submit.rc = int(newvalue)

    elif pname == "repr":
        submit.repr = newvalue

    elif pname == "noiserange":
        # There are 3 ways of providing the noise range.
        # Using a file: The file must have as many columns as the number of free parameters and placed in chflow/input/.
        # For each free parameter:
        # 2. Using a compact range specification: low,high,number of steps.
        # 3. Explicity specifying the points to be sampled: list of (length not equal to 3) points.
        # Note that 2 or fewer points can be specified using the first scheme.
        # The value for different free parameters must be separated by a ";".
        # If scale is not equal to 1, the noise rates values is interpretted as an exponent for the value of scale.
        if os.path.isfile("%s" % (newvalue)) == 1:
            submit.ratesfile = newvalue
            submit.noiserates = np.loadtxt(
                "%s" % (newvalue), comments="#", dtype=np.longdouble
            )
            submit.noiserange = None
        else:
            newRanges = list(
                map(
                    lambda arr: list(map(np.longdouble, arr.split(","))),
                    newvalue.split(";"),
                )
            )
            # print("Ranges\n{}".format(newRanges))
            submit.noiserange = []
            for i in range(len(newRanges)):
                if len(newRanges[i]) == 3:
                    submit.noiserange.append(
                        np.linspace(
                            np.longdouble(newRanges[i][0]),
                            np.longdouble(newRanges[i][1]),
                            np.int64(newRanges[i][2]),
                        )
                    )
                else:
                    submit.noiserange.append(newRanges[i])
            submit.noiserates = np.array(
                list(map(list, it.product(*submit.noiserange))), dtype=np.float64
            )

    elif pname == "samples":
        # The value must be an integer
        submit.samps = int(newvalue)

    elif pname == "frame":
        # The value must be an integer
        submit.frame = submit.eccframes[newvalue]

    elif pname == "filter":
        # The input is a filtering crieterion described as (metric, lower bound, upper bound).
        # Only the channels whose metric value is between the lower and the upper bounds are to be selected.
        if newvalue == "":
            submit.filter["metric"] = "fidelity"
            submit.filter["lower"] = 0
            submit.filter["upper"] = 1
        else:
            filterDetails = newvalue.split(",")
            submit.filter["metric"] = filterDetails[0]
            submit.filter["lower"] = np.float64(filterDetails[1])
            submit.filter["upper"] = np.float64(filterDetails[2])

    elif pname == "stats":
        # The value can be
        # an integer
        # an explicit range of numbers -- [<list of values separated by commas>]
        # compact range of numbers -- lower,upper,number_of_steps
        if IsNumber(newvalue) == 1:
            submit.stats = np.array([int(newvalue)], dtype=np.int64)
        else:
            if "[" in newvalue:
                submit.stats = np.array(
                    list(map(int, newvalue[1:-1].split(","))), dtype=np.int64
                )
            else:
                submit.stats = np.geomspace(
                    *np.array(list(map(int, newvalue.split(","))), dtype=np.int64),
                    dtype=np.int64
                )
        submit.isSubmission = 1

    elif pname == "metrics":
        # The metrics to be computed at the logical level
        submit.metrics = newvalue.split(",")

    elif pname == "wall":
        submit.isSubmission = 1
        submit.wall = int(newvalue)

    elif pname == "importance":
        submit.isSubmission = 1
        submit.importance = submit.samplingOptions[newvalue.lower()]

    elif pname == "hybrid":
        submit.isSubmission = 1
        submit.hybrid = int(newvalue)

    elif pname == "decbins":
        # Set the bins of channels that must be averged at intermediate levels
        # The bins shall be provided in a new file, whose name must be specified here.
        submit.isSubmission = 1
        submit.hybrid = 1
        ParseDecodingBins(submit, newvalue)

    elif pname == "job":
        submit.isSubmission = 1
        submit.job = newvalue

    elif pname == "host":
        submit.isSubmission = 1
        submit.host = newvalue
        submit.chgen_cluster = int(submit.host.lower() in gv.cluster_info)

    elif pname == "queue":
        submit.isSubmission = 1
        submit.queue = newvalue

    elif pname == "cores":
        submit.cores = list(map(int, newvalue.split(",")))

    elif pname == "nodes":
        submit.isSubmission = 1
        submit.nodes = int(newvalue)

    elif pname == "email":
        submit.email = newvalue

    elif pname == "account":
        submit.account = newvalue

    elif pname == "scheduler":
        submit.scheduler = newvalue

    elif pname == "outdir":
        submit.outdir = fn.OutputDirectory(newvalue, submit)

    elif pname == "scale":
        submit.scales = np.array(
            list(map(np.longdouble, newvalue.split(","))), dtype=np.longdouble
        )

    elif pname == "plot":
        settings_info = [sg.split(",") for sg in newvalue.split(";")]
        for i in range(len(settings_info)):
            if settings_info[i][0] not in submit.plotsettings:
                submit.plotsettings.update({settings_info[i][0]: None})
            submit.plotsettings[settings_info[i][0]] = settings_info[i][1]
    else:
        pass

    return None


def LoadSub(submit, subid, isgen, lookup_load=1):
    # Load the parameters of a submission from an input file
    # If the input file is provided as the submission id, load from that input file.
    # Else if the time stamp is provided, search for the corresponding input file and load from that.
    LoadTimeStamp(submit, subid)
    if os.path.exists(submit.inputfile):
        with open(submit.inputfile, "r") as infp:
            for (lno, line) in enumerate(infp):
                if line[0] == "#":
                    pass
                else:
                    linecontents = line.strip("\n").strip(" ").split(" ")
                    (variable, value) = (linecontents[0], " ".join(linecontents[1:]))
                    Update(
                        submit,
                        variable.strip("\n").strip(" "),
                        value.strip("\n").strip(" "),
                        lookup_load,
                    )
        return 1
    else:
        print("\033[2mInput file %s not found.\033[0m" % (submit.inputfile))
    return 0


def ParseDecodingBins(submit, fname):
    # Read the channels which must be averaged while decoding the intermediate concatenation levels
    submit.decoder_type = fname
    if os.path.isfile(fname):
        # Read the bins information form a file.
        # The file should contain one line for each concatenation level.
        # Each line should contain N columns (separated by spaces) where N is the number of encoded qubits at that level.
        # Every column entry should be an index of a bin into which that channel must be placed. Channels in the same bin are averaged.
        with open(fname, "r") as bf:
            for l in range(submit.levels):
                submit.decoderbins.append(
                    list(map(int, bf.readline().strip("\n").strip(" ").split(" ")))
                )
    else:
        chans = [
            np.prod(
                [submit.eccs[submit.levels - l - 1].N for l in range(inter)],
                dtype=np.int64,
            )
            for inter in range(submit.levels + 1)
        ][::-1]
        if fname == "soft":
            # Choose to put every channel in its own bin -- this is just soft decoding.
            for l in range(submit.levels):
                submit.decoderbins.append(np.arange(chans[l], dtype=np.int64))
        elif "random" in fname:
            # Choose to have random bins
            n_rand_decbins = int(fname.split("_")[1])
            if n_rand_decbins < 1:
                # print("\033[2mNumber of bins for channels cannot be less than 1, resetting this number to 1.\033[0m")
                n_rand_decbins = 1
            for l in range(submit.levels):
                # print("l: {}, chans[l] = {}".format(l, chans[l]))
                submit.decoderbins.append(
                    np.random.randint(
                        0, min(n_rand_decbins, chans[l] - 1), size=chans[l]
                    )
                )
            # print("decoder bins: {}".format(submit.decoderbins))
        elif fname == "hard":
            # All channels in one bin -- this is hard decoding.
            for l in range(submit.levels):
                submit.decoderbins.append(np.zeros(chans[l], dtype=np.int64))
        else:
            pass
    return None
