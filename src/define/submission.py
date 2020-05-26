import os
import sys
import time
from shutil import copyfile
import numpy as np
import itertools as it
from define import fnames as fn
from define import qcode as qec
from define import qchans as qc


class Submission:
    def __init__(self):
        # Logging options
        self.timestamp = (
            time.strftime("%d/%m/%Y %H:%M:%S")
            .replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )
        # Output options
        self.outdir = "."
        # Cluster options
        self.job = "X"
        self.host = "local"
        self.nodes = 0
        self.wall = 0
        self.params = np.array([1, 0], dtype=float)
        self.queue = "X"
        self.email = "X"
        self.account = "default"
        # Run time options
        self.cores = [1, 1]
        self.inputfile = fn.SubmissionInputs(self)
        self.isSubmission = 0
        self.scheduler = fn.SubmissionSchedule(self)
        self.complete = -1
        # Channel options
        self.channel = "X"
        self.iscorr = 0
        self.phychans = np.array([])
        self.rawchans = None
        self.repr = "process"
        self.noiserange = []
        self.noiserates = np.array([])
        self.scales = []
        self.samps = 1
        self.channels = 0
        self.available = np.array([])
        self.overwrite = 0
        # Metrics options
        self.metrics = ["frb"]
        self.filter = {"metric": "fidelity", "lower": 0, "upper": 1}
        # ECC options
        self.eccs = []
        self.eccframes = {"P": 0, "C": 1, "PC": 2}
        self.frame = 0
        self.levels = 0
        self.ecfiles = []
        # Decoder
        self.decoders = []
        self.decoder_type = "default_soft"
        self.hybrid = 0
        self.decoderbins = []
        self.ndecoderbins = []
        # Sampling options
        self.stats = np.array([])
        self.samplingOptions = {"direct": 0, "power": 1, "bravyi": 2}
        self.importance = 0
        self.nbins = 50
        self.maxbin = 50
        # Advanced options
        self.isAdvanced = 0
        # Plot settings -- color, marker, linestyle
        self.plotsettings = {}
        # {"color": "k", "marker": "o", "linestyle": "None"}
        # Randomized compiling of quantum gates
        self.rc = 0
        # Miscellaneous information
        self.misc = "None"


def IsNumber(numorstr):
    # test if the input is a number.
    try:
        float(numorstr)
        return 1
    except:
        return 0


def ChangeTimeStamp(submit, timestamp):
    # change the timestamp of a submission and all the related values to the timestamp.
    submit.timestamp = timestamp
    # Re define the variables that depend on the time stamp
    submit.outdir = fn.OutputDirectory(os.path.dirname(submit.outdir), submit)
    submit.inputfile = fn.SubmissionInputs(timestamp)
    submit.scheduler = fn.SubmissionSchedule(timestamp)
    # print("New time stamp: {}, output directory: {}, input file: {}, scheduler: {}".format(timestamp, submit.outdir, submit.inputfile, submit.scheduler))
    return None


def Schedule(submit):
    # List all the parameters that must be run in every node, explicity.
    # For every node, list out all the parameter values in a two-column format.
    # print("Time stamp: {}, output directory: {}, input file: {}, scheduler: {}".format(submit.timestamp, submit.outdir, submit.inputfile, submit.scheduler))
    with open(submit.scheduler, "w") as sch:
        for i in range(submit.nodes):
            sch.write("!!node %d!!\n" % (i))
            for j in range(submit.cores[0]):
                sch.write(
                    "%s %d\n"
                    % (
                        " ".join(
                            list(
                                map(
                                    lambda num: ("%g" % num),
                                    submit.noiserates[
                                        (i * submit.cores[0] + j) // submit.samps, :
                                    ],
                                )
                            )
                        ),
                        (i * submit.cores[0] + j) % submit.samps,
                    )
                )
                if i * submit.cores[0] + j == (
                    submit.noiserates.shape[0] * submit.samps - 1
                ):
                    break
    return None


def Update(submit, pname, newvalue):
    # Update the parameters to be submitted
    if pname == "timestamp":
        ChangeTimeStamp(submit, newvalue)

    elif pname == "ecc":
        submit.isSubmission = 1
        # Read all the Parameters of the error correcting code
        names = newvalue.split(",")
        submit.ecfiles = []
        submit.levels = len(names)
        submit.decoders = np.zeros(submit.levels, dtype=np.int)
        for i in range(submit.levels):
            submit.eccs.append(qec.QuantumErrorCorrectingCode(names[i]))
            qec.Load(submit.eccs[i])
            submit.ecfiles.append(submit.eccs[i].defnfile)
            submit.decoders[i] = 0

    elif pname == "decoder":
        decoder_info = newvalue.split(",")
        submit.decoders = np.zeros(submit.levels, dtype=np.int)
        for l in range(len(decoder_info)):
            submit.decoders[l] = int(decoder_info[l])

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
        if os.path.isfile("./../input/%s.txt" % (newvalue)) == 1:
            submit.noiserates = np.loadtxt(
                "./../input/%s.txt" % (newvalue), comments="#", dtype=np.longdouble
            )
        else:
            newRanges = list(
                map(
                    lambda arr: list(map(np.longdouble, arr.split(","))),
                    newvalue.split(";"),
                )
            )
            submit.noiserange = []
            for i in range(len(newRanges)):
                if len(newRanges[i]) == 3:
                    submit.noiserange.append(
                        np.linspace(
                            np.longdouble(newRanges[i][0]),
                            np.longdouble(newRanges[i][1]),
                            np.int(newRanges[i][2]),
                        )
                    )
                else:
                    submit.noiserange.append(newRanges[i])
            submit.noiserates = np.array(
                list(map(list, it.product(*submit.noiserange))), dtype=np.float
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
            submit.filter["lower"] = np.float(filterDetails[1])
            submit.filter["upper"] = np.float(filterDetails[2])

    elif pname == "stats":
        # The value can be
        # an integer
        # an explicit range of numbers -- [<list of values separated by commas>]
        # compact range of numbers -- lower,upper,number_of_steps
        if IsNumber(newvalue) == 1:
            submit.stats = np.array([int(newvalue)], dtype=np.int)
        else:
            if "[" in newvalue:
                submit.stats = np.array(
                    list(map(int, newvalue[1:-1].split(","))), dtype=np.int
                )
            else:
                submit.stats = np.geomspace(
                    *np.array(list(map(int, newvalue.split(","))), dtype=np.int),
                    dtype=np.int
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
                dtype=np.int,
            )
            for inter in range(submit.levels + 1)
        ][::-1]
        if fname == "soft":
            # Choose to put every channel in its own bin -- this is just soft decoding.
            for l in range(submit.levels):
                submit.decoderbins.append(np.arange(chans[l], dtype=np.int))
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
                submit.decoderbins.append(np.zeros(chans[l], dtype=np.int))
        else:
            pass
    return None


def PrintSub(submit):
    # Print the available details about the submission.
    colwidth = 30
    print("\033[2mTime stamp: %s\033[0m" % (submit.timestamp))
    print("\033[2m"),
    print("Physical channel")
    print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Parameters", "Values"))
    print(
        ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
            "Channel", "%s" % (submit.channel)
        )
    )
    if submit.scales[0] == 1:
        print(
            ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
                "Noise range",
                "%s." % (np.array_str(submit.noiserange[0], max_line_width=150)),
            )
        )
    else:
        print(
            ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
                "Noise range",
                "%s."
                % (
                    np.array_str(
                        np.power(submit.scales[0], submit.noiserange[0]),
                        max_line_width=150,
                    )
                ),
            )
        )
    for i in range(1, len(submit.noiserange)):
        if submit.scales[i] == 1:
            print(
                ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
                    "", "%s." % (np.array_str(submit.noiserange[i], max_line_width=150))
                )
            )
        else:
            print(
                ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
                    "",
                    "%s."
                    % (
                        np.array_str(
                            np.power(submit.scales[i], submit.noiserange[i]),
                            max_line_width=150,
                        )
                    ),
                )
            )
    print(
        ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
            "Scales of noise rates", "%s" % (np.array_str(submit.scales))
        )
    )
    print(
        ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
            "Number of Samples", "%d" % (submit.samps)
        )
    )

    print("Metrics")
    print(
        ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
            "Logical metrics", "%s" % (", ".join(submit.metrics))
        )
    )

    print("Error correction")
    print(
        ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
            "QECC",
            "%s" % (" X ".join([submit.eccs[i].name for i in range(len(submit.eccs))])),
        )
    )
    print(
        ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
            "[[N, K, D]]",
            "[[%d, %d, %d]]"
            % (
                reduce(
                    (lambda x, y: x * y),
                    [submit.eccs[i].N for i in range(len(submit.eccs))],
                ),
                reduce(
                    (lambda x, y: x * y),
                    [submit.eccs[i].K for i in range(len(submit.eccs))],
                ),
                reduce(
                    (lambda x, y: x * y),
                    [submit.eccs[i].D for i in range(len(submit.eccs))],
                ),
            ),
        )
    )
    print(
        ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
            "Levels of concatenation", "%d" % (submit.levels)
        )
    )
    print(
        ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
            "ECC frame",
            "%s"
            % (
                list(submit.eccframes.keys())[
                    list(submit.eccframes.values()).index(submit.frame)
                ]
            ),
        )
    )
    print(
        ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
            "Decoder", "%d" % (submit.decoder)
        )
    )
    print(
        ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
            "Syndrome samples at level %d" % (submit.levels),
            "%s" % (np.array_str(submit.stats)),
        )
    )
    print(
        ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
            "Type of syndrome sampling",
            "%s"
            % (
                list(submit.samplingOptions.keys())[
                    list(submit.samplingOptions.values()).index(submit.importance)
                ]
            ),
        )
    )

    if not (submit.host == "local"):
        print("Cluster")
        print(
            ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
                "Host", "%s" % (submit.host)
            )
        )
        print(
            ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
                "Account", "%s" % (submit.account)
            )
        )
        print(
            ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
                "Job name", "%s" % (submit.job)
            )
        )
        print(
            ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
                "Number of nodes", "%d" % (submit.nodes)
            )
        )
        print(
            ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
                "Walltime per node", "%d" % (submit.wall)
            )
        )
        print(
            ("{:<%d} {:<%d}" % (colwidth, colwidth)).format(
                "Submission queue", "%s" % (submit.queue)
            )
        )

        print("Usage")
        mam.Usage(submit)
    print("\033[0m"),
    return None


def Save(submit):
    # Write a file named const.txt with all the parameter values selected for the simulation
    # File containing the constant parameters
    # Input file
    # Change the timestamp before saving to avoid overwriting the input file.
    with open(submit.inputfile, "w") as infid:
        # Time stamp
        infid.write("# Time stamp\ntimestamp %s\n" % submit.timestamp)
        # Code type
        infid.write(
            "# Type of quantum error correcting code\necc %s\n"
            % ",".join([submit.eccs[i].name for i in range(len(submit.eccs))])
        )
        # Type of noise channel
        infid.write("# Type of quantum channel\nchannel %s\n" % submit.channel)
        # Channel representation
        infid.write(
            '# Representation of the quantum channel. (Available options: "krauss", "process", "choi", "chi", "stine")\nrepr %s\n'
            % submit.repr
        )
        # Noise range parameters
        infid.write(
            "# Noise rate exponents. The actual noise rate is (2/3)^exponent.\nnoiserange %g,%g,%g"
            % (
                submit.noiserange[0][0],
                submit.noiserange[0][-1],
                submit.noiserange[0].shape[0],
            )
        )
        for i in range(1, len(submit.noiserange)):
            infid.write(
                ";%g,%g,%g"
                % (
                    submit.noiserange[i][0],
                    submit.noiserange[i][-1],
                    submit.noiserange[i].shape[0],
                )
            )
        infid.write("\n")
        # Scale of noise range
        infid.write(
            "# Scales of noise range.\nscale %s\n"
            % (",".join(list(map(str, submit.scales))))
        )
        # Number of samples
        infid.write("# Number of samples\nsamples %d\n" % submit.samps)
        # File name containing the parameters to be run on the particular node
        infid.write("# Parameters schedule\nscheduler %s\n" % (submit.scheduler))
        # Decoder
        infid.write(
            "# Decoding algorithm to be used -- 0 for the maximum likelihood decoder and 1 for minimum weight decoder.\ndecoder %s\n"
            % ",".join(list(map(str, submit.decoders)))
        )
        infid.write(
            "# Hybrid decoding to be used -- 0 for soft decoding and 1 for hybrid decoding.\nhybrid %d\n"
            % (submit.hybrid)
        )
        if submit.hybrid > 0:
            infid.write(
                '# Channels to be averaged at intermediate levels by a hybrid decoder Either a file name containing bins for channels or a keyword from \{"soft", "random <number of bins>", "hard"\}.\ndecbins %s\n'
                % (submit.decoder_type)
            )
        # ECC frame to be used
        infid.write(
            '# Logical frame for error correction (Available options: "[P] Pauli", "[C] Clifford", "[PC] Pauli + Logical Clifford").\nframe %s\n'
            % (
                list(submit.eccframes.keys())[
                    list(submit.eccframes.values()).index(submit.frame)
                ]
            )
        )
        # Number of decoding trials per level
        infid.write(
            "# Number of syndromes to be sampled at top level\nstats [%s]\n"
            % (",".join(list(map(lambda num: ("%d" % num), submit.stats))))
        )
        # Importance distribution
        infid.write(
            '# Importance sampling methods (Available options: ["N"] None, ["A"] Power law sampling, ["B"] Noisy channel)\nimportance %s\n'
            % (
                list(submit.samplingOptions.keys())[
                    list(submit.samplingOptions.values()).index(submit.importance)
                ]
            )
        )
        # Metrics to be computed on the logical channel
        infid.write(
            "# Metrics to be computed on the effective channels at every level.\nmetrics %s\n"
            % ",".join(submit.metrics)
        )
        # Load distribution on cores.
        infid.write(
            "# Load distribution on cores.\ncores %s\n"
            % (",".join(list(map(str, submit.cores))))
        )
        # Number of nodes
        infid.write("# Number of nodes\nnodes %d\n" % submit.nodes)
        # Host
        infid.write("# Name of the host computer.\nhost %s\n" % (submit.host))
        # Account
        infid.write("# Name of the account.\naccount %s\n" % (submit.account))
        # Job name
        infid.write("# Batch name.\njob %s\n" % (submit.job))
        # Wall time
        infid.write("# Wall time in hours.\nwall %d\n" % (submit.wall))
        # Queue
        infid.write(
            "# Submission queue (Available options: see goo.gl/pTdqbV).\nqueue %s\n"
            % (submit.queue)
        )
        # Queue
        infid.write("# Email notifications.\nemail %s\n" % (submit.email))
        # Output directory
        infid.write(
            "# Output result's directory.\noutdir %s\n"
            % (os.path.dirname(submit.outdir))
        )
        # Ranodmized compiling of quantum gates
        infid.write("# Randomized compiling of quantum gates.\nrc %d\n" % (submit.rc))
        # Plot settings
        if submit.plotsettings:
            infid.write(
                "# Plot settings\nplot %s\n"
                % (
                    ";".join(
                        [
                            "%s,%s" % (key, submit.plotsettings[key])
                            for key in submit.plotsettings
                        ]
                    )
                )
            )
        # Miscellaneous information
        infid.write("# Miscellaneous information: %s\n" % (submit.misc))
    return None


def PrepOutputDir(submit):
    # Prepare the output directory -- create it, put the input files.
    # Copy the necessary input files, error correcting code.
    for subdir in ["input", "code", "physical", "channels", "metrics", "results"]:
        os.system("mkdir -p %s/%s" % (submit.outdir, subdir))
    # Copy the relevant code data
    for l in range(submit.levels):
        os.system("cp ./../code/%s %s/code/" % (submit.eccs[l].defnfile, submit.outdir))
    # Save a copy of the input file and the schedule file in the output director.
    copyfile(
        "./../input/%s" % os.path.basename(submit.inputfile),
        "%s/input/%s" % (submit.outdir, os.path.basename(submit.inputfile)),
    )
    copyfile(
        "./../input/%s" % os.path.basename(submit.scheduler),
        "%s/input/%s" % (submit.outdir, os.path.basename(submit.scheduler)),
    )
    return None


def SavePhysicalChannels(submit):
    # Save the physical channels generated to a file.
    save_raw = 1
    if submit.rawchans is None:
        save_raw = 0
    for i in range(submit.phychans.shape[0]):
        (folder, fname) = os.path.split(
            fn.PhysicalChannel(submit, submit.noiserates[i])
        )
        np.save("%s/%s" % (folder, fname), submit.phychans[i, :, :])
        if save_raw == 1:
            np.save("%s/raw_%s" % (folder, fname), submit.rawchans[i, :, :])
    return None


def Select(submit, chan_indices):
    # Identify the details of the physical noise maps given their channel indices.
    os.system("mkdir -p %s/physical/selected" % (submit.outdir))
    print("C   noise   samp")
    phychans = np.zeros(
        (len(chan_indices), 4 ** submit.eccs[0].K * 4 ** submit.eccs[0].K),
        dtype=np.double,
    )
    selected = submit.available[chan_indices, :]
    for c in range(len(chan_indices)):
        print(
            "%d   %s     %d"
            % (
                chan_indices[c],
                " ".join(
                    list(
                        map(
                            lambda num: "%g" % num,
                            submit.available[chan_indices[c], :-1],
                        )
                    )
                ),
                submit.available[chan_indices[c], -1],
            )
        )
        chanfile = fn.PhysicalChannel(submit, submit.available[chan_indices[c], :-1])
        phychans[c, :] = np.load(chanfile)[
            int(submit.available[chan_indices[c], -1]), :
        ]
        chan = np.reshape(phychans[c, :], [4, 4])
        # Save the channel into a new folder
        (folder, fname) = os.path.split(chanfile)
        (name, extn) = os.path.splitext(fname)
        os.system("mkdir -p %s/selected" % (folder))
        np.save(
            "%s/selected/%s_s%d%s"
            % (folder, name, submit.available[chan_indices[c], -1], extn),
            chan,
        )
    print("Total: %d channels." % (selected.shape[0]))
    # Save all the channels into one file
    np.save(
        "%s/physical/selected/%s_%s.npy"
        % (
            submit.outdir,
            submit.channel,
            "_".join(
                list(map(str, [1 for __ in range(submit.available.shape[1] - 2)]))
            ),
        ),
        phychans,
    )
    # Compute the noise range that will contain the selected channels.
    noise_range = [
        [np.min(selected[:, i]), np.max(selected[:, i]), Increment(selected[:, i])]
        for i in range(selected.shape[1] - 1)
    ]
    # print("noise_range: {}".format(noise_range))
    print(
        "encapsulating noise noise range:\n{}\nsamples = {}".format(
            ";".join(
                list(
                    map(
                        lambda nr: "%g,%g,%d"
                        % (nr[0], nr[1], 1 + (nr[1] - nr[0]) / nr[2]),
                        noise_range,
                    )
                )
            ),
            int(np.max(selected[:, -1])),
        )
    )
    return None


def Increment(arr):
    r"""
    Compute the smallest non-zero increment in an array.
    """
    min_increment = np.max(arr)
    atol = 10e-8
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            current_increment = abs(arr[i] - arr[j])
            if current_increment > atol:
                if current_increment < min_increment:
                    min_increment = current_increment
    return min_increment


def LoadSub(submit, subid, isgen):
    # Load the parameters of a submission from an input file
    # If the input file is provided as the submission id, load from that input file.
    # Else if the time stamp is provided, search for the corresponding input file and load from that.
    ChangeTimeStamp(submit, subid)
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
                    )
        return 1
    else:
        print("\033[2mInput file not found.\033[0m")
    return 0
