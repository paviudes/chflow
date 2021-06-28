import os
import sys
import time
from shutil import copyfile
import numpy as np
import itertools as it
from define import fnames as fn
from define import qcode as qec
from define import qchans as qc
from define import globalvars as gv


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
        self.chgen_cluster = 0
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
        self.ratesfile = ""
        self.noiserange = []
        self.noiserates = np.array([])
        self.scales = []
        self.samps = 1
        self.channels = 0
        self.available = np.array([])
        self.overwrite = 1
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
        self.decoder_fraction = 0
        self.decoder_type = "default_soft"
        self.hybrid = 0
        self.decoderbins = []
        self.ndecoderbins = []
        self.nr_weights = None
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


def LoadTimeStamp(submit, timestamp):
    # change the timestamp of a submission and all the related values to the timestamp.
    submit.timestamp = timestamp
    # Re define the variables that depend on the time stamp
    submit.outdir = fn.OutputDirectory(os.path.dirname(submit.outdir), submit)
    submit.inputfile = fn.SubmissionInputs(timestamp)
    submit.scheduler = fn.SubmissionSchedule(timestamp)
    # print("New time stamp: {}, output directory: {}, input file: {}, scheduler: {}".format(timestamp, submit.outdir, submit.inputfile, submit.scheduler))
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


def Select(submit, chan_indices):
    # Identify the details of the physical noise maps given their channel indices.
    os.system("mkdir -p %s/physical/selected" % (submit.outdir))
    print("C   noise   samp")
    
    if (submit.iscorr == 0):
        phychans = np.zeros(
            (len(chan_indices), 4 ** submit.eccs[0].K * 4 ** submit.eccs[0].K),
            dtype=np.double,
        )
    elif (submit.iscorr == 1):
        phychans = np.zeros(
            (len(chan_indices), 2**(submit.eccs[0].N + submit.eccs[0].K)),
            dtype=np.double,
        )
    elif (submit.iscorr == 2):
        phychans = np.zeros(
            (len(chan_indices), submit.eccs[0].N * 4 ** submit.eccs[0].K * 4 ** submit.eccs[0].K),
            dtype=np.double,
        )
    elif (submit.iscorr == 3):
        phychans = np.zeros(
            (len(chan_indices), 4**(submit.eccs[0].N + submit.eccs[0].K)),
            dtype=np.double,
        )
    else:
        pass


    if (submit.available.shape[0] == 0):
        submit.available = np.zeros((submit.noiserates.shape[0] * submit.samps, submit.noiserates.shape[1] + 1), dtype = np.double)
        for i in range(submit.noiserates.shape[0]):
            for j in range(submit.samps):
                submit.available[i * submit.samps + j, :] = np.concatenate((submit.noiserates[i, :], [j]))
    # print("available:\n{}".format(submit.available))
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
        phychans[c, :] = np.load(chanfile)[int(submit.available[chan_indices[c], -1]), :]
        if (submit.iscorr == 0):
            chan = np.reshape(phychans[c, :], [4, 4])
        elif (submit.iscorr == 1):
            chan = phychans[c, :]
        elif (submit.iscorr == 2):
            chan = np.reshape(phychans[c, :], [submit.eccs[0].N, 4, 4])
        elif (submit.iscorr == 3):
            chan = np.reshape(phychans[c, :], [2**(submit.eccs[0].N + submit.eccs[0].K), 2**(submit.eccs[0].N + submit.eccs[0].K)])
        else:
            pass
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


def CopyConsts(dset, merged):
    """
    Copy the constant parameters when datasets are merged.
    """
    merged.eccs = dset.eccs
    merged.channel = dset.channel
    merged.repr = dset.repr
    merged.scales = dset.scales
    merged.decoders = dset.decoders
    merged.hybrid = dset.hybrid
    merged.decoder_type = dset.decoder_type
    merged.eccframes = dset.eccframes
    merged.samplingOptions = dset.samplingOptions
    merged.metrics = dset.metrics
    merged.cores = dset.cores
    merged.nodes = dset.nodes
    merged.host = dset.host
    merged.account = dset.account
    merged.job = dset.job
    merged.wall = dset.wall
    merged.queue = dset.queue
    merged.email = dset.email
    merged.rc = dset.rc
    merged.plotsettings = dset.plotsettings
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
