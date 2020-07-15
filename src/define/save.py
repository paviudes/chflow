import os
import numpy as np
from shutil import copyfile
from define import fnames as fn


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
        if submit.noiserange is not None:
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
        else:
            infid.write(
                "# Noise rate exponents. The actual noise rate is (2/3)^exponent.\nnoiserange %s"
                % submit.ratesfile
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
            "# Fraction of Pauli probabilities accessible to the ML decoder.\ndcfraction %g\n"
            % (submit.decoder_fraction)
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


def PrepOutputDir(submit):
    # Prepare the output directory -- create it, put the input files.
    # Copy the necessary input files, error correcting code.
    for subdir in ["input", "code", "physical", "channels", "metrics", "results"]:
        os.system("mkdir -p %s/%s" % (submit.outdir, subdir))
    # Copy the relevant code data
    for l in range(submit.levels):
        os.system("cp ./../code/%s %s/code/" % (submit.eccs[l].defnfile, submit.outdir))
    # Save a copy of the input file and the schedule file in the output director.
    if os.path.isfile("./../input/%s" % submit.inputfile):
        copyfile(
            "./../input/%s" % os.path.basename(submit.inputfile),
            "%s/input/%s" % (submit.outdir, os.path.basename(submit.inputfile)),
        )
    if os.path.isfile("./../input/%s" % submit.scheduler):
        copyfile(
            "./../input/%s" % os.path.basename(submit.scheduler),
            "%s/input/%s" % (submit.outdir, os.path.basename(submit.scheduler)),
        )
    return None
