import os
from define import globalvars as gv


def GetCoresInNode():
    # get the number of cores in a node by getting the value of the $SLURM_NTASKS variable.
    # This is the value of ntasks in the slurm file, which is "SLURM_NTASKS".
    return int(os.environ["SLURM_NTASKS"])


def Usage(submit):
    # Print the amount of resources that will be used up by a simulation.
    limits = 50 * 365 * 24  # 50 core-years
    usage = submit.nodes * submit.wall * 100 / float(limits)
    print(
        "\033[2m%d nodes will run for a maximum time of %d hours.\n%g%% of total usage quota will be used.\033[0m"
        % (submit.nodes, submit.wall, usage)
    )
    return None


def CreatePreBatch(submit):
    """
    The prebatch script prepares the input for a simulation.
    The channels will be generated in this script.
    """
    if not (
        os.path.isfile("./../input/%s/pre_%s.txt" % (submit.host, submit.timestamp))
    ):
        with open(
            "./../input/%s/pre_%s.txt" % (submit.host, submit.timestamp), "w"
        ) as pb:
            pb.write("sbload %s\n" % (submit.timestamp))
            pb.write("chgen\n")
            pb.write("submit\n")
            pb.write("quit\n")

    if not (
        os.path.isfile("./../input/%s/pre_%s.sh" % (submit.host, submit.timestamp))
    ):
        with open(
            "./../input/%s/pre_%s.sh" % (submit.host, submit.timestamp), "w"
        ) as pb:
            pb.write("#!/bin/bash\n")
            pb.write("#SBATCH --account=%s\n" % (submit.account))
            pb.write("#SBATCH --begin=now\n")
            pb.write("#SBATCH --time=5:00:00\n\n")
            pb.write("#SBATCH --ntasks-per-node=%d\n" % (gv.cluster_info[submit.host]))
            pb.write("#SBATCH --nodes=1\n")
            # Redirecting STDOUT and STDERR files
            pb.write("#SBATCH -o %s/results/pre_ouptut_%%j.o\n" % (submit.outdir))
            pb.write("#SBATCH -e %s/results/pre_errors_%%j.o\n\n" % (submit.outdir))
            # Email notifications
            pb.write("#SBATCH --mail-type=ALL\n")
            pb.write("#SBATCH --mail-user=%s\n\n" % (submit.email))
            # Command to be executed for each job step
            pb.write("module load intel/2016.4 python/3.7.0 scipy-stack/2019a\n")
            pb.write("cd /project/def-jemerson/pavi/chflow\n")
            pb.write("./chflow.sh -- %s/pre_%s.txt\n" % (submit.host, submit.timestamp))
        print("\033[2m-----\033[0m")
        print(
            "\033[2mRun the following command to generate channels.\n\tsbatch input/%s/pre_%s.sh\033[0m"
            % (submit.host, submit.timestamp)
        )
        print("\033[2m-----\033[0m")
    return None


def CreatePostBatch(submit):
    """
    The prebatch script prepares the input for a simulation.
    The channels will be generated in this script.
    """
    if not (
        os.path.isfile("./../input/%s/post_%s.txt" % (submit.host, submit.timestamp))
    ):
        with open(
            "./../input/%s/post_%s.txt" % (submit.host, submit.timestamp), "w"
        ) as pb:
            pb.write("sbload %s\n" % (submit.timestamp))
            pb.write("pmetrics infid\n")
            pb.write("lpmetrics uncorr\n")
            pb.write("quit\n")

    if not (
        os.path.isfile("./../input/%s/post_%s.sh" % (submit.host, submit.timestamp))
    ):
        with open(
            "./../input/%s/post_%s.sh" % (submit.host, submit.timestamp), "w"
        ) as pb:
            pb.write("#!/bin/bash\n")
            pb.write("#SBATCH --account=%s\n" % (submit.account))
            pb.write("#SBATCH --begin=now\n")
            pb.write("#SBATCH --time=5:00:00\n\n")
            pb.write("#SBATCH --ntasks-per-node=%d\n" % (gv.cluster_info[submit.host]))
            pb.write("#SBATCH --nodes=1\n")
            # Redirecting STDOUT and STDERR files
            pb.write("#SBATCH -o %s/results/post_%%j.o\n" % (submit.outdir))
            pb.write("#SBATCH -e %s/results/post_%%j.o\n\n" % (submit.outdir))
            # Email notifications
            pb.write("#SBATCH --mail-type=ALL\n")
            pb.write("#SBATCH --mail-user=%s\n\n" % (submit.email))
            # Command to be executed for each job step
            pb.write("module load intel/2016.4 python/3.7.0 scipy-stack/2019a\n")
            pb.write("cd /project/def-jemerson/pavi/chflow\n")
            pb.write("./chflow -- post_%s.txt\n" % (submit.timestamp))
            pb.write("cd %s\n" % (os.path.dirname(submit.outdir)))
            pb.write(
                "tar -zcvf %s.tar.gz %s\n"
                % (os.path.basename(submit.outdir), os.path.basename(submit.outdir))
            )
        print("\033[2m-----\033[0m")
        print(
            "\033[2mRun the following command to compute metrics and zip results.\n\tsbatch input/%s/post_%s.sh\033[0m"
            % (submit.host, submit.timestamp)
        )
        print("\033[2m-----\033[0m")
    return None


def CreateLaunchScript(submit):
    # Write the script to launch a job-array describing all the simulations to be run.
    # See https://slurm.schedmd.com/sbatch.html
    with open("./../input/%s/%s.sh" % (submit.host, submit.timestamp), "w") as fp:
        fp.write("#!/bin/bash\n")
        # Account name to which the usage must be billed
        fp.write("#SBATCH --account=%s\n" % (submit.account))
        # Wall time in (DD-HH:MM)
        fp.write("#SBATCH --begin=now\n")
        if submit.wall < 24:
            fp.write("#SBATCH --time=%d:00:00\n\n" % (submit.wall))
        else:
            fp.write(
                "#SBATCH --time=%d-%d:00:00\n\n"
                % (submit.wall // 24, (submit.wall % 24))
            )
        # Job array specification
        fp.write("#SBATCH --array=0-%d:1\n" % (submit.nodes - 1))
        fp.write("#SBATCH --cpus-per-task=%d\n" % (submit.cores[1]))
        fp.write("#SBATCH --ntasks-per-node=%d\n" % (gv.cluster_info[submit.host]))
        fp.write("#SBATCH --nodes=1\n")
        # fp.write("#SBATCH --mem=31744\n")
        fp.write("#SBATCH --output=%s_%%A_%%a.out\n\n" % (submit.job))
        # Redirecting STDOUT and STDERR files
        fp.write("#SBATCH -o %s/results/ouptut_%%j.o\n" % (submit.outdir))
        fp.write("#SBATCH -e %s/results/errors_%%j.o\n\n" % (submit.outdir))
        # Email notifications
        fp.write("#SBATCH --mail-type=ALL\n")
        fp.write("#SBATCH --mail-user=%s\n\n" % (submit.email))
        # Command to be executed for each job step
        fp.write("module load intel/2016.4 python/3.7.0 scipy-stack/2019a\n")
        fp.write("cd /project/def-jemerson/pavi/chflow\n")
        fp.write("./chflow.sh %s ${SLURM_ARRAY_TASK_ID}\n" % (submit.timestamp))
    print("\033[2m-----\033[0m")
    print(
        "\033[2mRun the following command to launch the job.\n\tsbatch input/%s/%s.sh\033[0m"
        % (submit.host, submit.timestamp)
    )
    print("\033[2m-----\033[0m")
    return None
