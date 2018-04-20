def Usage(submit):
	# Print the amount of resources that will be used up by a simulation.
	limits = 48 * 365 * 24
	quota = submit.nodes * submit.wall * 100/float(limits)
	print("\033[2m%d nodes will run for a maximum time of %d hours.\n%g%% of total usage quota will be used.\033[0m" % (submit.nodes, submit.wall, quota))
	return None

def CreateLaunchScript(submit):
	# Write the script to launch a job-array describing all the simulations to be run.
	# See https://slurm.schedmd.com/sbatch.html
	with open("./../frontenac.sh", "w") as fp:
		fp.write("#!/bin/bash\n")

		# Account name to which the usage must be billed
		fp.write("#SBATCH --account=%s\n" % (submit.account))
		fp.write("#SBATCH --partition=standard\n\n")

		# Wall time in (DD-HH:MM)
		fp.write("#SBATCH --begin=now\n")
		fp.write("#SBATCH --time=%d:00:00\n\n" % (submit.wall))

		# Job array specification
		fp.write("#SBATCH --array=0-%d:1\n" % (submit.nodes - 1))
		fp.write("#SBATCH --cpus-per-task=24\n")
		fp.write("#SBATCH --nodes=%d\n" % (submit.nodes))
		# fp.write("#SBATCH --ntasks=1\n" % (submit.nodes))
		fp.write("#SBATCH --output=%s_%%A_%%a.out\n\n" % (submit.job))

		# Redirecting STDOUT and STDERR files
		fp.write("#SBATCH -o %s/results/ouptut_%%j.o\n" % (submit.outdir))
		fp.write("#SBATCH -e %s/results/errors_%%j.o\n\n" % (submit.outdir))

		# Email notifications
		fp.write("#SBATCH --mail-type=ALL\n")
		fp.write("#SBATCH --mail-user=%s\n\n" % (submit.email))

		# Command to be executed for each job step
		fp.write("module load anaconda/2.7.13\n")
		fp.write("module load gcc/6.4.0\n")
		fp.write("cd $SLURM_SUBMIT_DIR\n")
		fp.write("cd src/simulate/\n")
		fp.write("python compile.py build_ext --inplace > compiler_output.txt 2>&1\n")
		fp.write("cd $SLURM_SUBMIT_DIR\n")
		fp.write("./chflow.sh %s ${SLURM_ARRAY_TASK_ID}\n" % (submit.timestamp))
	print("\033[2mRun the following\nsbatch frontenac.sh\nto launch the job.\nSee https://cac.queensu.ca/wiki/index.php/SLURM#Running_jobs for details.\033[0m")
	return None