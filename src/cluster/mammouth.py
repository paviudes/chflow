import os

def GetCoresInNode():
	# get the number of cores in a node by getting the value of the $SLURM_NTASKS variable.
	# This is the value of ntasks in the slurm file, which is "SLURM_NTASKS".
	return int(os.environ["PBS_NUM_PPN"])

def Usage(submit):
	# Print the amount of resources that will be used up by a simulation.
	limits = {"ms":100000, "mp2":700000}
	quota = submit.nodes * submit.wall * 100/float(limits[submit.host])
	print("\033[2m%d nodes will run for a maximum time of %d hours.\n%g%% of total usage quota will be used.\033[0m" % (submit.nodes, submit.wall, quota))
	return None

def CreateLaunchScript(submit):
	# Produce the bqsubmit.dat file which contains all the parameters for mammouth execution.
	with open(("./../%s.pbs" % (submit.host)), 'w') as bq:
		bq.write("#!/bin/bash\n\n")
		
		# Job names
		bq.write("#PBS -N %s\n\n" % (submit.job))

		# Account name to which the usage must be billed
		bq.write("#PBS -A fia-010-aa\n\n")

		# Name of the submission queue
		bq.write("#PBS -q %s\n\n" % (submit.queue))

		# Wall time
		bq.write("#PBS -l walltime=%d:00:00\n\n" % (submit.wall))

		# Redirecting STDOUT and STDERR files
		bq.write("#PBS -o %s/results/ouptut.txt\n" % (submit.outdir))
		bq.write("#PBS -e %s/results/errors.txt\n" % (submit.outdir))
		bq.write("#PBS -W umask=022\n\n")

		# Email notifications
		bq.write("#PBS -m bea\n")
		bq.write("#PBS -M %s\n\n" % (submit.email))

		# Job arrays
		bq.write("#PBS -t 1-%d\n\n" % (submit.nodes))

		# Command to be run
		bq.write("cd \"${PBS_O_WORKDIR}\"/\n")
		bq.write("eval ./chflow.sh %s $PBS_ARRAYID\n" % (submit.timestamp))
	print("\033[2mSubmit the job using\nqsub %s.pbs\nSee https://wiki.calculquebec.ca/w/Running_jobs#tab=tab7 for details.\033[0m" % (submit.host))
	return None