import os
import sys
# Force the module scripts to run locally -- https://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path
import inspect as ins
current = os.path.realpath(os.path.abspath(os.path.dirname(ins.getfile(ins.currentframe()))))
if (not (current in sys.path)):
	sys.path.insert(0, current)

def Scheduler(submit):
	# List all the parameters that must be run in every node, explicity.
	# For every node, list out all the parameter values in a two-column format.
	with open(submit.scheduler, 'w') as sch:
		for i in range(submit.nodes):
			sch.write("!!node %d!!\n" % (i))
			for j in range(submit.cores[0]):
				sch.write("%s %d\n" % (" ".join(map(lambda num: ("%g" % num), submit.params[i * submit.cores[0] + j, :-1])), submit.params[i * submit.cores[0] + j, -1]))
				if (i * submit.cores[0] + j == (submit.params.shape[0] - 1)):
					break
	return None

def WriteToBqSubmit(submit):
	# Produce the bqsubmit.dat file which contains all the parameters for mammouth execution.
	with open("./../bqsubmit.dat", 'w') as bq:
		bq.write("batchName = %s\n\n" % (submit.job))
		# copy the lattice text files
		bq.write("copyFiles = chflow.sh,docs,src,physical,input,code\n\n")
		# Input file with the values of variables
		bq.write("templateFiles = input/%s\n\n" % (os.path.basename(submit.inputfile)))
		# Command to run on the compute node
		bq.write("command = mv %s input/;export OMP_NUM_THREADS=%d;./chflow.sh %s\n\n" % (os.path.basename(submit.inputfile), submit.cores[0] * submit.cores[1], submit.timestamp))
		# Command to be run after all jobs are exited or deleted
		bq.write("postBatch = ")
		bq.write("cp -r code %s;" % (submit.outdir))
		bq.write("cp -r src %s;" % (submit.outdir))
		bq.write("cp -r input %s;" % (submit.outdir))
		bq.write("cp -r physical %s;" % (submit.outdir))
		bq.write("cp chflow.sh %s;" % (submit.outdir))
		bq.write("mkdir -p %s/channels;" % (submit.outdir))
		bq.write("mv %s_node*.BQ/temp/channels/* %s/channels/;" % (submit.job, submit.outdir))
		bq.write("mkdir -p %s/metrics;" % (submit.outdir))
		bq.write("mv %s_node*.BQ/temp/metrics/* %s/metrics/;" % (submit.job, submit.outdir))
		bq.write("mkdir -p %s/results;" % (submit.outdir))
		bq.write("cat %s_node*.BQ/temp/perf.txt >> %s/results/perf.txt;" % (submit.job, submit.outdir))
		bq.write("cd /home/pavi/chbank;")
		bq.write("tar -zcvf %s.tar.gz %s;" % (os.path.basename(submit.outdir), os.path.basename(submit.outdir)))
		bq.write("cd /home/pavi/chflow;")
		bq.write("rm -rf %s_*.BQ/;" % (submit.job))
		bq.write("rm -rf %s/\n\n" % (submit.outdir))
		# Required resources for each node
		bq.write("submitOptions=-q %s@%s -l walltime=%d:00:00,nodes=1\n\n" % (submit.queue, submit.host, submit.wall))
		# Number of jobs to be run at a given time on a node and number of jobs to be accumulated for execution on a single node.
		if (submit.host == "mp2"):
			bq.write("accJobsPerNode = 1\nrunJobsPerNode = 1\n\n")
		# Parameters to be scanned in the batch. The batch spans different nodes and each node has sumit.ncores cores.
		# We would like to distribute jobs such that all available cores are used up.
		# For each node, we the noise rate sample index that must be simulated within the node.
		# The dtstribution must ensure that maximum number of cores are used up in every node.
		bq.write("param1 = node = [%s]\n\n" % (",".join(map(lambda num: ("%d" % num), range(submit.nodes)))))
		# Number of nodes to be used concurrently
		bq.write("concurrentJobs = %d\n\n" % (min(submit.nodes, 144)))
		# send an email
		bq.write("emailAddress = pavithran.sridhar@gmail.com")
	return None


def CheckDuplicateSubmissions(submit):
	# Check for any running jobs with the same name as the current job and produce an alert.
	isdup = 0
	bqstatus = os.system("timeout 5s bqstatus > bqstatus.txt 2>&1 > /dev/null 2>&1")
	if (os.path.isfile("bqstatus.txt")):
		with open("bqstatus.txt", "r") as bfp:
			for line in bfp:
				if (submit.job in bfp):
					print("\033[2mA job with the same name is presently running.\033[0m")
					isdup = 1
		os.remove("bqstatus.txt")
		# Check for any output folders which have the same job name and produce an alert.
		os.system("timeout 5s ls | grep %s_node*.BQ > bqfolders.txt 2>&1 > /dev/null 2>&1" % (submit.job))
		if (os.path.isfile("bqfolders.txt")):
			if (os.path.getsize("bqfolders.txt") > 0):
				print("\033[2mOutput files for this job seem to exist.\033[0m")
				isdup = 1
			os.remove("bqfolders.txt")
	return isdup


def SubmitOnMammouth(submit):
	## Submit a job on mammouth
	# Schedule the jobs according to nodes.
	Scheduler(submit)
	# produce the bqsubmit.dat file
	WriteToBqSubmit(submit)
	# Check for submissions with the same name which are either active or completed.
	CheckDuplicateSubmissions(submit)
	return None
