import os
import sys
import time
import numpy as np
import itertools as it
# Files from the define module
import fnames as fn
import qcode as qec
from cluster import mammouth as mam

class Submission():
	def __init__(self):
		# Logging options
		self.timestamp = time.strftime("%d/%m/%Y %H:%M:%S").replace("/", "_").replace(":", "_").replace(" ", "_")
		
		# Mammouth options
		self.job = "X"
		self.host = "local"
		self.nodes = 0
		self.wall = 0
		self.params = np.array([1, 0], dtype = float)
		self.queue = "X"
		self.current = ""

		# Run time options
		self.cores = [1, 1]
		self.inputfile = InputFile(self.timestamp)
		self.isSubmission = 0
		self.scheduler = Scheduler(self.timestamp)

		# Channel options
		self.channel = "X"
		self.repr = "process"
		self.noiserange = np.array([])
		self.noiserates = np.array([[]])
		self.scale = 1
		self.samps = 1
		self.chfiles = []
		self.channels = 0
		self.available = np.array([])

		# Metrics options
		self.metrics = ["frb"]
		self.filter = {'metric':'fidelity', 'lower':0, 'upper':1}

		# ECC options
		self.eccs = []
		self.decoder = 0
		self.eccframes = {"P":0, "C":1, "PC":2}
		self.frame = 0
		self.levels = 0
		self.ecfiles = []
		
		# Sampling options
		self.stats = 0
		self.samplingOptions = {"N": 0, "A": 1, "B": 2}
		self.importance = 0
		
		# Advanced options
		self.isAdvanced = 0

		# Output options
		self.outdir = fn.OutputDirectory(os.path.abspath("./../../"), self)
		

def Scheduler(timestamp):
	# name of the scheduler file.
	return ("./../input/schedule_%s.txt" % (timestamp))

def InputFile(timestamp):
	# name of the script containing the commands to be run
	return ("./../input/%s.txt" % timestamp)


def Usage(submit):
	# Print the amount of resources that will be used up by a simulation.
	if (submit.isSubmission == 1):
		if (submit.host == "ms"):
			totalavailable = 100000
		else:
			totalavailable = 700000
		quota = submit.nodes * submit.wall * 100/float(100000)
		print("\033[2m%d nodes will run for a maximum time of %d hours.\n%g%% of total usage quota will be used up if the simulation runs for the entrie walltime.\033[0m" % (submit.nodes, submit.wall, quota))
	return None

def ChangeTimeStamp(submit, timestamp):
	# change the timestamp of a submission and all the related values to the timestamp.
	submit.timestamp = timestamp
	# Re define the variables that depend on the time stamp
	submit.inputfile = InputFile(submit.timestamp)
	submit.scheduler = Scheduler(submit.timestamp)
	submit.outdir = fn.OutputDirectory(os.path.dirname(submit.outdir), submit)
	return None


def Update(submit, pname, newvalue):
	# Update the parameters to be submitted
	if (pname == "timestamp"):
		ChangeTimeStamp(submit, newvalue)

	elif (pname == "ecc"):
		submit.isSubmission = 1
		# Read all the Parameters of the error correcting code
		names = newvalue.split(",")
		submit.ecfiles = []
		for i in range(len(names)):
			submit.eccs.append(qec.QuantumErrorCorrectingCode(names[i]))
			submit.eccs[i].Load()
			submit.ecfiles.append([submit.eccs[i].defnfile] + [submit.eccs[i].LAOpsFname, submit.eccs[i].LAPhaseFname, submit.eccs[i].LGensFname, submit.eccs[i].TGensFname, submit.eccs[i].SGensFname, submit.eccs[i].stabSyndSignsFname, submit.eccs[i].lookupFname, submit.eccs[i].conjfname, submit.eccs[i].pbasisfname])
		if (not (submit.eccs[i].name in ["Steane", "FiveQubit", "Cat", "FourQubit", "FiveRep"])):
			sys.stderr.write("\033[93mWarning, possibly unknown error correcting code: %s.\n\033[0m" % (submit.eccs[i]))

	elif (pname == "channel"):
		submit.channel = newvalue

	elif (pname == "repr"):
		submit.repr = newvalue

	elif (pname == "noiserange"):
		# For each free parameter, the value must be a float array: low,high,number of steps.
		# The value for different free parameters must be separated by a ";".
		# The noise range is interpretted in the logarithmic scale.
		newRanges = map(lambda arr: map(np.longdouble, arr.split(",")), newvalue.split(";"))
		submit.noiserange = []
		for i in range(len(newRanges)):
			submit.noiserange.append(np.linspace(np.longdouble(newRanges[i][0]), np.longdouble(newRanges[i][1]), np.int(newRanges[i][2])))
		submit.noiserates = np.array(map(list, it.product(*submit.noiserange)), dtype = np.float)
	elif (pname == "samples"):
		# The value must be an integer
		submit.samps = int(newvalue)

	elif (pname == "frame"):
		# The value must be an integer
		submit.frame = submit.eccframes[newvalue]

	elif (pname == "filter"):
		# The input is a filtering crieterion described as (metric, lower bound, upper bound).
		# Only the channels whose metric value is between the lower and the upper bounds are to be selected.
		if (newvalue == ""):
			submit.filter['metric'] = "fidelity"
			submit.filter['lower'] = 0
			submit.filter['upper'] = 1
		else:
			filterDetails = newvalue.split(",")
			submit.filter['metric'] = filterDetails[0]
			submit.filter['lower'] = np.float(filterDetails[1])
			submit.filter['upper'] = np.float(filterDetails[2])

	elif (pname == "levels"):
		# The value must be an integer
		submit.isSubmission = 1
		submit.levels = int(newvalue)

	elif (pname == "stats"):
		# The value must be an integer
		submit.isSubmission = 1
		submit.stats = int(newvalue)

	elif (pname == "metrics"):
		# The metrics to be computed at the logical level
		submit.metrics = newvalue.split(",")

	elif (pname == "wall"):
		submit.isSubmission = 1
		submit.wall = int(newvalue)

	elif (pname == "importance"):
		submit.isSubmission = 1
		submit.importance = submit.samplingOptions[newvalue]

	elif (pname == "decoder"):
		submit.isSubmission = 1
		submit.decoder = int(newvalue)

	elif (pname == "job"):
		submit.isSubmission = 1
		submit.job = newvalue

	elif (pname == "host"):
		submit.isSubmission = 1
		submit.host = newvalue

	elif (pname == "queue"):
		submit.isSubmission = 1
		submit.queue = newvalue

	elif (pname == "cores"):
		submit.cores = map(int, newvalue.split(","))

	elif (pname == "nodes"):
		submit.isSubmission = 1
		submit.nodes = int(newvalue)
	
	elif (pname == "current"):
		submit.current = newvalue

	elif (pname == "scheduler"):
		submit.scheduler = newvalue

	elif (pname == "outdir"):
		submit.outdir = fn.OutputDirectory(newvalue, submit)

	elif (pname == "scale"):
		submit.scale = np.longdouble(newvalue)

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
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Channel", "%s" % (submit.channel)))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Noise range", "%s." % (np.array_str(submit.noiserange[0], max_line_width = 150))))
	for i in range(1, len(submit.noiserange)):
		print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("", "%s." % (np.array_str(submit.noiserange[i], max_line_width = 150))))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Scale of noise rates", "%g" % (submit.scale)))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Number of Samples", "%d" % (submit.samps)))

	print("Metrics")
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Logical metrics", "%s" % (", ".join(submit.metrics))))

	print("Error correction")
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("QECC", "%s" % (" X ".join([submit.eccs[i].name for i in range(len(submit.eccs))]))))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("[[N, K, D]]", "[[%d, %d, %d]]" % (reduce((lambda x,y: x * y), [submit.eccs[i].N for i in range(len(submit.eccs))]), reduce((lambda x,y: x * y), [submit.eccs[i].K for i in range(len(submit.eccs))]), reduce((lambda x,y: x * y), [submit.eccs[i].D for i in range(len(submit.eccs))]))))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Levels of concatenation", "%d" % (submit.levels)))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("ECC frame", "%d" % (submit.frame)))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Decoder", "%d" % (submit.decoder)))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Syndrome samples at level %d" % (submit.levels), "%d" % (submit.stats)))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Type of syndrome sampling", "%d" % (submit.importance)))
	
	print("Mammouth")
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Host", "%s" % (submit.host)))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Job name", "%s" % (submit.job)))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Number of nodes", "%d" % (submit.nodes)))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Walltime per node", "%d" % (submit.wall)))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Submission queue", "%s" % (submit.queue)))
	
	print("Usage")
	Usage(submit)

	print("\033[0m")
	return None


def MergeSubs(parent, *children):
	# Merge the simulation data obtained in two submissions and create a new submission record with the combined simulation data
	parent.timestamp = ("merged_%s" % ("_".join([children[i].timestamp for i in range(len(children))])))
	# ECC, channel, importance, decoder must be the same for all children
	parent.ecc = children[0].ecc
	parent.channel = children[0].channel
	parent.importance = children[0].importance
	parent.decoder = children[0].decoder
	parent.outdir = children[0].outdir
	parent.noiserange = np.union1d([children[i].noiserange for i in range(len(children))])
	accumulation = [[] for i in range(parent.noiserange.shape[0])]
	samps = np.zeros(len(children), dtype = np.int)
	for i in range(parent.noiserange.shape[0]):
		for j in range(len(children)):
			if (parent.noiserange[i] in children[j].noiserange):
				accumulation[i].append(j)
				samps[i] = samps[i] + children[j].samps
	parent.samps = np.max(samps, dtype = np.int)
	parent.levels = np.max([children[i].levels for i in range(len(children))], dtype = np.int)
	parent.stats = np.max([children[i].stats for i in range(len(children))], dtype = np.int)
	parent.metrics = list(set([metname for i in range(len(children)) for metname in children[i].metrics]))
	# Gather all the simulation input and output data and write into a file in the parent directory.
	for i in range(parent.noiserange.shape[0]):
		physical = np.zeros((parent.samps, 4, 4), dtype = np.longdouble)
		logical = np.zeros((parent.samps, parent.levels, 4, 4), dtype = np.longdouble)
		filled = 0
		for j in range(len(accumulation[i])):
			physical[filled:(filled + children[accumulation[i][j]].samps)] = np.load("%s/channels/%s_%g.npy" % (children[accumulation[i][j]].outdir, children[accumulation[i][j]].channel, parent.noiserange[i]))
			for k in range(children[accumulation[i][j]].samps):
				logical[filled + k, :parent.levels, :, :] = np.load(fn.LogicalChannel(children[accumulation[i][j]], parent.noiserange[i], k))[:parent.levels]
				for m in range(len(children[accumulation[i][j]])):
					fname = fn.LogicalErrorRate(children[accumulation[i][j]], children[accumulation[i][j]].metrics[m], parent.noiserange[i], k)
					metrics[filled + k, parent.metrics.index(children[accumulation[i][j]].metrics[m]), :] = np.load(fname)
			filled = filled + children[accumulation[i][j]].samps
		# Write into file
		np.save(("%s/channels/%s_%g.npy" % (parent.outdir, parent.channel, parent.noiserange[i])), physical)
		for k in range(filled):
			np.save(fn.LogicalChannel(parent, parent.noiserange[i], k), logical)
			for m in range(parent.metrics):
				fname = fn.LogicalErrorRate(parent, parent.metrics[m], parent.noiserange[i], k)
				np.save(fname, metrics[k, m, :])
	Save(parent)
	return None

def Validate(submit):
	# Validate the submission details
	# 1. Check if all the necessary parameters are provided
	# 2. Check if the required files exist
	hard = []
	soft = []
	if (submit.isSubmission == 1):
		# Check if the source files exist
		if (os.path.isfile("srclist.txt")):
			with open("srclist.txt", "r") as sfp:
				for name in sfp:
					if (not (name[0] == "#")):
						if (not os.path.isfile(name.strip("\n").strip(" "))):
							hard.append(name)
		else:
			soft.append("srclist.txt -- cannot determine if source files exist.")
		# Check if the physical channels exist
		for i in range(len(submit.chfiles)):
			if (not os.path.isfile(submit.chfiles[i])):
				hard.append(submit.chfiles[i])
		# Check if the ecc files exist
		for i in range(len(submit.ecfiles)):
			for j in range(len(submit.ecfiles[i])):
				if (not os.path.isfile(submit.ecfiles[i][j])):
					if (submit.ecfiles[i][j].endswith("npy")):
						soft.append(submit.ecfiles[i][j])
					else:
						hard.append(submit.ecfiles[i][j])

		if (not os.path.isfile(submit.inputfile)):
			hard.append(submit.inputfile)
		if (not os.path.isfile(submit.scheduler)):
			hard.append(submit.scheduler)

	if (len(hard) > 0):
		print("\033[2m\033[91mError: The following essential files were missing.\033[0m")
		for i in range(len(hard)):
			print("\033[2m%s\033[0m" % (hard[i]))
	else:
		if (len(soft) > 0):
			print("\033[2m\033[93mWarning: The following files were missing.\033[0m")
			for i in range(len(soft)):
				print("\033[2m\033[93m%s\033[0m" % (soft[i]))
		else:
			print("\033[92m\033[2m_/ Everything seems to be OK!\033[0m")
	# Print the amount of resources that will be used up by a simulation.
	Usage(submit)
	return None


def Save(submit):
	# Write a file named const.txt with all the parameter values selected for the simulation
	# File containing the constant parameters
	# Input file
	# Change the timestamp before saving to avoid overwriting the input file.
	with open(submit.inputfile, 'w') as infid:
		# Time stamp
		infid.write("# Time stamp\ntimestamp %s\n" % submit.timestamp)
		# Code type
		infid.write("# Type of quantum error correcting code\necc %s\n" % ",".join([submit.eccs[i].name for i in range(len(submit.eccs))]))
		# Type of noise channel
		infid.write("# Type of quantum channel\nchannel %s\n" % submit.channel)
		# Channel representation
		infid.write("# Representation of the quantum channel. (Available options: \"krauss\", \"process\", \"choi\", \"chi\", \"stine\")\nrepr %s\n" % submit.repr)
		# Noise range parameters
		infid.write("# Noise rate exponents. The actual noise rate is (2/3)^exponent.\nnoiserange %g,%g,%g" % (submit.noiserange[0][0], submit.noiserange[0][-1], submit.noiserange[0].shape[0]))
		for i in range(1, len(submit.noiserange)):
			infid.write(";%g,%g,%g" % (submit.noiserange[i][0], submit.noiserange[i][-1], submit.noiserange[i].shape[0]))
		infid.write("\n")
		# Scale of noise range
		infid.write("# Scale of noise range.\nscale %g\n" % (submit.scale))
		# Number of samples
		infid.write("# Number of samples\nsamples %d\n" % submit.samps)
		# File name containing the parameters to be run on the particular node
		infid.write("# Parameters schedule\nscheduler %s\n" % (submit.scheduler))
		# The parameter list that mist be evaluated in the current node.
		infid.write("# Parameters to be sampled within a node. It would be replaced with values by bqtools.\ncurrent ~~node~~\n")
		# Decoder
		infid.write("# Decoder to be used -- 0 for soft decoding and 1 for Hard decoding.\ndecoder %d\n" % (submit.decoder))
		# ECC frame to be used
		infid.write("# Logical frame for error correction (Available options: \"[P] Pauli\", \"[C] Clifford\", \"[PC] Pauli + Logical Clifford\").\nframe %s\n" % submit.eccframes.keys()[submit.eccframes.values().index(submit.frame)])
		# Number of concatenation levels
		infid.write("# Number of concatenation levels\nlevels %d\n" % submit.levels)
		# Number of decoding trials per level
		infid.write("# Number of syndromes to be sampled at top level\nstats %d\n" % submit.stats)
		# Importance distribution
		infid.write("# Importance sampling methods (Available options: [\"N\"] None, [\"A\"] Power law sampling, [\"B\"] Noisy channel)\nimportance %s\n" % (submit.samplingOptions.keys()[submit.samplingOptions.values().index(submit.importance)]))
		# Metrics to be computed on the logical channel
		infid.write("# Metrics to be computed on the effective channels at every level.\nmetrics %s\n" % ",".join(submit.metrics))
		# Load distribution on cores.
		infid.write("# Load distribution on cores.\ncores %s\n" % (",".join(map(str, submit.cores))))
		# Number of nodes
		infid.write("# Number of nodes\nnodes %d\n" % submit.nodes)
		# Job name
		infid.write("# Name of the host computer.\nhost %s\n" % (submit.host))
		# Job name
		infid.write("# Batch name.\njob %s\n" % (submit.job))
		# Wall time
		infid.write("# Wall time in hours.\nwall %d\n" % (submit.wall))
		# Queue
		infid.write("# Submission queue (Available options: see goo.gl/pTdqbV).\nqueue %s\n" % (submit.queue))
		# Output directory
		infid.write("# Output result\'s directory.\noutdir %s\n" % (os.path.dirname(submit.outdir)))
			
	# Append the content of the input file to the log file.
	if (os.path.isfile("bqsubmit.dat")):
		os.system(("echo \"\\n****************** Created on %s *************\\n++++++++++\\nInput file\\n++++++++++\n$(cat %s)\\n++++++++++\\nbqsubmit file\\n++++++++++\\n$(cat ./../bqsubmit.dat)\\n\\n\" >> log.txt" % (submit.timestamp, submit.inputfile)))
	else:
		os.system(("echo \"\\n****************** Created on %s *************\\n++++++++++\\nInput file\\n++++++++++\n$(cat %s)\\n++++++++++\\nbqsubmit file\\n++++++++++\\nNot provided\\n\\n\" >> log.txt" % (submit.timestamp, submit.inputfile)))
	return None


def LoadSub(submit, subid, isgen):
	# Load the parameters of a submission from an input file
	# If the input file is provided as the submission id, load from that input file.
	# Else if the time stamp is provided, search for the corresponding input file and load from that.
	inputfile = ("./../input/%s.txt" % (subid))
	exists = 0
	if (os.path.exists(inputfile)):
		with open(inputfile, 'r') as infp:
			for (lno, line) in enumerate(infp):
				if (line[0] == "#"):
					pass
				else:
					(variable, value) = line.strip("\n").strip(" ").split(" ")
					Update(submit, variable.strip("\n").strip(" "), value.strip("\n").strip(" "))
		exists = 1
	else:
		print("\033[2mInput file not found.\033[0m")
		exists = 0
	return exists


def Clean(submit, git = 0):
	# Clean up all the files that are associated with compilation and execution.
	# Remove the files generated by the Cython compiler -- *.c and *.so and build/ directory in the simulate/ folder.
	if (not (os.path.exists("./../.gitignore/"))):
		os.mkdir("./../.gitignore/")
	if (os.path.isfile("srclist.txt")):
		with open("srclist.txt", "r") as sfp:
			for name in sfp:
				for extn in ["c", "so", "pyc"]:
					fname = ("%s.%s" % (name.strip("\n").strip(" ").split(".")[0], extn))
					if (os.path.isfile(fname)):
						# print("Removing file %s" % (fname))
						os.system("mv %s ./../.gitignore/ > /dev/null 2>&1" % (fname))
	os.system("mv simulate/build/ ./../.gitignore/ > /dev/null 2>&1")
	os.system("rm -rf ./../temp/")
	if (git == 1):
		# Remove all *.sh files from cluster/
		os.system("mv cluster/*.sh ./../.gitignore/ > /dev/null 2>&1")
		# Remove all input files except for the template one.
		os.system("find ./../input ! -name \'sample_*.txt\' -type f -exec mv \'{}\' ./../.gitignore/ \\; > /dev/null 2>&1")
		# Remove all physical channels
		os.system("mv ./../physical/*.npy ./../.gitignore/ > /dev/null 2>&1")
		# Remove the bqsubmit files
		os.system("mv ./../bqsubmit.dat ./../.gitignore/ > /dev/null 2>&1")
		os.system("mv ./../log.txt ./../.gitignore/ > /dev/null 2>&1")
	return None
