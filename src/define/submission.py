import os
import sys
import time
try:
	import numpy as np
	import itertools as it
except:
	pass
# Files from the define module
import fnames as fn
import qcode as qec

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
		self.email = "X"
		
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
		self.scales = []
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

		# Plot settings -- color, marker, linestyle
		self.plotsettings = ["k", "o", "--"]

		# Output options
		self.outdir = fn.OutputDirectory(os.path.abspath("./../../"), self)
		
def Scheduler(timestamp):
	# name of the scheduler file.
	return ("./../input/scheduler_%s.txt" % (timestamp))

def InputFile(timestamp):
	# name of the script containing the commands to be run
	return ("./../input/%s.txt" % timestamp)


def ChangeTimeStamp(submit, timestamp):
	# change the timestamp of a submission and all the related values to the timestamp.
	submit.timestamp = timestamp
	# Re define the variables that depend on the time stamp
	submit.inputfile = InputFile(submit.timestamp)
	submit.scheduler = Scheduler(submit.timestamp)
	submit.outdir = fn.OutputDirectory(os.path.dirname(submit.outdir), submit)
	return None


def Schedule(submit):
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


def Update(submit, pname, newvalue):
	# Update the parameters to be submitted
	if (pname == "timestamp"):
		ChangeTimeStamp(submit, newvalue)

	elif (pname == "ecc"):
		submit.isSubmission = 1
		# Read all the Parameters of the error correcting code
		names = newvalue.split(",")
		submit.ecfiles = []
		submit.levels = len(names) 
		for i in range(submit.levels):
			submit.eccs.append(qec.QuantumErrorCorrectingCode(names[i]))
			qec.Load(submit.eccs[i])
			submit.ecfiles.append(submit.eccs[i].defnfile)
	
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
		if (len(submit.scales) == 0):
			submit.scaled = [1 for __ in range(len(newRanges))]

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

	elif (pname == "email"):
		submit.email = newvalue
	
	elif (pname == "scheduler"):
		submit.scheduler = newvalue

	elif (pname == "outdir"):
		submit.outdir = fn.OutputDirectory(newvalue, submit)

	elif (pname == "scale"):
		submit.scales = np.array(map(np.longdouble, newvalue.split(",")), dtype = np.longdouble)

	elif (pname == "plot"):
		submit.plotsettings = newvalue.split(",")

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
	if (submit.scales[0] == 1):
		print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Noise range", "%s." % (np.array_str(submit.noiserange[0], max_line_width = 150))))
	else:
		print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Noise range", "%s." % (np.array_str(np.power(submit.scales[0], submit.noiserange[0]), max_line_width = 150))))
	for i in range(1, len(submit.noiserange)):
		if (submit.scales[i] == 1):
			print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("", "%s." % (np.array_str(submit.noiserange[i], max_line_width = 150))))
		else:
			print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("", "%s." % (np.array_str(np.power(submit.scales[i], submit.noiserange[i]), max_line_width = 150))))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Scales of noise rates", "%s" % (np.array_str(submit.scales))))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Number of Samples", "%d" % (submit.samps)))

	print("Metrics")
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Logical metrics", "%s" % (", ".join(submit.metrics))))

	print("Error correction")
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("QECC", "%s" % (" X ".join([submit.eccs[i].name for i in range(len(submit.eccs))]))))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("[[N, K, D]]", "[[%d, %d, %d]]" % (reduce((lambda x,y: x * y), [submit.eccs[i].N for i in range(len(submit.eccs))]), reduce((lambda x,y: x * y), [submit.eccs[i].K for i in range(len(submit.eccs))]), reduce((lambda x,y: x * y), [submit.eccs[i].D for i in range(len(submit.eccs))]))))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Levels of concatenation", "%d" % (submit.levels)))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("ECC frame", "%s" % (submit.eccframes.keys()[submit.eccframes.values().index(submit.frame)])))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Decoder", "%d" % (submit.decoder)))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Syndrome samples at level %d" % (submit.levels), "%d" % (submit.stats)))
	print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Type of syndrome sampling", "%s" % (submit.samplingOptions.keys()[submit.samplingOptions.values().index(submit.importance)])))
	
	if (not (submit.host == "local")):
		print("Mammouth")
		print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Host", "%s" % (submit.host)))
		print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Job name", "%s" % (submit.job)))
		print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Number of nodes", "%d" % (submit.nodes)))
		print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Walltime per node", "%d" % (submit.wall)))
		print(("{:<%d} {:<%d}" % (colwidth, colwidth)).format("Submission queue", "%s" % (submit.queue)))
	
		print("Usage")
		mam.Usage(submit)
	print("\033[0m"),
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
		infid.write("# Scales of noise range.\nscale %s\n" % (",".join(map(str, submit.scales))))
		# Number of samples
		infid.write("# Number of samples\nsamples %d\n" % submit.samps)
		# File name containing the parameters to be run on the particular node
		infid.write("# Parameters schedule\nscheduler %s\n" % (submit.scheduler))
		# Decoder
		infid.write("# Decoder to be used -- 0 for soft decoding and 1 for Hard decoding.\ndecoder %d\n" % (submit.decoder))
		# ECC frame to be used
		infid.write("# Logical frame for error correction (Available options: \"[P] Pauli\", \"[C] Clifford\", \"[PC] Pauli + Logical Clifford\").\nframe %s\n" % (submit.eccframes.keys()[submit.eccframes.values().index(submit.frame)]))
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
		# Queue
		infid.write("# Email notifications.\nemail %s\n" % (submit.email))
		# Output directory
		infid.write("# Output result\'s directory.\noutdir %s\n" % (os.path.dirname(submit.outdir)))
	# Append the content of the input file to the log file.
	if (os.path.isfile("bqsubmit.dat")):
		os.system(("echo \"\\n****************** Created on %s *************\\n++++++++++\\nInput file\\n++++++++++\n$(cat %s)\\n++++++++++\\nbqsubmit file\\n++++++++++\\n$(cat ./../bqsubmit.dat)\\n\\n\" >> log.txt" % (submit.timestamp, submit.inputfile)))
	else:
		os.system(("echo \"\\n****************** Created on %s *************\\n++++++++++\\nInput file\\n++++++++++\n$(cat %s)\\n++++++++++\\nbqsubmit file\\n++++++++++\\nNot provided\\n\\n\" >> log.txt" % (submit.timestamp, submit.inputfile)))
	return None

def PrepOutputDir(submit):
	# Prepare the output directory -- create it, put the input files.
	# Copy the necessary input files, error correcting code.
	if (not (os.path.exists(submit.outdir))):
		os.mkdir(submit.outdir)
	for subdir in ["input", "code", "physical", "channels", "metrics", "results"]:
		if (not (os.path.exists("%s/%s" % (subdir, submit.outdir)))):
			os.mkdir("%s/%s" % (submit.outdir, subdir))
	# Copy the relevant code data
	for l in range(submit.levels):
		os.system("cp %s %s/code/" % (submit.eccs[l].defnfile, submit.outdir))
	# Copy the physical channels data
	for i in range(submit.noiserates.shape[0]):
		os.system("cp ./../physical/%s %s/physical/" % (fn.PhysicalChannel(submit, submit.noiserates[i, :], loc = "local"), submit.outdir))
	os.system("cp ./../input/%s.txt %s/input/" % (submit.timestamp, submit.outdir))
	os.system("cp ./../input/scheduler_%s.txt %s/input/" % (submit.timestamp, submit.outdir))
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
	
	os.system("find ./../ -maxdepth 3 \( -name \"*.pbs\" -o -name \"*.out\" -o -name \"*.aux\" -o -name \"*.log\" -o -name \"*.fls\" -o -name \"*.bbl\" -o -name \"*.synctex.gz\" -o -name \"*.pyc\" -o -name \"*.c\" -o -name \"*.so\" \) -type f -exec mv \'{}\' ./../.gitignore/ \\; > /dev/null 2>&1")
	# Remove all latex generated files from docs/	
	os.system("find ./../docs \( -name \"*.out\" -o -name \"*.aux\" -o -name \"*.log\" -o -name \"*.fls\" -o -name \"*.bbl\" -o -name \"*.synctex.gz\" \) -type f -exec mv \'{}\' ./../.gitignore/ \\; > /dev/null 2>&1")
	os.system("mv simulate/build/ ./../.gitignore/simulate_build > /dev/null 2>&1")
	os.system("mv analyze/build/ ./../.gitignore/analyze_build > /dev/null 2>&1")
	if (git == 1):
		# remove compiler logs
		os.system("mv simulate/compiler_output.txt ./../.gitignore/simulate_compiler.txt > /dev/null 2>&1")
		os.system("mv analyze/compiler_output.txt ./../.gitignore/analyze_compiler.txt > /dev/null 2>&1")
		# Remove all input files except for the template one.
		os.system("find ./../input ! -name \'sample_*.txt\' -type f -exec mv \'{}\' ./../.gitignore/ \\; > /dev/null 2>&1")
		# Remove all physical channels
		os.system("mv ./../physical/*.npy ./../.gitignore/ > /dev/null 2>&1")
		# Remove the bqsubmit files
		os.system("mv ./../bqsubmit.dat ./../.gitignore/ > /dev/null 2>&1")
		os.system("mv ./../log.txt ./../.gitignore/ > /dev/null 2>&1")
	return None
