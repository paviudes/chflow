import os
import sys
import time
import numpy as np
import multiprocessing as mp
from benchmark import Benchmark
from define import qcode as qec
from define import qchans as qch
from define import submission as sub
from define import fnames as fn

def SimulateSampleIndex(submit, rate, sample, coreidx, results):
	# channel, rate, sample, nlevels, nstats, ecmode, metricsToCompute, qcodeinfo, importance, decoder, coreidx, results):
	start = time.time()
	## Load the physical channel and the reference (noisier) channel if importance sampling is selected.
	# print("Physical channel for noise = %g, sample = %d\n%s" % (rate, sample, np.load("physical/%s_%g.npy" % (submit.channel, rate))[sample, :, :]))
	physchan = np.load(fn.PhysicalChannel(submit, rate, loc = "local"))[sample, :, :]
	if (submit.importance == 2):
		refchan = np.load(fn.PhysicalChannel(submit, rate, sample, loc = "local"))[sample, :, :]
	else:
		refchan = np.zeros_like(physchan)
	## Benchmark the noise model.
	Benchmark(submit, rate, sample, physchan, refchan)
	####
	runtime = time.time() - start
	results.put((coreidx, rate, sample, runtime))
	return None


def LogResultsToStream(submit, stream, endresults):
	# Print the results on to a file or stdout.
	for i in range(len(endresults)):
		(coreindex, rate, sample, runtime) = endresults[i]
		# Load the logical channels
		fname = ("./../temp/channels/%s" % os.path.basename(fn.LogicalChannel(submit, rate, sample)))
		logchans = np.load(fname)
		# Load the metrics
		metvals = np.zeros((len(submit.metrics), 1 + submit.levels), dtype = np.longdouble)
		for m in range(len(submit.metrics)):
			fname = ("./../temp/metrics/%s" % os.path.basename(fn.LogicalErrorRate(submit, rate, sample, submit.metrics[m])))
			metvals[m, :] = np.load(fname)
		stream.write("Core %d:\n" % (coreindex + 1))
		stream.write("    Noise rate: %s\n" % (np.array_str(rate)))
		stream.write("    sample = %d\n" % (sample))
		stream.write("    Runtime: %g seconds.\n" % (runtime))
		stream.write("\033[92m \tMetrics\033[0m\n")
		stream.write("\033[92m \txxxxxxxxxxxxxxx\033[0m\n")
		stream.write("\033[92m\t{:<10}\033[0m".format("Level"))
		for m in range(len(submit.metrics)):
			stream.write("\033[92m {:<20}\033[0m".format(submit.metrics[m])),
		stream.write("\n")
		for l in range(submit.levels + 1):
			stream.write("\t\033[92m{:<10}\033[0m".format("%d" % (l)))
			for m in range(len(submit.metrics)):
				stream.write("\033[92m {:<20}\033[0m".format("%g" % (metvals[m, l]))),
			stream.write("\n")
		stream.write("\033[92m xxxxxxxxxxxxxxx\n\033[0m")
		stream.write("\033[92mAverage logical channels\033[0m\n")
		for l in range(submit.levels + 1):
			stream.write("\t\033[92mLevel %d\n%s\n\t--------\033[0m\n" % (l, np.array_str(logchans[l, :, :])))
		stream.write("\033[92m*******************\033[0m\n")
	stream.write("\033[92m************** Finished batch **************\033[0m\n")
	return None


def OrganizeResults(submit):
	# Create a folder whose name is the same as the timestamp.
	# Move the channels and metrics folders.
	# Move the relevant code data and the input data required for this simulation.
	# Move the oupout log-file of the simulation.
	if (not (os.path.exists(submit.outdir))):
		os.mkdir(submit.outdir)
	if (not (os.path.exists("%s/results" % (submit.outdir)))):
		os.mkdir("%s/results" % (submit.outdir))
	# os.system("cp -r ./../src %s/ > /dev/null 2>&1 2>&1" % (submit.outdir))
	# os.system("cp -r ./../physical %s/ > /dev/null 2>&1" % (submit.outdir))
	# os.system("cp -r ./../input %s/ > /dev/null 2>&1" % (submit.outdir))
	# os.system("cp -r ./../code %s/ > /dev/null 2>&1" % (submit.outdir))
	# os.system("cp -r ./../docs %s/ > /dev/null 2>&1" % (submit.outdir))
	os.system("mv ./../temp/perf.txt %s/results/ > /dev/null 2>&1" % (submit.outdir))
	os.system("mv ./../temp/channels/ %s/ > /dev/null 2>&1" % (submit.outdir))
	os.system("mv ./../temp/metrics/ %s/ > /dev/null 2>&1" % (submit.outdir))
	return None


def LocalSimulations(submit, stream):
	# run all simulations designated for a node.
	# All the parameters are stored in the scheduler file. Each parameter must be run in a separate core.
	params = []
	with open(submit.scheduler, "r") as schfp:
		isfound = 0
		for (lno, line) in enumerate(schfp):
			if (len(line.strip("\n").strip(" ")) > 0):
				if (isfound == 1):
					if (line.strip("\n").strip(" ")[0] == "!"):
						break
					params.append(map(np.float, line.strip("\n").strip(" ").split(" ")))
				if (line.strip("\n").strip(" ") == ("!!node %s!!" % (submit.current))):
					isfound = 1

	params = np.array(params)
	submit.cores[0] = min(submit.cores[0], params.shape[0])
	# print("Parameters to be simulated in node %d with %d cores.\n%s" % (submit.current, min(params.shape[0], submit.cores[0]), np.array_str(params)))
	
	if (not (os.path.exists("./../temp/channels/"))):
		os.system("mkdir -p ./../temp/channels/")
	if (not (os.path.exists("./../temp/metrics/"))):
		os.system("mkdir -p ./../temp/metrics/")
	availcores = mp.cpu_count()
	finished = 0
	while (finished < submit.cores[0]):
		stream.write("\033[2mCode: %s\n\033[0m" % (" X ".join([submit.eccs[i].name for i in range(len(submit.eccs))])))
		stream.write("\033[2mChannel: %s\n\033[0m" % (qch.Channels[submit.channel][0]))
		stream.write("\033[2mNoise rates: %s\n\033[0m" % (np.array_str(params[finished:min(submit.cores[0], finished + availcores), :-1])))
		stream.write("\033[2mSamples: %s\n\033[0m" % (np.array_str(params[finished:min(submit.cores[0], finished + availcores), -1])))
		stream.write("\033[2mImportance: %g\n\033[0m" % (submit.importance))
		stream.write("\033[2mDecoder: %d\n\033[0m" % (submit.decoder))
		stream.write("\033[2mConcatenation levels: %d\n\033[0m" % (submit.levels))
		stream.write("\033[2mDecoding trials per level: %d\n\033[0m" % (submit.stats))
		stream.write("\033[2mMetrics to be computed at every level: %s\n\033[0m" % (", ".join(submit.metrics)))
		stream.write("\033[2m---------------------------\n\033[0m")
		stream.write("\033[2mCalculating...\n\033[0m")
		processes = []
		nproc = min(availcores, submit.cores[0] - finished)
		results = mp.Queue()
		results.cancel_join_thread()
		for p in range(nproc):
			processes.append(mp.Process(target = SimulateSampleIndex, args = (submit, params[finished + p, :-1], np.int(params[finished + p, -1]), p, results)))
		for p in range(nproc):
			processes[p].start()
		# wait for all the processes to finish
		interrupted = 0
		endresults = []
		try:
			for p in range(nproc):
				endresults.append(results.get())
			for p in range(nproc):
				processes[p].join()
		except KeyboardInterrupt:
			stream.write("\033[91mThe user has interrupted the process!\033[0m\n")
			for p in range(nproc):
				processes[p].terminate()
				processes[p].join()
			interrupted = 1

		if (interrupted == 0):
			# collect the results
			# Print the results to either a file or to stdout.
			LogResultsToStream(submit, stream, endresults)
		else:
			stream.write("\033[91m************** Exited batch **************\033[0m\n")

		finished = finished + availcores
	stream.write("\033[92m************** Finished all batches **************\033[0m\n")
	return None