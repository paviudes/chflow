import os
import sys
import time

try:
	import numpy as np
	import multiprocessing as mp
except:
	pass
from simulate.benchmark import Benchmark
from define import qcode as qec
from define import qchans as qch
from define import submission as sub
from define import fnames as fn
from define.metrics import InfidelityPhysical
from define.decoder import PrepareChannelDecoder, TailorDecoder
from cluster import cluster as cl


def SimulateSampleIndex(submit, rate, sample, coreidx, results):
	# Simulate each noise rate and sample
	# Check if simulations results already exist. if yes, do not overwrite.
	if submit.overwrite == 0:
		if os.path.isfile(fn.LogicalChannel(submit, rate, sample)):
			print(
				"Data already exists in : {}".format(
					fn.LogicalChannel(submit, rate, sample)
				)
			)
			results.put((coreidx, rate, sample, 0))
			return None
	start = time.time()
	np.random.seed()
	## Load the physical channel and the reference (noisier) channel if importance sampling is selected.
	physchan = np.load(fn.PhysicalChannel(submit, rate))[sample, :]
	rawchan = None
	if submit.iscorr == 0:
		infidelity = -1
	elif submit.iscorr == 2:
		infidelity = InfidelityPhysical(
			physchan, {"corr": submit.iscorr, "qcode": submit.eccs[0]}
		)
	else:
		rawchan = np.load(fn.RawPhysicalChannel(submit, rate))[sample, :]
		infidelity = InfidelityPhysical(rawchan, {"corr": submit.iscorr})

	# if submit.decoders[0] == 1:
	# 	# print("Bias = {}^{} = {}".format(submit.scales[1], rate[1], np.power(submit.scales[1], rate[1])))
	# 	TailorDecoder(submit.eccs[0], submit.channel, submit.levels, np.power(submit.scales[1], rate[1])) # Comment this for using the traditional min-weight.
	# 	# print("Lookup table given to backend\n{}".format(submit.eccs[0].lookup))

	if submit.decoders[0] == 2:
		refchan = PrepareChannelDecoder(submit, rate, sample)
		# print("Shape of physchan : {} refchan : {}".format(physchan.shape, refchan.shape))
	else:
		refchan = np.zeros_like(physchan)
	# print("Refchan entries : {}".format(refchan))

	# if submit.importance == 2:
	#     refchan = np.load(fn.PhysicalChannel(submit, rate, sample))[sample, :]
	# else:
	#     refchan = np.zeros_like(physchan)

	## Benchmark the noise model.
	print("Infidelity = %.14f" % (infidelity))
	# G = physchan.reshape(256, 256)
	# print("process[0, 0] = {}".format(G[0,0]))
	# print("process[0, :] = {}".format(G[0,:]))
	# print("nonzero(chi) = {}".format(np.nonzero(rawchan < 0)))
	# print("sum(chi) = {}".format(np.sum(rawchan)))
	# print("1 - chi[0,0] = {}".format(1 - rawchan[0]))
	Benchmark(submit, rate, sample, physchan, refchan, infidelity, rawchan)
	####
	runtime = time.time() - start
	results.put((coreidx, rate, sample, runtime))
	return None


def LogResultsToStream(submit, stream, endresults):
	# Print the results on to a file or stdout.
	for i in range(len(endresults)):
		(coreindex, rate, sample, runtime) = endresults[i]
		# Load the logical channels
		logchans = np.load(fn.LogicalChannel(submit, rate, sample))
		# Load the metrics
		metvals = np.zeros(
			(len(submit.metrics), 1 + submit.levels), dtype=np.longdouble
		)
		variance = np.zeros(
			(len(submit.metrics), 1 + submit.levels), dtype=np.longdouble
		)
		for m in range(len(submit.metrics)):
			metvals[m, :] = np.load(
				fn.LogicalErrorRate(submit, rate, sample, submit.metrics[m])
			)
			variance[m, :] = np.load(
				fn.LogErrVariance(submit, rate, sample, submit.metrics[m])
			)
		stream.write("Core %d:\n" % (coreindex + 1))
		stream.write(
			"    Noise rate: %s or %s\n"
			% (np.array_str(rate), np.array_str(np.power(submit.scales, rate)))
		)
		stream.write("    sample = %d\n" % (sample))
		stream.write("    Runtime: %g seconds.\n" % (runtime))
		stream.write("\tMetrics\n")
		stream.write("\txxxxxxxxxxxxxxx\n")
		stream.write("\t{:<10}".format("Level"))
		for m in range(len(submit.metrics)):
			stream.write(" {:<20}".format(submit.metrics[m])),
		stream.write("\n")
		for l in range(submit.levels + 1):
			stream.write("\t{:<10}".format("%d" % (l)))
			for m in range(len(submit.metrics)):
				stream.write(" {:<12}".format("%g" % (metvals[m, l]))),
				stream.write(" {:<12}".format(" +/- %g" % (variance[m, l]))),
			stream.write("\n")
		stream.write("xxxxxxxxxxxxxxx\n")
		stream.write("Average logical channels\n")
		for l in range(submit.levels + 1):
			stream.write(
				"\tLevel %d\n%s\n\t--------\n" % (l, np.array_str(logchans[l, :, :]))
			)
		stream.write("*******************\n")
	stream.write("************** Finished batch **************\n")
	return None


def LocalSimulations(submit, node, stream=sys.stdout):
	# run all simulations designated for a node.
	# All the parameters are stored in the scheduler file. Each parameter must be run in a separate core.
	params = []
	with open(submit.scheduler, "r") as schfp:
		isfound = 0
		for (lno, line) in enumerate(schfp):
			if len(line.strip("\n").strip(" ")) > 0:
				if isfound == 1:
					if line.strip("\n").strip(" ")[0] == "!":
						break
					params.append(
						list(map(np.float64, line.strip("\n").strip(" ").split(" ")))
					)
				if line.strip("\n").strip(" ") == ("!!node %d!!" % (node)):
					isfound = 1

	params = np.array(params)
	# print("params: {}".format(params))
	submit.cores[0] = min(submit.cores[0], params.shape[0])
	print("Parameters to be simulated in node %d with %d cores.\n%s" % (node, min(params.shape[0], submit.cores[0]), np.array_str(params)))
	if submit.host == "local":
		availcores = mp.cpu_count()
	else:
		availcores = cl.GetCoresInNode()
	finished = 0
	while finished < submit.cores[0]:
		stream.write(
			"Code: %s\n"
			% (" X ".join([submit.eccs[i].name for i in range(len(submit.eccs))]))
		)
		stream.write("Channel: %s\n" % (qch.Channels[submit.channel]["name"]))
		stream.write(
			"Noise rates: %s\n"
			% (
				np.array_str(
					params[finished : min(submit.cores[0], finished + availcores), :-1]
				)
			)
		)
		stream.write(
			"Samples: %s\n"
			% (
				np.array_str(
					params[finished : min(submit.cores[0], finished + availcores), -1]
				)
			)
		)
		stream.write("Importance: %g\n" % (submit.importance))
		stream.write("Decoder: %s\n" % np.array_str(submit.decoders))
		if submit.decoders[0] > 1:
			stream.write(
				"Fraction of Pauli probabilities for the ML Decoder: %s\n"
				% submit.decoder_fraction
			)
		if submit.hybrid > 0:
			stream.write("Decoding bins: {}\n".format(submit.decoderbins))
		stream.write("Concatenation levels: %d\n" % (submit.levels))
		stream.write("Decoding trials per level: %s\n" % (np.array_str(submit.stats)))
		stream.write(
			"Metrics to be computed at every level: %s\n" % (", ".join(submit.metrics))
		)
		stream.write("---------------------------\n")
		stream.write("Calculating...\n")
		processes = []
		nproc = min(availcores, submit.cores[0] - finished)
		results = mp.Queue()
		results.cancel_join_thread()
		for p in range(nproc):
			processes.append(
				mp.Process(
					target=SimulateSampleIndex,
					args=(
						submit,
						params[finished + p, :-1],
						np.int64(params[finished + p, -1]),
						p,
						results,
					),
				)
			)
			# SimulateSampleIndex(submit, params[finished + p, :-1], np.int64(params[finished + p, -1]), p, results)
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
			stream.write("The user has interrupted the process!\n")
			for p in range(nproc):
				processes[p].terminate()
				processes[p].join()
			interrupted = 1

		if interrupted == 0:
			# collect the results
			# Print the results to either a file or to stdout.
			LogResultsToStream(submit, stream, endresults)
		else:
			stream.write("************** Exited batch **************\n")

		finished = finished + availcores
	stream.write("************** Finished all batches **************\n")
	return None
