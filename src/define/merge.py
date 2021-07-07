import os
import time
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from define.submission import Submission
from define.utils import LoadTimeStamp
from define.save import Save, Schedule, PrepOutputDir
from define.fnames import PhysicalChannel, LogicalChannel, LogChanVariance, LogicalErrorRate, LogErrVariance, RawPhysicalChannel, NRWeightsFile


def MergeSubs(newname, *submits):
	"""
	Merge a bunch of submissions, to create a larger one with more channels.
	The submissions being merged should have the same:
	(i) Number of parameters defining a noise rate.
	(ii) Number of syndrome samples.
	"""
	combined_submit = Submission()
	combined_submit.outdir = submits[0].outdir
	if newname is None:
		newname = time.strftime("%d/%m/%Y %H:%M:%S").replace("/", "_").replace(":", "_").replace(" ", "_")
	LoadTimeStamp(combined_submit, newname)
	PrepOutputDir(combined_submit)

	# print("submits[0].outdir = {}\ncombined_submit.outdir = {}".format(submits[0].outdir, combined_submit.outdir))

	# folders associated with a submission: input, channels, metrics, results, physical.
	combined_submit.noiserange = None
	CopyConsts(submits[0], combined_submit)
	combined_submit.misc = "+".join([submits[i].timestamp for i in range(len(submits))])

	merged_rates = []
	combined_submit.samps = max(submits[0].samps, submits[1].samps)
	visited = np.zeros(submits[1].noiserates.shape[0], dtype=np.int)
	for i in tqdm(range(submits[0].noiserates.shape[0]), ascii=True, desc="Merging channels:"):
		# Is rate[i] in the second matrix?
		# If yes: combine the samples. and set visited[j] = 1
		# If no: then add rate[i] to merged.
		# Add all rate[j] to merged for which visited[j] = 0.
		# for j in range(submits[1].noiserates.shape[0]):
		#     if np.allclose(submits[0].noiserates[i], submits[1].noiserates[j]) == 1:
		#         arg_found = j
		#         break
		args_found = np.argwhere([np.allclose(rate, submits[0].noiserates[i]) for rate in submits[1].noiserates])[0]
		if len(args_found) == 0:
			arg_found = -1
		else:
			arg_found = args_found[0]
		if arg_found > -1:
			# print("Overlapping noise rate: {}".format(submits[0].noiserates[i, :]))
			visited[arg_found] = 1
			# Add to merged rates.
			merged_rates.append(submits[0].noiserates[i])
			# Combine samples of rate[i] from db[0] and rate[arg_found] from db[1]
			start = time.time()
			CombinePhysicalChannels(submits, i, arg_found, combined_submit)
			# Copy the logical channels and metrics in db[0] for rate[i].
			CopyElements(submits[0], combined_submit, [i], sample_offset=0)
			CopyElements(submits[1], combined_submit, [arg_found], sample_offset=submits[0].samps)
		else:
			# Add to merged rates.
			merged_rates.append(submits[0].noiserates[i])
			# Copy all the elements of db[0] for rate[i] into the merged folder.
			CopyElements(submits[0], combined_submit, [i], sample_offset=0)
	# Add to merged rates.
	merged_rates.extend(submits[1].noiserates[np.nonzero(1 - visited)[0]])
	combined_submit.noiserates = np.array(merged_rates, dtype=np.double)
	# For all j which is not visited, add elements corresponding to rate[j] from db[1] to merged.
	CopyElements(submits[1], combined_submit, np.nonzero(1 - visited)[0])
	# Save the input file, sceduler and prepare the output directory.
	Schedule(combined_submit)
	combined_submit.noiserates = "%s/input/rates_%s.txt" % (
		combined_submit.outdir,
		combined_submit.timestamp,
	)
	np.savetxt(combined_submit.noiserates, np.array(merged_rates, dtype=np.double))
	Save(combined_submit)
	PrepOutputDir(combined_submit)
	return combined_submit.timestamp


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
	merged.stats = dset.stats
	merged.plotsettings = dset.plotsettings
	return None


def CopyElements(source, destn, rates, sample_offset=0):
	"""
	Copy logical channels and metrics from one dataset to another.
	"""
	for i in rates:
		# # Copy the physical channel
		# if sample_offset == 0:
		#     copyfile(
		#         PhysicalChannel(source, source.noiserates[i, :]),
		#         PhysicalChannel(destn, source.noiserates[i, :]),
		#     )
		# Copy the logical channels and metrics for rate[i], from source into destn.
		for s in range(source.samps):
			copyfile(
				LogicalChannel(source, source.noiserates[i, :], s),
				LogicalChannel(destn, source.noiserates[i, :], s + sample_offset),
			)
			copyfile(
				LogChanVariance(source, source.noiserates[i, :], s),
				LogChanVariance(destn, source.noiserates[i, :], s + sample_offset)
			)
			for m in range(len(source.metrics)):
				copyfile(
					LogicalErrorRate(source, source.noiserates[i, :], s, source.metrics[m]),
					LogicalErrorRate(destn, source.noiserates[i, :], s + sample_offset, source.metrics[m]),
				)
				copyfile(
					LogErrVariance(source, source.noiserates[i, :], s, source.metrics[m]),
					LogErrVariance(destn, source.noiserates[i, :], s + sample_offset, source.metrics[m]),
				)
	return None


def CombinePhysicalChannels(submits, rateidx1, rateidx2, combined_submit):
	"""
	Merge the physical channels from the submissions of the corresponding noise rates.
	"""
	phy_part1 = np.load(PhysicalChannel(submits[0], submits[0].noiserates[rateidx1, :]))
	phy_part2 = np.load(PhysicalChannel(submits[1], submits[1].noiserates[rateidx2, :]))
	fname = PhysicalChannel(combined_submit, submits[0].noiserates[rateidx1, :])
	combined = np.vstack((phy_part1, phy_part2))
	np.save(fname, combined)
	if combined_submit.samps < combined.shape[0]:
		combined_submit.samps = combined.shape[0]
	
	raw_part1 = np.load(RawPhysicalChannel(submits[0], submits[0].noiserates[rateidx1, :]))
	raw_part2 = np.load(RawPhysicalChannel(submits[1], submits[1].noiserates[rateidx2, :]))
	fname = RawPhysicalChannel(combined_submit, submits[0].noiserates[rateidx1, :])
	np.save(fname, np.vstack((raw_part1, raw_part2)))
	
	nrw_part1 = np.load(NRWeightsFile(submits[0], submits[0].noiserates[rateidx1, :]))
	nrw_part2 = np.load(NRWeightsFile(submits[1], submits[1].noiserates[rateidx2, :]))
	fname = NRWeightsFile(combined_submit, submits[0].noiserates[rateidx1, :])
	np.save(fname, np.vstack((nrw_part1, nrw_part2)))

	return None