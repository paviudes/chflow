import os
import sys
import time
import numpy as np
# Files from the "cluster" module.
from cluster import mammouth as mam
# Files from the "define" module.
from define import qcode as qec
from define import verifychans as vc
from define import fnames as fn
from define import submission as sub
from define import metrics as ml
from define import qchans as qc
from define import genchans as chgen
from define import chandefs as chdef
from define import chanapprox as capp
from define import photonloss as ploss
from define import gendamp as gdamp
from define import chanreps as crep
# Files from the "analyze" module.
from analyze import collect as cl
from analyze import plots as pl

if __name__ == '__main__':
	availChReps = map(lambda rep: ("\"%s\"" % (rep)), ["krauss", "choi", "chi", "process", "stine"])
	availMetrics = map(lambda met: "\"%s\"" % (met), ml.Metrics.keys())
	availChans = map(lambda chan: "\"%s\"" % (chan), qc.Channels.keys())
	mannual = {"qcode":["Load a quantum error correcting code.",
						"qcode s(string)\n\twhere s is the name of the file containing the details of the code."],
			   "qcbasis":["Output the Canonical basis for the quantum error correcting code.",
			   			  "No parameters."],
			   "qcminw":["Prepare the syndrome lookup table for (hard) minimum weight decoding algorithm.",
			   			 "No parameters."],
			   "qcprint":["Print all the details on the underlying quantum error correcting code.",
			   			  "No parameters."],
			   "chan":["Load a channel.",
					   "chan s1(string) x1(float),x2(float),...\n\twhere s1 is either the name of a channel or a file name containing the channel information x1,x2,... specify the noise rates."],
			   "chsave":["Save a channel into a file.",
			   			"chsave s1(string)\n\twhere s1 is the name of the file."],
			   "chrep":["Convert from its current representation to another form.",
			   			"chrep s1(string)\n\twhere s1 must be one of %s." % (", ".join(availChReps))],
			   "chtwirl":["Twirl a quantum channel.",
			   			  "No parameters."],
			   "chpa":["Honest Pauli Approximation of a quantum channel.",
			   		   "No parameters."],
			   "chmetrics":["Compute standard metric(s).",
			   				"chmetrics s1(string)[,s2(string)]\n\ts1 must be one of %s." % (", ".join(availMetrics))],
			   "chval":["Test if all the properties of a CPTP map are satisfied.",
			   			"No parameters."],
			   "chprint":["Print a quantum channel in its current representation.",
			   			  "No parameters."],
			   "chcalib":["Measure all the norms for a quantum channel.",
			   			  "calibrate s1(string) x11(float),x12(float),n1(int);x21(float),x22(float),n2(int),... [m1(string),m2(string),m3(string),...] [c(int)]\n\twhere s1 must be one of %s\n\t\"xi1\", \"xi2\" and \"ni\" specify the noise range for calibration: from \"xi1\" to \"xi2\", in \"ni\" steps.\n\teach of m1, m2, m3, ... must be one of %s.\n\tc is a number that specified which noise parameter must be varied in the calibration plots. By default it is the first parameter, i.e, c = 0." % (", ".join(availChans), ", ".join(availMetrics))],
			   "sbprint":["Print details about the currently loaded submission.",
			   			  "No parameters."],
			   "sbload":["Create a new submission of channels to be simulated.",
			   		     "load [s1(string)]\n\twhere s1 either a time stamp specifying a submission or a file containing parameters. If no inputs are given, the user will be prompted on the console."],
			   "sbmerge":["Merge the results from two or more simulations.",
			   			  "merge s(string) s1(string)[,s2(string),s3(string)...]\nwhere s is a name of the merged submission and s1, s2,... are names of the simulations to be merged."],
			   "submit":["Create a simulation input record with the current submission parameters.",
			   			 "No parameters."],
			   "sbsave":["Save the current simulation parameters.",
			   		     "No parameters."],
			   "metrics":["Compute metrics for the physical channels in the current submission.",
			   			  "metrics m1(string)[,m2(string),m3(string),...]\nwhere m1,m2,... should be one of %s." % (availMetrics)],
			   "threshold":["Estimate the threshold of the ECC and quantum channel with the current simulation data.",
			   				"threshold s1(string)\nwhere s1 is a metric computed at the logical level."],
			   "compare":["Compare the logical error rates from two different simulations.",
			   			  "compare s1(string) s2(string) m(string)\nwhere s1, s2 are names of the respective simulations to be compared and m is a metric that is computed at the logical level. Note: the logical error rates for the simulations should be available and gathered in the results/ folder."],
			   "ecc":["Run error correction simulations, if the simulation data doesn\'t already exist.",
			   		  "No parameters"],
			   "collect":["Collect results.",
			   			  "No parameters"],
			   "tplot":["Threshold plots for the current database.",
			   		   "tplot s11(string)[,s12(string),s13(string),...] s2(string)\nwhere each of s1i are either metric names or indices of indepenent parameters in the defition of the physical noise model; s2 is the name of the logical metric."],
			   "lplot":["Level-wise plots of the logical error rates vs physical noise rates in the current (and/or other) database(s).",
			   		   "lplot s11(string)[,s12(string),s13(string),...] s2(string) [s31(string),s32(string),...]\nwhere each of s1i are either metric names or indices of indepenent parameters in the defition of the physical noise model; s2 is the name of the logical metric; s3i are time stamps of other simulation records that also need to be plotted alongside."],
			   "lplot2d":["Level-wise 2D (density) plots of the logical error rates vs. a pair of physical noise rates in the current (and/or other) database(s).",
			   		   "lplot2d s11(string),s12(string) s2(string)\nwhere each of s1i are either metric names or indices of indepenent parameters in the defition of the physical noise model; s2 is the name of the logical metric."],
			   "clean":["Remove compilation and run time files.",
			   		  "No parameters."],
			   "man":["Mannual",
			   		  "No parameters."],
			   "quit":["Quit",
			   		   "No parameters."],
			   "exit":["Quit",
			   		   "No parameters."]}
	
	fileinput = 0
	infp = None
	if (len(sys.argv) > 1):
		if (os.path.isfile(sys.argv[1])):
			instructions = sys.argv[1]
			fileinput = 1
			infp = open(instructions, "r")
	isquit = 0
	rep = "process"
	channel = np.zeros((4, 4), dtype = np.longdouble)
	qeccode = None
	submit = sub.Submission()
	
	while (isquit == 0):
		print(">>"),
		if (fileinput == 0):
			try:
				user = map(lambda val: val.strip("\n").strip(" "), raw_input().strip(" ").strip("\n").split(" "))
			except KeyboardInterrupt:
				user = ["quit"]
				print("")
		else:
			user = map(lambda val: val.strip("\n").strip(" "), infp.readline().strip(" ").strip("\n").split(" "))

		
		if (user[0] == "qcode"):
			# define a quantum code
			qeccode = qec.QuantumErrorCorrectingCode(user[1])
			qec.Load(qeccode)

		#####################################################################

		elif (user[0] == "qcbasis"):
			# display the canonical basis for the code
			qec.IsCanonicalBasis(qeccode.S, qeccode.L, qeccode.T, verbose = 1)

		#####################################################################

		elif (user[0] == "qcminw"):
			# prepare a syndrome lookup table for minimum weight decoding
			# Syndrome look-up table for hard decoding.
			print("\033[2mPreparing syndrome lookup table.\033[0m")
			qec.PrepareSyndromeLookUp(qeccode)

		#####################################################################

		elif (user[0] == "qcprint"):
			# print details of the error correcting code
			qec.Print(qeccode)

		#####################################################################

		elif (user[0] == "chan"):
			noiserates = []
			if (len(user) > 2):
				noiserates = map(np.longdouble, user[2].split(","))
			channel = chdef.GetKraussForChannel(user[1], *noiserates)
			rep = "krauss"
			print("\033[2mNote: the current channel is in the \"krauss\" representation.\033[0m")

		#####################################################################

		elif (user[0] == "chsave"):
			qc.Save(user[1], channel, rep)

		#####################################################################

		elif (user[0] == "chrep"):
			channel = np.copy(crep.ConvertRepresentations(channel, rep, user[1]))
			rep = user[1]

		#####################################################################

		elif (user[0] == "chtwirl"):
			proba = capp.Twirl(channel, rep)
			print("\033[2mTwirled channel\n\tE(R) = %g R + %g X R X + %g Y R Y + %g Z R Z.\033[0m" % (proba[0], proba[1], proba[2], proba[3]))

		#####################################################################

		elif (user[0] == "chpa"):
			(proba, proxim) = capp.HonestPauliApproximation(channel, rep)
			print("\033[2mHonest Pauli Approximation\n\tE(R) = %g R + %g X R X + %g Y R Y + %g Z R Z,\n\tand it has a diamond distance of %g from the original channel.\033[0m" % (1 - np.sum(proba, dtype = np.float), proba[0], proba[1], proba[2], proxim))

		#####################################################################

		elif (user[0] == "chprint"):
			qc.Print(channel, rep)
			print("\033[2mxxxxxx\033[0m")

		#####################################################################

		elif (user[0] == "chmetrics"):
			metrics = user[1].split(",")
			if (not (rep == "choi")):
				metvals = ml.ComputeNorms(crep.ConvertRepresentations(channel, rep, "choi"), metrics)
			else:
				metvals = ml.ComputeNorms(channel, metrics)
			print("{:<20} {:<10}".format("Metric", "Value"))
			print("-------------------------------")
			for m in range(len(metrics)):
				print("{:<20} {:<10}".format(ml.Metrics[metrics[m]][0], ("%.2e" % metvals[m])))
			print("xxxxxx")

		#####################################################################

		elif (user[0] == "chcalib"):
			if (len(user) == 3):
				# noise metric, xcol and ycol are not specified.
				out = "fidelity"
				xcol = 0
				ycol = -1
			elif (len(user) == 4):
				# xcol and ycol are not provided
				out = user[3]
				xcol = 0
				ycol = -1
			elif (len(user) == 5):
				# ycol is not provided
				out = int(user[3])
				xcol = int(user[4])
				ycol = -1
			else:
				out = user[3]
				xcol = int(user[4])
				ycol = int(user[5])
			qc.Calibrate(user[1], user[2], out, xcol = xcol, ycol = ycol)	

		#####################################################################

		elif (user[0] == "chval"):
			vc.IsQuantumChannel(channel, rep)

		#####################################################################

		elif (user[0] == "sbprint"):
			sub.PrintSub(submit)
		
		#####################################################################
		
		elif (user[0] == "sbload"):
			if (len(user) == 1):
				# no file name or time stamp was provided.
				print("Console input is not set up currently.")
			else:
				submit = sub.Submission()
				exisis = sub.LoadSub(submit, user[1], 1)
			if (exisis == 1):
				# Generate new physical channels if needed
				if (cl.IsComplete(submit) == 0):
					chgen.PreparePhysicalChannels(submit)
				else:
					# prepare the set of parameters
					submit.params = np.zeros((submit.noiserates.shape[0] * submit.samps, submit.noiserates.shape[1] + 1), dtype = np.longdouble)
					for i in range(submit.noiserates.shape[0]):
						for j in range(submit.samps):
							submit.params[i * submit.samps + j, :-1] = submit.noiserates[i, :]
							submit.params[i * submit.samps + j, -1] = j
			
		#####################################################################

		elif (user[0] == "sbmerge"):
			sub.MergeSubs(submit, *user[1].split(","))
		
		#####################################################################

		elif (user[0] == "submit"):
			sub.ChangeTimeStamp(submit, time.strftime("%d/%m/%Y %H:%M:%S").replace("/", "_").replace(":", "_").replace(" ", "_"))
			sub.Save(submit)
			if (submit.host == "local"):
				mam.Scheduler(submit)
				print("\033[2mFor remote execution, run: \"./chflow.sh %s\"\033[0m" % (submit.timestamp))
			else:
				mam.SubmitOnMammouth(submit)
				sub.Validate(submit)
		
		#####################################################################

		elif (user[0] == "ecc"):
			# Check if the logical channels and error dates already exist for this simulation.
			# If yes, directly display the logical error rates data. Else, simulate error correction.
			if (cl.IsComplete(submit) == 0):
				# Compile the cythonizer file to be able to perform error correction simulations
				# Syndrome look-up table for hard decoding.
				start = time.time()
				if (submit.decoder == 1):
					start = time.time()
					for l in range(submit.levels):
						if (submit.ecc[l].lookup is None):
							print("\033[2mPreparing syndrome lookup table for the %s code.\033[0m" % (submit.eccs[l].name))
							qec.PrepareSyndromeLookUp(submit.eccs[l])
					print("\033[2mHard decoding tables built in %d seconds.\033[0m" % (time.time() - start))
				# Files from the "simulate" module.
				os.system("cd simulate/;python compile.py build_ext --inplace > compiler_output.txt 2>&1;cd ..")
				from simulate import simulate as sim
				# Error correction simulation
				start = time.time()
				print("\033[2mPlease wait ...\033[0m")
				stream = open("perf.txt", "w")
				try:
					for i in range(submit.nodes):
						submit.current = ("%d" % i)
						stnode = time.time()
						sim.LocalSimulations(submit, stream)
						print("\r\033[2m%d%% done, approximately %d seconds remaining ...\033[0m" % (100 * (i + 1)/float(submit.nodes), (submit.nodes - i - 1) * (time.time() - stnode))),
					print("")
				except KeyboardInterrupt:
					print("\033[2mProcess terminated by user.\033[0m")
				# Create a folder with the timestamp as its name and move the channels, metrics data and the input files, bqsubmit.dat data into the timestamp-folder.
				stream.close()
				sim.OrganizeResults(submit)
				print("\033[2mdone, in %d seconds.\033[0m" % (time.time() - start))
			else:
				cl.GatherLogErrData(submit)

		#####################################################################

		elif (user[0] == "collect"):
			# Collect the logical failure rates into one file.
			if (cl.IsComplete(submit) > 0):
				cl.GatherLogErrData(submit)
			
		#####################################################################

		elif (user[0] == "tplot"):
			# Produce threshold plots for a particular logical metric.
			# Plot the logical error rate with respect to the concatnation layers, with a new curve for every physical noise rate.
			# At the threshold in the physical noise strengh, the curves will have a bifurcation.
			pl.ThresholdPlot(user[1], user[2], submit)

		elif (user[0] == "lplot"):
			# Plot the logical error rate with respect to a physical noise strength, with a new figure for every concatenation layer.
			# One or more simulation data can be plotted in the same figure with a new curve for every dataset.
			# One of more measures of physical noise strength can be plotted on the same figure with a new curve for each definition.
			if (len(user) < 4):
				refs = []
			else:
				refs = [sub.LoadSub(sub.Submission(), ts, 1) for ts in user[3].split(",")]
			pl.LevelWisePlot(user[1], user[2], [submit] + refs)

		elif (user[0] == "lplot2d"):
			# Plot the logical error rates with respect to two parameters of the physical noise rate.
			# The plot will be a 2D density plot with the logical error rates represented by the density of the colors.
			print("\033[2m"),
			pl.LevelWisePlot2D(user[1], user[2], submit)
			print("\033[0m"),

		# elif (user[0] == "plot"):
		# 	# plot the gathered logical error rates data
		# 	# The parameters are: physical metric, logical metric, reference datasets.
		# 	# Other plot settings must be controlled in the file plotsettings.txt
		# 	# Also input reference datasets as timestamps
		# 	if (submit.channel == "pl"):
		# 		# ploss.PLThreshPlots(submit, user[1])
		# 		# ploss.PLPerfPlots(submit, user[1])
		# 		ploss.PLPerfPlots2D(submit, user[1])
		# 	elif (submit.channel in ["gd", "gdt", "gdtx"]):
		# 		if (submit.channel == "gd"):
		# 			gdamp.GDPerfPlots(submit, user[1])
		# 		if (len(user) < 3):
		# 			colx = 0
		# 			coly = 1
		# 		else:
		# 			colx = int(user[2])
		# 			coly = int(user[3])
		# 		gdamp.GDPerfTimeScales(submit, user[1], colx = colx, coly = coly)
		# 	else:
		# 		pmet = user[1]
		# 		refs = []
		# 		if (len(user) > 3):
		# 			lmet = user[2]
		# 			refstamps = user[3].split(",")
		# 			for i in range(len(refstamps)):
		# 				refsub = sub.Submission()
		# 				sub.LoadSub(refsub, "input/%s.txt" % (refstamps[i]))
		# 				refs.append(refsub)
		# 		else:
		# 			refs = []
		# 			if (len(user) < 3):
		# 				lmet = submit.metrics[0]
		# 			else:
		# 				lmet = user[2]
		# 		pl.LevelWisePlot(submit, refs, pmet, lmet, maxlevel = 3)
		# 		pl.ThresholdPlot(submit, pmet, lmet, maxlevel = 3)

		#####################################################################

		elif (user[0] == "sbsave"):
			sub.Save(submit)
		
		#####################################################################

		elif (user[0] == "metrics"):
			# compute level-0 metrics.
			physmetrics = user[1].split(",")
			ml.ComputePhysicalMetrics(submit, physmetrics)
		
		#####################################################################

		elif (user[0] == "threshold"):
			if (user[1] in submit.metrics):
				thresh = cl.ComputeThreshold(submit, user[1])
				print("The estimated threshold for the %s code over the %s channel is %g." % (" X ".join([submit.eccs[i].name for i in range(len(submit.eccs))]), submit.channel, thresh))
			else:
				print("Only %s are computed at the logical levels." % (", ".join([ml.Metrics[met][0] for met in submit.metrics])))
		
		#####################################################################

		elif (user[0] == "compare"):
			tocompare = [Submission(), Submission()]
			for i in range(2):
				LoadSub(tocompare[i], user[1 + i])
			if ((user[3] in tocompare[0].metrics) and (user[3] in tocompare[1].metrics)):
				if (os.path.isfile(fn.GatheredLogErrData(tocompare[0], logicalmetric)) and os.path.isfile(fn.GatheredLogErrData(tocompare[1], logicalmetric))):
					CompareSubs(tocompare, user[3])
				else:
					print("Logical error rates data is not available for one of the simulations.")
			else:
				print("The logical metrics that are available in both simulation data are %s." % (", ".join([ml.Metrics[met][0] for met in tocompare[0].metrics if met in tocompare[1].metrics])))
		
		#####################################################################

		elif (user[0] == "man"):
			if (len(user) > 1):
				if (user[1] in mannual):
					print("\t\033[2m\"%s\"\n\tDescription: %s\n\tUsage:\n\t%s\033[0m" % (user[1], mannual[user[1]][0], mannual[user[1]][1]))
				elif (user[1] in qc.Channels):
					print("\t\033[2m\"%s\"\n\tDescription: %s\n\tParameters: %s\033[0m" % (user[1], qc.Channels[user[1]][0], qc.Channels[user[1]][1]))
				elif (user[1] in ml.Metrics):
					print("\t\033[2m\"%s\"\n\tDescription: %s\n\tExpression: %s\033[0m" % (user[1], ml.Metrics[user[1]][0], ml.Metrics[user[1]][-2]))
				else:
					pass
			else:
				print("\tList of commands and thier functions.")
				for (i, item) in enumerate(mannual):
					print("\t%d). \033[2m\"%s\"\n\tDescription: %s\n\tUsage\n\t%s\033[0m" % (i, item, mannual[item][0], mannual[item][1]))
			print("xxxxxx")

		#####################################################################

		elif (user[0] == "clean"):
			git = 0
			if (len(user) > 1):
				if (user[1] == "git"):
					git = 1
			sub.Clean(submit, git)
			
		#####################################################################

		elif (user[0] in ["quit", "exit"]):
			isquit = 1

		#####################################################################

		else:
			print("\033[2mNo action.\033[0m")
			pass

	if (fileinput == 1):
		infp.close()