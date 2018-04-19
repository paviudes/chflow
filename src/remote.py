import os
import sys
import time
from define import submission as sub
from define import qcode as qec
from simulate import simulate as sim
# Force the module scripts to run locally -- https://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path
# import inspect as ins
# current = os.path.realpath(os.path.abspath(os.path.dirname(ins.getfile(ins.currentframe()))))
# if (not (current in sys.path)):
# 	sys.path.insert(0, current)

# Simulate a noise channel, by computing the database of logical channels for several rounds of error correction.
if (not os.path.exists("./../temp/")):
	os.mkdir("./../temp/")
stream = open("./../temp/perf.txt", "w")
submit = sub.Submission()
if (len(sys.argv) > 1):
	timestamp = sys.argv[1].strip("\n").strip(" ")
	sub.LoadSub(submit, timestamp, 0)
else:
	print("Console input is not yet set up.")
	# ConsoleInput(submit)
# Syndrome look-up table for hard decoding.
if (submit.decoder == 1):
	start = time.time()
	for l in range(submit.levels):
		if (submit.ecc[l].lookup is None):
			print("\033[2mPreparing syndrome lookup table for the %s code.\033[0m" % (submit.eccs[l].name))
			qec.PrepareSyndromeLookUp(submit.eccs[l])
	print("\033[2mHard decoding tables built in %d seconds.\033[0m" % (time.time() - start))

# If no node information is specified, then simulate all nodes in serial. Else simulate only the given node.
if (len(sys.argv) > 2):
	sim.LocalSimulations(submit, stream, int(sys.argv[2]))
	stream.close()
else:
	for i in range(submit.nodes):
		sim.LocalSimulations(submit, stream, i)
	stream.close()