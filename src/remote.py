import os
import sys
from define import submission as sub
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
# If the current node is not specified, then simulate all nodes in serial.
if ((submit.host == "local") or (submit.current == "~~node~~")):
	for i in range(submit.nodes):
		submit.current = ("%d" % i)
		sim.LocalSimulations(submit, stream)
	# Create a folder with the timestamp as its name and move the channels, metrics data and the input files, bqsubmit.dat data into the timestamp-folder.
	stream.close()
	sim.OrganizeResults(submit)
else:
	sim.LocalSimulations(submit, stream)
	stream.close()