import os
import sys
import time
from define import submission as sub
from define import qcode as qec
from simulate import simulate as sim

# Load a submission record
submit = sub.Submission()
timestamp = sys.argv[1].strip("\n").strip(" ")
# sub.ChangeTimestamp(submit, timestamp)
sub.LoadSub(submit, timestamp, 0)

# Prepare syndrome look-up table for hard decoding.
if np.any(submit.decoders == 1):
    for l in range(submit.levels):
        if submit.decoders[l] == 1:
            if submit.ecc[l].lookup is None:
                print(
                    "\033[2mPreparing syndrome lookup table for the %s code.\033[0m"
                    % (submit.eccs[l].name)
                )
                qec.PrepareSyndromeLookUp(submit.eccs[l])

# If no node information is specified, then simulate all nodes in serial. Else simulate only the given node.
if len(sys.argv) > 2:
    sim.LocalSimulations(submit, int(sys.argv[2]))
else:
    for i in range(submit.nodes):
        sim.LocalSimulations(submit, i)
