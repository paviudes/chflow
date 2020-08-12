import multiprocessing
import subprocess
import time

import multiprocessing as mp


def rerun(ts):
    # renaming the subprocess call is silly - remove the rename
    com = subprocess.call("./chflow.sh %s" % (ts), shell="True")
    return com


timestamps = []
with open("partial_cluster.txt") as pf:
    for line in pf.readlines()[:2]:
        timestamps.append(line.strip("\n").strip(" "))
pool = mp.Pool(processes=len(timestamps))
output = pool.map(rerun, timestamps)
