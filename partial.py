import os
import multiprocessing as mp
from subprocess import Popen

def rerun(ts):
    Popen(['./chflow.sh', ts])
    # os.system("./chflow.sh %s" % (ts))
    # print("calling chflow %s" % (ts))
    return None


processes = []
timestamps = []
with open("partial_cluster.txt") as pf:
    for line in pf.readlines()[:2]:
        timestamps.append(line.strip("\n").strip(" "))

for p in range(len(timestamps)):
    processes.append(mp.Process(target=rerun, args=(timestamps[p],)))

for p in range(len(timestamps)):
    processes[p].start()

for p in range(len(timestamps)):
    processes[p].join()
