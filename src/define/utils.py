import os
from define import fnames as fn


def IsNumber(numorstr):
    # test if the input is a number.
    try:
        float(numorstr)
        return 1
    except:
        return 0


def LoadTimeStamp(submit, timestamp):
    # change the timestamp of a submission and all the related values to the timestamp.
    submit.timestamp = timestamp
    # Re define the variables that depend on the time stamp
    submit.outdir = fn.OutputDirectory(os.path.dirname(submit.outdir), submit)
    submit.inputfile = fn.SubmissionInputs(timestamp)
    submit.scheduler = fn.SubmissionSchedule(timestamp)
    # print("New time stamp: {}, output directory: {}, input file: {}, scheduler: {}".format(timestamp, submit.outdir, submit.inputfile, submit.scheduler))
    return None
