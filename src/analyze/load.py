# Critical packages
import numpy as np

# Functions from other modules
from define import fnames as fn
from define import metrics as ml


def LoadPhysicalErrorRates(dbs, pmet, settings, level):
    """
    Load the physical error rates.
    """
    if pmet in ml.Metrics:
        settings["xlabel"] = ml.Metrics[pmet]["latex"]
        if pmet == "uncorr":
            settings["xaxis"] = np.load(fn.PhysicalErrorRates(dbs, pmet))[:, level]
        else:
            settings["xaxis"] = np.load(fn.PhysicalErrorRates(dbs, pmet))
        if "marker" in settings:
            if settings["marker"] == "":
                settings["marker"] = ml.Metrics[pmet]["marker"]
        if "color" in settings:
            if settings["color"] == "":
                settings["color"] = ml.Metrics[pmet]["color"]
    else:
        settings["xlabel"] = qc.Channels[dbs.channel]["latex"][np.int(pmet)]
        settings["xaxis"] = dbs.available[:, np.int(pmet)]
        settings["marker"] = gv.Markers[int(pmet)]
        settings["color"] = gv.Colors[int(pmet)]
        if not (dbs.scales[int(pmet)] == 1):
            settings["xaxis"] = np.power(dbs.scales[int(pmet)], phyerrs)

    settings["linestyle"] = ["None", "--"][dbs.samps == 1]
    return None
