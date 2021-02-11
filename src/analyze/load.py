# Critical packages
import numpy as np

# Functions from other modules
from define import qchans as qc
from define import fnames as fn
from define import metrics as ml
from define import globalvars as gv


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
        settings["marker"] = gv.Markers[int(pmet)]
        settings["color"] = gv.Colors[int(pmet)]
        
        if (dbs.channel == "bpauli"):
            # For the biased Pauli channel, the X-axis should be eta = rX/rZ
            settings["xlabel"] = "$\\eta = r_{Z}/r_{X}$"
        
            if (dbs.scales[int(pmet)] == 1):
                settings["xaxis"] = dbs.available[:, 1] / dbs.available[:, 0]
            else:
                settings["xaxis"] = np.power(dbs.scales[1], dbs.available[:, 1]) / np.power(dbs.scales[0], dbs.available[:, 0])
        
        else:
            settings["xlabel"] = qc.Channels[dbs.channel]["latex"][int(pmet)]
            settings["xaxis"] = dbs.available[:, int(pmet)]
        
            if not (dbs.scales[int(pmet)] == 1):
                settings["xaxis"] = np.power(dbs.scales[int(pmet)], settings["xaxis"])
    
    settings["linestyle"] = ["None", "--"][dbs.samps == 1]
    
    return None