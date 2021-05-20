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
    atol = 1E-16
    if pmet in ml.Metrics:
        if pmet == "uncorr":
            phyerrs = np.load(fn.PhysicalErrorRates(dbs, pmet))[:, level]
        else:
            phyerrs = np.load(fn.PhysicalErrorRates(dbs, pmet))
        if settings is not None:
            settings["xlabel"] = ml.Metrics[pmet]["latex"]
            if "marker" in settings:
                if settings["marker"] == "":
                    settings["marker"] = ml.Metrics[pmet]["marker"]
            if "color" in settings:
                if settings["color"] == "":
                    settings["color"] = ml.Metrics[pmet]["color"]
    else:
        phyerrs = dbs.available[:, int(pmet)]
        if settings is not None:
            settings["marker"] = gv.Markers[int(pmet)]
            settings["color"] = gv.Colors[int(pmet)]
            settings["xlabel"] = qc.Channels[dbs.channel]["latex"][int(pmet)]
            
            if not (dbs.scales[int(pmet)] == 1):
                phyerrs = np.power(dbs.scales[int(pmet)], phyerrs)
    
    # Set negligible values to atol
    negligible = (phyerrs <= atol)
    phyerrs[negligible] = atol

    if settings is not None:
        settings["linestyle"] = ["None", "--"][dbs.samps == 1]
        settings["xaxis"] = phyerrs
    
    return phyerrs