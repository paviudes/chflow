import numpy as np
from define.QECCLfid.utils import get_interactions, SamplePoisson
from define.QECCLfid.cptps import GenerateSupport

if __name__ == '__main__':
	nmaps = 15
	nqubits = 7
	cutoff = 2
	mean = 2
	interaction_range = get_interactions(nmaps, mean, cutoff)
	supports = GenerateSupport(nqubits, interaction_range)
	print("interaction range: {}\nSupport = {}".format(interaction_range, supports))