import numpy as np
from define.chanreps import ConvertRepresentations

def GetLeadingKrauss(plvchan):
	# Given the Pauli Liouville representation of a quantum channel, compute its leading Kraus operator
	# The leading Krauss operator is simply the leading eigen vector of the corresponding Choi matrix.
	# First convert the channel to its Choi matrix and then compute the leading eigen vector.
	choi = ConvertRepresentations(plvchan, "process", "choi")	
	(eigvals, eigvecs) = np.linalg.eig(choi)
	leadkraus = np.sqrt(eigvals[np.argmax(eigvals)]) * eigvecs[:, np.argmax(eigvals)]
	return leadkraus
