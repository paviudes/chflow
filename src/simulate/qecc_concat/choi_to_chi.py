from sys import stdout
import numpy as np

def Dot(matrices):
	# perform a dot product of matrices in a list, from left to right.	
	if (matrices.shape[0] == 1):
		return matrices[0]
	else:
		return np.dot(matrices[0], Dot(matrices[1:]))

if __name__ == '__main__':
	Pauli = np.array([[[1, 0], [0, 1]],
					  [[0, 1], [1, 0]],
					  [[0, -1j], [1j, 0]],
					  [[1, 0], [0, -1]]], dtype = np.complex128)

	# Convert from the Choi matrix to the Chi matrix
	# Tr(J(E) . (Pa o Pb^T)) = 1/2 \sum_(ij) X_(ij) W_((ij)(ab))
	# where W_(ijab) = 1/2 * Tr(Pi Pb Pj Pa).
	# Let v_(ab) = Tr(J(E) . (Pa o Pb^T)), then we have
	# v_(ab) = X_(ij) W_((ij)(ab)). which is the relation: <v| = <x|W.
	# We can rewrite this as: <x| = <v|W^{-1}.
	basis = np.zeros((4, 4, 4, 4), dtype = np.complex128)
	for i in range(4):
		for j in range(4):
			for a in range(4):
				for b in range(4):
					basis[i, j, a, b] = 0.5 * np.trace(Dot(Pauli[[i, b, j, a], :, :]))
	transformation = np.linalg.inv(np.reshape(basis, [16, 16]))
	for i in range(16):
		for j in range(16):
			if (np.imag(transformation[i, j]) < 10E-20):
				if (np.real(transformation[i, j]) > 10E-20):
					stdout.write("(consts->choitochi)[%d][%d] = %g; " % (i, j, np.real(transformation[i, j])))
			else:
				if (np.real(transformation[i, j]) > 10E-20):
					stdout.write("(consts->choitochi)[%d][%d] = %g + %g * I; " % (i, j, np.real(transformation[i, j]), np.imag(transformation[i, j])))
				else:
					stdout.write("(consts->choitochi)[%d][%d] = %g * I; " % (i, j, np.imag(transformation[i, j])))
		print("")