#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <complex.h>

struct constants_t
{
	double complex ***pauli;
	/*
		Linear transformation from the Choi matrix respresentation to the Chi matrix respresentation for quantum channels.
		Read the basis change matrix form the file choi_to_chi.txt.
		The matrix to be read is complex. Every column of the complex matrix is stored as a pair of successive columns in the text file, representing the real and imaginary parts.
		The following Python code can be used to generate the transformation matrix.
		=======
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
		=======
	*/
	double complex **choitochi;

	/*
		Conjugating Pauli operators with Cliffords.
		List out all the conjugation rules for clifford operators
		The clifford operators do a permutation of the Pauli frame axes, keeping the condition i (X Y Z) = I.
		So we have the following conjugation rules. X -> {+,- X, +,- Y, +,- Z} and for each choice of X, we have Y -> {+,- A, +,- B} where A, B are the Pauli operators that do not include the result of the transformation of X. Finally, Z is fixed by ensuring the condition that i (X Y Z) = I.
		C 	X |	Y |	Z
		--------------
		C1	X 	Y 	Z
		C2	X 	-Y	-Z

		C5	-X	Y 	-Z
		C6	-X 	-Y 	Z

		C3	X	Z 	-Y
		C4	X 	-Z 	Y

		C7	-X 	Z 	Y
		C8	-X 	-Z 	-Y

		C9	Y 	X 	-Z
		C10	Y 	-X 	Z
		C11	Y 	Z 	X
		C12	Y 	-Z 	-X

		C13	-Y 	X 	Z
		C14	-Y 	-X 	-Z
		C15	-Y 	Z 	X
		C16	-Y 	-Z 	-X

		C17	Z 	X 	Y
		C18	Z 	-X 	-Y
		C19	Z 	Y 	-X
		C20	Z 	-Y 	X

		C21	-Z 	X 	-Y
		C22	-Z 	-X 	Y
		C23	-Z 	Y	X
		C24	-Z 	-Y	-X

		For all Cliffords, C I C = I.

		Encoding to follow: {I: [0, 1], "X":[1, 1], "-X":[1, -1], "Y":[2, 1], "-Y":[2, -1], "Z":[3, 1], "-Z":[3, -1]}
	*/
	int ***algebra;

	// Numerical precision.
	double atol;

	// Number of Cliford operators.
	int nclifford;
};

// Initialize and fill values into the constants used in the simulation.
extern void InitConstants(struct constants_t *consts);

// Free memeory allocated to the various elements of consts.
extern void FreeConstants(struct constants_t *consts);

#endif /* CONSTANTS_H */