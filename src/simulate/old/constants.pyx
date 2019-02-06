#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False

cdef int InitConstants(constants_t *consts):
	# initialize and fill values into the constants used in the simulation
	################### Defining Pauli matrices ###################
	cdef int p, r, c
	for p in range(4):
		for r in range(2):
			for c in range(2):
				consts[0].pauli[p][r][c] = 0 + 0 * 1j
	# Identity matrix
	consts[0].pauli[0][0][0] = 1 + 1j*0
	consts[0].pauli[0][1][1] = 1 + 1j*0
	# Pauli X
	consts[0].pauli[1][0][1] = 1 + 1j*0
	consts[0].pauli[1][1][0] = 1 + 1j*0
	# Pauli Y
	consts[0].pauli[2][0][1] = 0 - 1j
	consts[0].pauli[2][1][0] = 0 + 1j
	# Pauli Z
	consts[0].pauli[3][0][0] = 1 + 1j*0
	consts[0].pauli[3][1][1] = -1 + 1j*0
	###############################################################
	consts[0].nclifford = 24
	consts[0].atol = 10E-50
	return 0

####################
# Converted to C.

void InitConstants(constants_t *consts){
	// Initialize and fill values into the constants used in the simulation.
	// ################### Defining Pauli matrices ###################
	int p, r, c;
	for (p = 0; p < 4; p ++){
		for (r = 0; r < 2; r ++){
			for (c = 0; c < 2; c ++){
				(consts->pauli)[p][r][c] = 0 + 0 * I;
			}
		}
	}
	// Identity matrix.
	(consts->pauli)[0][0][0] = 1 + I * 0;
	(consts->pauli)[0][1][1] = 1 + I * 0;
	// Pauli X.
	(consts->pauli)[1][0][1] = 1 + I * 0;
	(consts->pauli)[1][1][0] = 1 + I * 0;
	// Pauli Y.
	(consts->pauli)[2][0][1] = 0 - I;
	(consts->pauli)[2][1][0] = 0 + I;
	// Pauli Z.
	(consts->pauli)[3][0][0] = 1 + I * 0;
	(consts->pauli)[3][1][1] = -1 + I * 0;
	// ###############################################################
	consts->nclifford = 24;
	consts->atol = 10E-50;
	// ###############################################################
	// Read the basis change matrix form the file choi_to_chi.txt.
	// The matrix to be read is complex. Every column of the complex matrix is stored as a pair of successive columns in the text file, representing the real and imaginary parts.
	complex128_t **choichi = malloc(sizeof(complex128_t *) * 16);
	double real, imag;
	FILE *basisfp = open("choi_to_chi.txt", "r");
	for (r = 0; r < 16; r ++){
		(consts->choichi)[i] = malloc(sizeof(complex128_t) * 16);
		for (c = 0; c < 32; c ++){
			fscanf(basisfp, "%Lf %Lf", &real, &imag);
		}
		(consts->choichi)[r][c] = real + I * imag;
	}
}
####################

cdef int FreeConstants(constants_t *consts):
	# free space allocated to store constants
	# there is currently nothing to free
	return 0

###################
# Converted to C.
void FreeConstants(constants_t *consts){
	# Free memeory allocated to the various elements of consts.
	int i;
	for (r = 0; r < 16; r ++){
		free((consts->choichi)[r]);
	}
	free(consts->choichi);
}

###################