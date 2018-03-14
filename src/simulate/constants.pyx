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

cdef int FreeConstants(constants_t *consts):
	# free space allocated to store constants
	# there is currently nothing to free
	return 0

