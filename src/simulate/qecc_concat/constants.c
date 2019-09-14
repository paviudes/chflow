#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include "constants.h"

void InitConstants(struct constants_t *consts){
	// Initialize and fill values into the constants used in the simulation.
	consts->atol = 10E-20;
	// Assign Pauli matrices.
	consts->pauli = malloc(sizeof(double complex**) * 4);
	int p, r;
	for (p = 0; p < 4; p ++){
		(consts->pauli)[p] = malloc(sizeof(double complex*) * 2);
		for (r = 0; r < 2; r ++)
			(consts->pauli)[p][r] = malloc(sizeof(double complex) * 2);
	}
	// Identity Matrix.
	(consts->pauli)[0][0][0] = 1; (consts->pauli)[0][0][1] = 0;
	(consts->pauli)[0][1][0] = 0; (consts->pauli)[0][1][1] = 1;
	// Pauli X.
	(consts->pauli)[1][0][0] = 0; (consts->pauli)[1][0][1] = 1;
	(consts->pauli)[1][1][0] = 1; (consts->pauli)[1][1][1] = 0;
	// Pauli Y.
	(consts->pauli)[2][0][0] = 0; (consts->pauli)[2][0][1] = -I;
	(consts->pauli)[2][1][0] = I; (consts->pauli)[2][1][1] = 0;
	// Pauli Z.
	(consts->pauli)[3][0][0] = 1; (consts->pauli)[3][0][1] = 0;
	(consts->pauli)[3][1][0] = 0; (consts->pauli)[3][1][1] = -1;

	/*
		Pauli Clifford conjugation table.
	*/
	consts->nclifford = 24;
	consts->algebra = malloc(sizeof(int **) * 2);
	for (p = 0; p < 2; p ++){
		(consts->algebra)[p] = malloc(sizeof(int *) * consts->nclifford);
		for (r = 0; r < consts->nclifford; r ++)
			(consts->algebra)[p][r] = malloc(sizeof(int) * 4);
	}
	/*
		Use the following Python code to generate the table.
		line1 = "{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 1, 3, 2}, {0, 1, 3, 2}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 2, 3, 1}, {0, 2, 1, 3}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 1, 2}, {0, 3, 2, 1}, {0, 3, 2, 1}, {0, 3, 1, 2}, {0, 3, 1, 2}, {0, 3, 2, 1}, {0, 3, 2, 1}}"
		line2 = "{{1, 1, 1, 1}, {1, 1, -1, -1}, {1, -1, 1, -1}, {1, -1, -1, 1}, {1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {1, -1, -1, -1}, {1, 1, 1, -1}, {1, 1, -1, 1}, {1, 1, 1, 1}, {1, 1, -1, -1}, {1, -1, 1, 1}, {1, -1, -1, -1}, {1, -1, 1, 1}, {1, -1, -1, -1}, {1, 1, 1, 1}, {1, 1, -1, -1}, {1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, -1}, {1, -1, -1, 1}, {1, -1, 1, 1}, {1, -1, -1, -1}}"
		T1 = np.array(list(map(lambda str: int(str.replace("}", "").replace("{","")), line1.split(", "))), dtype=int).reshape(24, 4)
		T2 = np.array(list(map(lambda str: int(str.replace("}", "").replace("{","")), line2.split(", "))), dtype=int).reshape(24, 4)
		for i in range(24):
			print("(consts->algebra)[0][%d][0] = %d; (consts->algebra)[0][%d][1] = %d; (consts->algebra)[0][%d][2] = %d; (consts->algebra)[0][%d][3] = %d;" % (i, T1[i, 0], i, T1[i, 1], i, T1[i, 2], i, T1[i, 3]))
		for i in range(24):
			print("(consts->algebra)[1][%d][0] = %d; (consts->algebra)[1][%d][1] = %d; (consts->algebra)[1][%d][2] = %d; (consts->algebra)[1][%d][3] = %d;" % (i, T2[i, 0], i, T2[i, 1], i, T2[i, 2], i, T2[i, 3]))
	*/
	(consts->algebra)[0][0][0] = 0; (consts->algebra)[0][0][1] = 1; (consts->algebra)[0][0][2] = 2; (consts->algebra)[0][0][3] = 3;
	(consts->algebra)[0][1][0] = 0; (consts->algebra)[0][1][1] = 1; (consts->algebra)[0][1][2] = 2; (consts->algebra)[0][1][3] = 3;
	(consts->algebra)[0][2][0] = 0; (consts->algebra)[0][2][1] = 1; (consts->algebra)[0][2][2] = 2; (consts->algebra)[0][2][3] = 3;
	(consts->algebra)[0][3][0] = 0; (consts->algebra)[0][3][1] = 1; (consts->algebra)[0][3][2] = 2; (consts->algebra)[0][3][3] = 3;
	(consts->algebra)[0][4][0] = 0; (consts->algebra)[0][4][1] = 1; (consts->algebra)[0][4][2] = 3; (consts->algebra)[0][4][3] = 2;
	(consts->algebra)[0][5][0] = 0; (consts->algebra)[0][5][1] = 1; (consts->algebra)[0][5][2] = 3; (consts->algebra)[0][5][3] = 2;
	(consts->algebra)[0][6][0] = 0; (consts->algebra)[0][6][1] = 1; (consts->algebra)[0][6][2] = 3; (consts->algebra)[0][6][3] = 2;
	(consts->algebra)[0][7][0] = 0; (consts->algebra)[0][7][1] = 1; (consts->algebra)[0][7][2] = 3; (consts->algebra)[0][7][3] = 2;
	(consts->algebra)[0][8][0] = 0; (consts->algebra)[0][8][1] = 2; (consts->algebra)[0][8][2] = 1; (consts->algebra)[0][8][3] = 3;
	(consts->algebra)[0][9][0] = 0; (consts->algebra)[0][9][1] = 2; (consts->algebra)[0][9][2] = 1; (consts->algebra)[0][9][3] = 3;
	(consts->algebra)[0][10][0] = 0; (consts->algebra)[0][10][1] = 2; (consts->algebra)[0][10][2] = 3; (consts->algebra)[0][10][3] = 1;
	(consts->algebra)[0][11][0] = 0; (consts->algebra)[0][11][1] = 2; (consts->algebra)[0][11][2] = 3; (consts->algebra)[0][11][3] = 1;
	(consts->algebra)[0][12][0] = 0; (consts->algebra)[0][12][1] = 2; (consts->algebra)[0][12][2] = 1; (consts->algebra)[0][12][3] = 3;
	(consts->algebra)[0][13][0] = 0; (consts->algebra)[0][13][1] = 2; (consts->algebra)[0][13][2] = 1; (consts->algebra)[0][13][3] = 3;
	(consts->algebra)[0][14][0] = 0; (consts->algebra)[0][14][1] = 2; (consts->algebra)[0][14][2] = 3; (consts->algebra)[0][14][3] = 1;
	(consts->algebra)[0][15][0] = 0; (consts->algebra)[0][15][1] = 2; (consts->algebra)[0][15][2] = 3; (consts->algebra)[0][15][3] = 1;
	(consts->algebra)[0][16][0] = 0; (consts->algebra)[0][16][1] = 3; (consts->algebra)[0][16][2] = 1; (consts->algebra)[0][16][3] = 2;
	(consts->algebra)[0][17][0] = 0; (consts->algebra)[0][17][1] = 3; (consts->algebra)[0][17][2] = 1; (consts->algebra)[0][17][3] = 2;
	(consts->algebra)[0][18][0] = 0; (consts->algebra)[0][18][1] = 3; (consts->algebra)[0][18][2] = 2; (consts->algebra)[0][18][3] = 1;
	(consts->algebra)[0][19][0] = 0; (consts->algebra)[0][19][1] = 3; (consts->algebra)[0][19][2] = 2; (consts->algebra)[0][19][3] = 1;
	(consts->algebra)[0][20][0] = 0; (consts->algebra)[0][20][1] = 3; (consts->algebra)[0][20][2] = 1; (consts->algebra)[0][20][3] = 2;
	(consts->algebra)[0][21][0] = 0; (consts->algebra)[0][21][1] = 3; (consts->algebra)[0][21][2] = 1; (consts->algebra)[0][21][3] = 2;
	(consts->algebra)[0][22][0] = 0; (consts->algebra)[0][22][1] = 3; (consts->algebra)[0][22][2] = 2; (consts->algebra)[0][22][3] = 1;
	(consts->algebra)[0][23][0] = 0; (consts->algebra)[0][23][1] = 3; (consts->algebra)[0][23][2] = 2; (consts->algebra)[0][23][3] = 1;

	(consts->algebra)[1][0][0] = 1; (consts->algebra)[1][0][1] = 1; (consts->algebra)[1][0][2] = 1; (consts->algebra)[1][0][3] = 1;
	(consts->algebra)[1][1][0] = 1; (consts->algebra)[1][1][1] = 1; (consts->algebra)[1][1][2] = -1; (consts->algebra)[1][1][3] = -1;
	(consts->algebra)[1][2][0] = 1; (consts->algebra)[1][2][1] = -1; (consts->algebra)[1][2][2] = 1; (consts->algebra)[1][2][3] = -1;
	(consts->algebra)[1][3][0] = 1; (consts->algebra)[1][3][1] = -1; (consts->algebra)[1][3][2] = -1; (consts->algebra)[1][3][3] = 1;
	(consts->algebra)[1][4][0] = 1; (consts->algebra)[1][4][1] = 1; (consts->algebra)[1][4][2] = 1; (consts->algebra)[1][4][3] = -1;
	(consts->algebra)[1][5][0] = 1; (consts->algebra)[1][5][1] = 1; (consts->algebra)[1][5][2] = -1; (consts->algebra)[1][5][3] = 1;
	(consts->algebra)[1][6][0] = 1; (consts->algebra)[1][6][1] = -1; (consts->algebra)[1][6][2] = 1; (consts->algebra)[1][6][3] = 1;
	(consts->algebra)[1][7][0] = 1; (consts->algebra)[1][7][1] = -1; (consts->algebra)[1][7][2] = -1; (consts->algebra)[1][7][3] = -1;
	(consts->algebra)[1][8][0] = 1; (consts->algebra)[1][8][1] = 1; (consts->algebra)[1][8][2] = 1; (consts->algebra)[1][8][3] = -1;
	(consts->algebra)[1][9][0] = 1; (consts->algebra)[1][9][1] = 1; (consts->algebra)[1][9][2] = -1; (consts->algebra)[1][9][3] = 1;
	(consts->algebra)[1][10][0] = 1; (consts->algebra)[1][10][1] = 1; (consts->algebra)[1][10][2] = 1; (consts->algebra)[1][10][3] = 1;
	(consts->algebra)[1][11][0] = 1; (consts->algebra)[1][11][1] = 1; (consts->algebra)[1][11][2] = -1; (consts->algebra)[1][11][3] = -1;
	(consts->algebra)[1][12][0] = 1; (consts->algebra)[1][12][1] = -1; (consts->algebra)[1][12][2] = 1; (consts->algebra)[1][12][3] = 1;
	(consts->algebra)[1][13][0] = 1; (consts->algebra)[1][13][1] = -1; (consts->algebra)[1][13][2] = -1; (consts->algebra)[1][13][3] = -1;
	(consts->algebra)[1][14][0] = 1; (consts->algebra)[1][14][1] = -1; (consts->algebra)[1][14][2] = 1; (consts->algebra)[1][14][3] = 1;
	(consts->algebra)[1][15][0] = 1; (consts->algebra)[1][15][1] = -1; (consts->algebra)[1][15][2] = -1; (consts->algebra)[1][15][3] = -1;
	(consts->algebra)[1][16][0] = 1; (consts->algebra)[1][16][1] = 1; (consts->algebra)[1][16][2] = 1; (consts->algebra)[1][16][3] = 1;
	(consts->algebra)[1][17][0] = 1; (consts->algebra)[1][17][1] = 1; (consts->algebra)[1][17][2] = -1; (consts->algebra)[1][17][3] = -1;
	(consts->algebra)[1][18][0] = 1; (consts->algebra)[1][18][1] = 1; (consts->algebra)[1][18][2] = 1; (consts->algebra)[1][18][3] = -1;
	(consts->algebra)[1][19][0] = 1; (consts->algebra)[1][19][1] = 1; (consts->algebra)[1][19][2] = -1; (consts->algebra)[1][19][3] = 1;
	(consts->algebra)[1][20][0] = 1; (consts->algebra)[1][20][1] = -1; (consts->algebra)[1][20][2] = 1; (consts->algebra)[1][20][3] = -1;
	(consts->algebra)[1][21][0] = 1; (consts->algebra)[1][21][1] = -1; (consts->algebra)[1][21][2] = -1; (consts->algebra)[1][21][3] = 1;
	(consts->algebra)[1][22][0] = 1; (consts->algebra)[1][22][1] = -1; (consts->algebra)[1][22][2] = 1; (consts->algebra)[1][22][3] = 1;
	(consts->algebra)[1][23][0] = 1; (consts->algebra)[1][23][1] = -1; (consts->algebra)[1][23][2] = -1; (consts->algebra)[1][23][3] = -1;

	
	consts->choitochi = malloc(sizeof(double complex *) * 16);
	for (p = 0; p < 16; p ++){
		(consts->choitochi)[p] = malloc(sizeof(double complex) * 16);
		for (r = 0; r < 16; r ++)
			(consts->choitochi)[p][r] = 0;
	}
	(consts->choitochi)[1][1] = 0.25; (consts->choitochi)[1][4] = 0.25; (consts->choitochi)[1][11] = 0.25 * I; 
	(consts->choitochi)[2][2] = 0.25; (consts->choitochi)[2][8] = 0.25; (consts->choitochi)[2][13] = 0.25 * I; 
	(consts->choitochi)[3][3] = 0.25; (consts->choitochi)[3][6] = 0.25 * I; (consts->choitochi)[3][12] = 0.25; 
	(consts->choitochi)[4][1] = 0.25; (consts->choitochi)[4][4] = 0.25; (consts->choitochi)[4][14] = 0.25 * I; 
	(consts->choitochi)[5][0] = 0.25; (consts->choitochi)[5][5] = 0.25; 
	(consts->choitochi)[6][6] = 0.25; (consts->choitochi)[6][9] = 0.25; (consts->choitochi)[6][12] = 0.25 * I; 
	(consts->choitochi)[7][2] = 0.25 * I; (consts->choitochi)[7][7] = 0.25; (consts->choitochi)[7][13] = 0.25; 
	(consts->choitochi)[8][2] = 0.25; (consts->choitochi)[8][7] = 0.25 * I; (consts->choitochi)[8][8] = 0.25; 
	(consts->choitochi)[9][3] = 0.25 * I; (consts->choitochi)[9][6] = 0.25; (consts->choitochi)[9][9] = 0.25; 
	(consts->choitochi)[10][0] = 0.25; (consts->choitochi)[10][10] = 0.25; 
	(consts->choitochi)[11][4] = 0.25 * I; (consts->choitochi)[11][11] = 0.25; (consts->choitochi)[11][14] = 0.25; 
	(consts->choitochi)[12][3] = 0.25; (consts->choitochi)[12][9] = 0.25 * I; (consts->choitochi)[12][12] = 0.25; 
	(consts->choitochi)[13][7] = 0.25; (consts->choitochi)[13][8] = 0.25 * I; (consts->choitochi)[13][13] = 0.25; 
	(consts->choitochi)[14][1] = 0.25 * I; (consts->choitochi)[14][11] = 0.25; (consts->choitochi)[14][14] = 0.25; 
	(consts->choitochi)[15][0] = 0.25; (consts->choitochi)[15][15] = 0.25;
}

void FreeConstants(struct constants_t *consts){
	// Free memeory allocated to the various elements of consts.
	int p, r;
	for (p = 0; p < 4; p ++){
		for (r = 0; r < 2; r ++)
			free((consts->pauli)[p][r]);
		free((consts->pauli)[p]);
	}
	free(consts->pauli);
	for (p = 0; p < 2; p ++){
		for (r = 0; r < consts->nclifford; r ++)
			free((consts->algebra)[p][r]);
		free((consts->algebra)[p]);
	}
	free(consts->algebra);
	for (r = 0; r < 16; r ++){
		free((consts->choitochi)[r]);
	}
	free(consts->choitochi);
}