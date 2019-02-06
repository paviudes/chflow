#ifndef QECC_H
#define QECC_H

#include <complex.h>
#include "memory.h"

struct qecc_t{
	int N;
	int K;
	int D;
	int nlogs;
	int nstabs;
	int **projector;
	int ***action;
	double complex **phases;
};

// Allocate memory for the elements of the quantum error correcting code.
extern void InitQECC(struct qecc_t *qecc);

// Allocate memory allocated to the elements of the quantum error correcting code..
extern void FreeQECC(struct qecc_t *qecc);

// Compute the effective logical channel, when error correction is applied over a set of input physical channels.
extern void SingleShotErrorCorrection(int isPauli, int frame, struct qecc_t *qecc, struct simul_t *sim, struct constants_t *consts);

#endif /* QECC_H */