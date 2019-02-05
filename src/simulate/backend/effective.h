#ifndef EFFECTIVE_H
#define EFFECTIVE_H

extern void Performance(struct qecc_t **qecc, struct simul_t **sims, struct constants_t *consts);

// only for testing
extern void ProcessToChoi(double **process, int nlogs, double complex **choi, double complex ***pauli);

#endif /* EFFECTIVE_H */