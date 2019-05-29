/* Code that cannot be vectorized since the (icc) compiler detects possible dependencies between array indices */
void GetFullProcessMatrix(struct qecc_t *qecc, struct simul_t *sim, int isPauli){
	// For each pair of logical operators, we only need the entries of the Chi matrix that correspond to Pauli operators from different logical classes.
	// We construct the sections of the Chi matrices that correspond to rows and columns in the logical classes.
	int i, j, k, l, q;
	double contribution = 1;
	if (isPauli == 0){
		for (i = 0; i < qecc->nlogs; i ++){
			for (j = 0; j < qecc->nlogs; j ++){
				for (k = 0; k < qecc->nstabs; k ++){
					for (l = 0; l < qecc->nstabs; l ++){
						contribution = 1;
						for (q = 0; q < qecc->N; q ++)
							contribution *= (sim->virtchan)[q][(qecc->action)[i][k][q]][(qecc->action)[j][l][q]];
						(sim->process)[i][j][k][l] = creal((qecc->phases)[i][k] * (qecc->phases)[j][l]) * contribution;
					}
				}
			}
		}
	}
	else{
		// For a Pauli channel, the process matrix is diagonal.
		for (i = 0; i < qecc->nlogs; i ++){
			for (j = 0; j < qecc->nstabs; j ++){
				contribution = 1;
				for (q = 0; q < qecc->N; q ++)
					contribution *= (sim->virtchan)[q][(qecc->action)[i][j][q]][(qecc->action)[i][j][q]];
				(sim->process)[i][i][j][j] = creal((qecc->phases)[i][j] * (qecc->phases)[i][j]) * contribution;
			}
		}
	}
}

/* Code that can be succesfully auto-vectorized by the compiler as the nested structure is now *flattended out* into a single large loop. */.
void GetFullProcessMatrix(struct qecc_t *qecc, struct simul_t *sim, int isPauli){
	// For each pair of logical operators, we only need the entries of the Chi matrix that correspond to Pauli operators from different logical classes.
	// We construct the sections of the Chi matrices that correspond to rows and columns in the logical classes.
	// This function should be vectorized for performance.
	int v, i, j, k, l, q;
	double contribution = 1;
	if (isPauli == 0){
		#pragma omp simd
		for (v = 0; v < qecc->nlogs * qecc->nlogs * qecc->nstabs * qecc->nstabs; v ++){
			l = v % qecc->nstabs;
			k = (v/qecc->nstabs) % qecc->nstabs;
			j = (v/(qecc->nstabs * qecc->nstabs)) % qecc->nlogs;
			i = (v/(qecc->nstabs * qecc->nstabs * qecc->nlogs)) % qecc->nlogs;
			contribution = 1;
			for (q = 0; q < qecc->N; q ++)
				contribution *= (sim->virtchan)[q][(qecc->action)[i][k][q]][(qecc->action)[j][l][q]];
			(sim->process)[i][j][k][l] = creal((qecc->phases)[i][k] * (qecc->phases)[j][l]) * contribution;
		}
	}
	else{
		// For a Pauli channel, the process matrix is diagonal.
		for (v = 0; v < qecc->nlogs * qecc->nstabs; v ++){
			j = v % qecc->nstabs;
			i = (v/qecc->nstabs) % qecc->nlogs;
			contribution = 1;
			for (q = 0; q < qecc->N; q ++)
				contribution *= (sim->virtchan)[q][(qecc->action)[i][j][q]][(qecc->action)[i][j][q]];
			(sim->process)[i][i][j][j] = creal((qecc->phases)[i][j] * (qecc->phases)[i][j]) * contribution;
		}
	}
}