#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "constants.h"
#include "printfuns.h"
#include "memory.h"
#include "qecc.h"
#include "checks.h" // only for testing purposes.

void InitQECC(struct qecc_t *qecc){
	// Initialize the elements of a quantum error correcting code.
	qecc->nlogs = (int) pow(4, (double) (qecc->K));
	qecc->nstabs = (int) pow(2, (double) (qecc->N - qecc->K));

	// printf("Function: InitQECC: nlogs = %d, nstabs = %d.\n", qecc->nlogs, qecc->nstabs);

	int s;
	qecc->projector = malloc(qecc->nstabs * sizeof(int *));
	for (s = 0; s < qecc->nstabs; s ++)
		qecc->projector[s] = malloc(qecc->nstabs * sizeof(int));

	int i;
	qecc->action = malloc(qecc->nlogs * sizeof(int **));
	for (i = 0; i < qecc->nlogs; i ++){
		(qecc->action)[i] = malloc(qecc->nstabs * sizeof(int *));
		for (s = 0; s < qecc->nstabs; s ++)
			(qecc->action)[i][s] = malloc(qecc->N * sizeof(int));
	}

	qecc->phases = malloc(qecc->nlogs * sizeof(double complex *));
	for (i = 0; i < qecc->nlogs; i ++)
		(qecc->phases)[i] = malloc(qecc->nstabs * sizeof(double complex));

	// printf("Done InitQECC.\n");
}

void FreeQECC(struct qecc_t *qecc){
	// Free the memory assigned to a quantum error correcting code.
	int s;
	for (s = 0; s < qecc->nstabs; s ++)
		free((qecc->projector)[s]);
	free(qecc->projector);

	int i;
	for (i = 0; i < qecc->nlogs; i ++){
		for (s = 0; s < qecc->nstabs; s ++)
			free((qecc->action)[i][s]);
		free((qecc->action)[i]);
	}
	free(qecc->action);

	for (i = 0; i < qecc->nlogs; i ++)
		free((qecc->phases)[i]);
	free(qecc->phases);
}

void ChoiToProcess(double **process, double complex **choi, double complex ***pauli){
	// Convert from the Choi matrix to the process matrix, of a quantum channel.
	// CHI[a,b] = Trace( Choi * (Pb \otimes Pa^T) ).
	int i, j, k, l;
	double complex contribution;
	for (i = 0; i < 4; i ++){
		for (j = 0; j < 4; j ++){
			contribution = 0;
			for (k = 0; k < 4; k ++){
				for (l = 0; l < (k + 1); l ++)
					contribution = contribution + choi[k][l] * pauli[j][l/2][k/2] * pauli[i][k%2][l%2];
				for (l = k + 1; l < 4; l ++)
					contribution = contribution + conj(choi[l][k]) * pauli[j][l/2][k/2] * pauli[i][k%2][l%2];
			}
			process[i][j] = creal(contribution);
		}
	}
}

void GetFullProcessMatrix(struct qecc_t *qecc, struct simul_t *sim, int isPauli){
	// For each pair of logical operators, we only need the entries of the Chi matrix that correspond to Pauli operators from different logical classes.
	// We construct the sections of the Chi matrices that correspond to rows and columns in the logical classes.
	// printf("Function: GetFullProcessMatrix for isPauli = %d.\n", isPauli);
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
	// PrintDoubleArray2D((sim->process)[0][0], "process[0][0]", qecc->nstabs, qecc->nstabs);
	// printf("done computing full process matrix.\n");
}

void ComputeSyndromeDistribution(struct qecc_t *qecc, struct simul_t *sim, int isPauli){
	// Compute the probability of all the syndromes in the qecc code, for the given error channel and input state. Sample a syndrome from the resulting probability distribution of the syndromes.
	// Probability of a syndrome s, denoted by P(s) is given by the following expression.
	// P(s) = 1/2^(n-k) * sum_(i,j: P_i and P_j are stabilizers) CHI[i,j] * (-1)^sign(P_j).
	const double atol = 10E-15;
	int i, j, s;
	for (s = 0; s < qecc->nstabs; s ++){
		(sim->syndprobs)[s] = 0;
		if (isPauli == 0)
			for (i = 0; i < qecc->nstabs; i ++)
				for (j = 0; j < qecc->nstabs; j ++)
					(sim->syndprobs)[s] += (sim->process)[0][0][i][j] * (qecc->projector)[s][j];
		else
			for (i = 0; i < qecc->nstabs; i ++)
				(sim->syndprobs)[s] = (sim->syndprobs)[s] + (sim->process)[0][0][i][i] * (qecc->projector)[s][i];
		if ((sim->syndprobs)[s] < atol)
			(sim->syndprobs)[s] = 0;
		(sim->syndprobs)[s] = (sim->syndprobs)[s]/((double)(qecc->nstabs));
		if (s == 0)
			(sim->cumulative)[s] = (sim->syndprobs)[s];
		else
			(sim->cumulative)[s] = (sim->cumulative)[s - 1] + (sim->syndprobs)[s];
	}
	// PrintDoubleArray1D((sim->syndprobs), "Syndrome distribution", qecc->nstabs);
}

void MLDecoder(struct qecc_t *qecc, struct simul_t *sim, struct constants_t *consts, int currentframe, int isPauli){
	// Perform maximum likelihood decoding.
	// Compute the probabilities of the logical classes, considitioned on a particular syndrome.
	// The ML Decoder picks the logical error which belongs to the class that has the maximum probability.
	// The probability of a logical class is P(L|s) = Tr( L r L . Ts PI_s E(r) PI_s Ts )/P(s) which can be simplified to
	// P(L|s) = 1/P(s) * sum_(u: Paulis) sum_(i: P_i is in the [u] logical class) sum_(j: Pj is in the [L u L] logical class) CHI[i,j] * (-1)^(P_j).
	// inputs: nqecc, kqecc, chi, algebra (conjugations).
	// printf("Function: MLDecoder\n");
	const double atol = 10E-10;
	int i, j, u, l, s;
	double prob, maxprob, contrib;
	for (s = 0; s < qecc->nstabs; s ++){
		if ((sim->syndprobs)[s] > atol){
			(sim->corrections)[s] = 0;
			maxprob = 0;
			for (l = 0; l < currentframe; l ++){
				prob = 0;
				if (isPauli == 0){
					for (u = 0; u < qecc->nlogs; u ++){
						contrib = 0;
						for (i = 0; i < qecc->nstabs; i ++)
							for (j = 0; j < qecc->nstabs; j ++)
								contrib += (sim->process)[u][(consts->algebra)[0][l][u]][i][j] * (qecc->projector)[s][j];
						prob += (consts->algebra)[1][l][u] * contrib;
					}
				}
				else{
					for (u = 0 ; u < qecc->nlogs; u ++){
						contrib = 0;
						for (i = 0; i < qecc->nstabs; i ++)
							contrib = contrib + (sim->process)[u][u][i][i] * (qecc->projector)[s][i];
						prob = prob + (consts->algebra)[1][l][u] * contrib;
					}
				}
				if (prob > maxprob){
					sim->corrections[s] = l;
					maxprob = prob;
				}
			}
		}
	}
	// PrintIntArray1D(sim->corrections, "sim->corrections", qecc->nstabs);
}

void ComputeEffectiveStates(struct qecc_t *qecc, struct simul_t *sim, struct constants_t *consts, int isPauli){
	// Compute the Choi matrix of the effective logical channel, after applying noise + error correction.
	// The effective Choi matrix (state) is given by: L Ts PI_s E(r) PI_s T_s L which can be written in its unencoded form by expanding in the Pauli basis.
	// un-encoded state = sum_(a,b) Tr( L Ts PI_s E(r) PI_s T_s L . (PI_0 Pa PI_0 \o Pb)).
	// CHOI[a,b] = sum_(i: Pi is in the logical class of Pb) sum_(j: Pj is in the logical class of [L Pa L]) CHI[i,j] * (-1)^(P_j) * (-1)**(if {L, Pb} == 0).
	// printf("Function: ComputeEffectiveStates with isPauli = %d, nlogs = %d, nstabs = %d.\n", isPauli, qecc->nlogs, qecc->nstabs);
	// PrintIntArray2D((qecc->projector), "projector", qecc->nstabs, qecc->nstabs);
	const double atol = 10E-10;
	int r, c, s, a, b, i, j;
	for (s = 0; s < qecc->nstabs; s ++){
		// Initializing all elements to zero
		for (r = 0; r < qecc->nlogs; r ++)
			for (c = 0; c < qecc->nlogs; c ++)
				(sim->effective)[s][r][c] = 0;

		if ((sim->syndprobs)[s] > atol){
			if (isPauli == 0)
				for (r = 0; r < qecc->nlogs; r ++)
					for (c = 0; c < qecc->nlogs; c ++)
						for (a = 0; a < qecc->nlogs; a ++)
							for (b = 0; b < qecc->nlogs; b ++)
								for (i = 0; i < qecc->nstabs; i ++)
									for (j = 0; j < qecc->nstabs; j ++)
										(sim->effective)[s][r][c] += pow(-1, ((int)(b == 2))) * (consts->algebra)[1][(sim->corrections)[s]][a] * (sim->process)[b][(consts->algebra)[0][(sim->corrections)[s]][a]][i][j] * (qecc->projector)[s][j] * (consts->pauli)[a][r/2][c/2] * (consts->pauli)[b][r%2][c%2];
			else
				for (r = 0; r < qecc->nlogs; r ++)
					for (c = 0; c < qecc->nlogs; c ++)
						for (a = 0; a < qecc->nlogs; a ++)
							for (b = 0; b < qecc->nlogs; b ++)
								if (b == (consts->algebra)[0][(sim->corrections)[s]][a])
									for (i = 0; i < qecc->nstabs; i ++)
										(sim->effective)[s][r][c] += pow(-1, ((int)(b == 2))) * (consts->algebra)[1][(sim->corrections)[s]][a] * (sim->process)[b][b][i][i] * (qecc->projector)[s][i] * (consts->pauli)[a][r/2][c/2] * (consts->pauli)[b][r%2][c%2];
			// Normalization
			for (r = 0; r < qecc->nlogs; r ++)
				for (c = 0; c < qecc->nlogs; c ++)
					(sim->effective)[s][r][c] /= (4 * (double)((sim->syndprobs)[s] * qecc->nstabs));
			// printf("s = %d, P(s) = %g\n", s, (sim->syndprobs)[s]);
			// PrintComplexArray2D((sim->effective)[s], "effective channel", qecc->nlogs, qecc->nlogs);
			// if (IsState((sim->effective)[s]) == 1)
			// 	printf("_/ is a quantum state.\n");
			// else
			// 	printf("X not a quantum state.\n");
		}
	}
}

void SingleShotErrorCorrection(int isPauli, int frame, struct qecc_t *qecc, struct simul_t *sim, struct constants_t *consts){
	// Compute the effective logical channel, when error correction is applied over a set of input physical channels.
	// printf("Constructing the full process matrix\n");
	GetFullProcessMatrix(qecc, sim, isPauli);
	// Compute the probabilities of all the syndromes.
	ComputeSyndromeDistribution(qecc, sim, isPauli);
	// Maximum Likelihood Decoding (MLD) -- For every syndrome, compute the probabilities of the logical classes and pick the one that is most likely.
	// printf("Maximum likelihood decoding\n");
	MLDecoder(qecc, sim, consts, frame, isPauli);
	// For every syndrome, apply the correction and compute the new effective choi matrix of the single qubit logical channel.
	// printf("Computing the effective logical channels.\n");
	ComputeEffectiveStates(qecc, sim, consts, isPauli);
	// Convert the effective channels from choi representation to the process matrix form.
	// printf("Converting effective channels to PTM.\n");
	int s;
	for (s = 0; s < qecc->nstabs; s ++){
		ChoiToProcess((sim->effprocess)[s], (sim->effective)[s], consts->pauli);
		// printf("s = %d\n", s);
		// PrintDoubleArray2D((sim->effprocess)[s], "process", qecc->nlogs, qecc->nlogs);
	}
	// printf("Single shot error correction completed.\n");
}