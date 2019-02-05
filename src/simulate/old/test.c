long double Infidelity(complex128_t **choi){
	// Compute the Infidelity between the input Choi matrix and the Choi matrix corresponding to the identity state.
	// Choi matrix for the identity is: 0.5 * [[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]].
	// Returns 1-fidelity.
	infidelity = 1 - (1/((long double)2)) * (long double)(creall(choi[0][0] + choi[3][0] + choi[0][3] + choi[3][3]));
	return infidelity;
}

long double ProcessFidelity(complex128_t **choi){
	// Compute the average infidelity, given by: 1/6 * (4 - Tr(N)) where N is the process matrix describing a noise channel.
	// The above expression for infidelity can be simplified to 2/3 * entanglement infidelity.
	return (2/((long double)3) * Infidelity(choi));
}

long double FrobeniousNorm(complex128_t **choi){
	// Compute the Frobenious norm of the difference between the input Choi matrix and the Choi matrix corresponding to the Identity channel.
	// Frobenious of A is defined as: sqrtl(Trace(A^\dagger . A)).
	// https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm .
	int i, j;
	long double frobenious = 0;
	for (i = 0 ; i < 4; i ++){
		for (j = 0 ; j < 4; j ++){
			frobenious = frobenious + cabsl(choi[i][j]) * cabsl(choi[i][j]);
		}
	}
	frobenious = frobenious + 1 - (long double)(creall(choi[0][0] + choi[3][0] + choi[0][3] + choi[3][3]));
	frobenious = sqrtl(fabsl(frobenious));
	return frobenious;
}


void ChoiToChi(complex128_t **choi, complex128_t **chi){
	// Convert from the Choi to the Chi representation.
	// Use the idea in ConvertRepresentations.
	// printf("function: ChoiToChi\n");
	int i, j, a, b;
	for (i = 0; i < 16; i ++){
		for (i = 0; i < 16; i ++){
			chi[j/4, j%4] += choi[i/4, i%4] * (consts->choichi)[i][j];
		}
	}
}

long double NonPauliness(complex128_t **choi){
	// Quantify the behaviour of a quantum channel by its difference from a Pauli channel.
	// Convert the input Choi matrix to it's Chi-representation.
	// Compute the ration between the  sum of offdiagonal entries to the sum of disgonal entries.
	// While computing the sums, consider the absolution values of the entries.
	// printf("function: NonPauliness\n");
	int i, j;
	long double nonpauli = 0.0, atol = 10E-20;
	complex128_t **chi = malloc(sizeof(complex128_t *) * 4);
	for (i = 0 ; i < 4; i ++){
		chi[i] = malloc(sizeof(complex128_t) * 4);
		for (j = 0 ; j < 4; j ++){
			chi[i][j] = 0.0 + 0.0 * I;
		}
	}
	ChoiToChi(choi, chi);
	nonpauli = 0.0;
	for (i = 0 ; i < 4; i ++){
		for (j = 0 ; j < 4; j ++){
			if (i != j){
				if ((long double)cabsl(chi[i][i]) * (long double)cabsl(chi[j][j]) >= atol){
					nonpauli = nonpauli + (long double)cabsl(chi[i][j]) * (long double)cabsl(chi[i][j])/((long double)cabsl(chi[i][i]) * (long double)cabsl(chi[j][j]));
				}
			}
		}
	}
	// Free memory for chi.
	for (i = 0 ; i < 4; i ++){
		free(chi[i]);
	}
	free(chi);
	// printf("non Pauliness = %Lf.\n", nonpauli);
	return nonpauli;
}

void ComputeMetrics(long double *metvals, int nmetrics, char **metricsToCompute, complex128_t **choi, char *chname, constants *consts){
	// Compute all the metrics for a given channel, in the Choi matrix form.
	int m;
	for (m = 0; m < nmetrics; m ++){
		if (strcmp(metricsToCompute[m], "frb") == 0){
			metvals[m] = FrobeniousNorm(choi);
		}
		else if (strcmp(metricsToCompute[m], "infid") == 0){
			metvals[m] = Infidelity(choi);
		}
		else if (strcmp(metricsToCompute[m], "processfidelity") == 0){
			metvals[m] = ProcessFidelity(choi);
		}
		else if (strcmp(metricsToCompute[m], "np1") == 0){
			metvals[m] = NonPauliness(choi, consts);
		}
		else{
			// Metrics that are not optimized in C cannot be evaluated at the Logical levels. For these, the metric value is simply zero.
			metvals[m] = 0;
		}
	}
}