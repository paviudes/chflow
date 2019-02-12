#include <stdlib.h>
#include <stdio.h>
// #include "printfuns.h" // only for testing purposes
#include "memory.h"
#include "hybrid.h"

void Coarsegrain(int level, struct simul_t **sims, double *****channels, int nchans, int nlogs){
	// We would like to average over some of the channels in the given level.
	// This averaging provides a coarse-grained information on the different channels in the level.
	// The decbins contains a number for every channel in the level, such that the number for those channels which must be averaged are identical.
	// printf("Function: Coarsegrain(%d, sims, channels, %d, %d)\n", level, nchans, nlogs);
	int b, i, j, s, c;
	int *binsizes;
	double ***avgchans;
	for (s = 0; s < 1 + (int)(sims[0]->importance == 2); s ++){
		// printf("sims[%d]->hybrid = %d, sims[%d]->importance = %d\n", s, sims[s]->hybrid, s, sims[s]->importance);
		if (sims[s]->hybrid > 0){
			// printf("Coarse graining process with %d bins.\n", (sims[s]->ndecbins)[level]);
			// PrintIntArray1D((sims[s]->decbins)[level], "decoder bins", nchans);
			binsizes = malloc(sizeof(int) * (sims[s]->ndecbins)[level]);
			avgchans = malloc(sizeof(double **) * (sims[s]->ndecbins)[level]);
			for (b = 0; b < (sims[s]->ndecbins)[level]; b ++){
				binsizes[b] = 0;
				avgchans[b] = malloc(sizeof(double *) * nlogs);
				for (i = 0; i < nlogs; i ++){
					avgchans[b][i] = malloc(sizeof(double) * nlogs);
					for(j = 0; j < nlogs; j ++)
						avgchans[b][i][j] = 0;
				}
			}
			for (c  = 0; c < nchans; c ++){
				for (i = 0; i < nlogs; i ++)
					for (j = 0; j < nlogs; j ++)
						avgchans[(sims[s]->decbins)[level][c]][i][j] += channels[level][c][s][i][j];
				binsizes[(sims[s]->decbins)[level][c]] ++;
			}
			for (c = 0; c < nchans; c ++)
				for (i = 0; i < nlogs; i ++)
					for (j = 0; j < nlogs; j ++)
						channels[level][c][s][i][j] = avgchans[(sims[s]->decbins)[level][c]][i][j]/((double) binsizes[(sims[s]->decbins)[level][c]]);
			// Free memory
			free(binsizes);
			for (b = 0; b < (sims[0]->ndecbins)[level]; b ++){
				for (i = 0; i < nlogs; i ++)
					free(avgchans[b][i]);
				free(avgchans[b]);
			}
			free(avgchans);
		}
	}
	// printf("Completed coarse-graining.\n");
}
