#ifndef HYBRID_H
#define HYBRID_H

/*
	We would like to average over some of the channels in the given level.
	This averaging provides a coarse-grained information on the different channels in the level.
	The decbins contains a number for every channel in the level, such that the number for those channels which must be averaged are identical.
	printf("Function: Coarsegrain(%d, sims, channels, %d, %d)\n", level, nchans, nlogs);
*/
extern void Coarsegrain(int level, struct simul_t **sims, double *****channels, int nchans, int nlogs);

#endif /* HYBRID_H  */
