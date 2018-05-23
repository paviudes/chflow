# Define the complex128_t data structure first.
ctypedef long double complex complex128_t


cdef struct simulation:
	## Quantum error correction
	char *chname
	int decoder
	long double **physical
	long double ***virtual
	long double ***logical
	long double ****process
	long double *syndprobs
	long double *cumulative
	long double *levelOneSynds
	long double *levelOneCumul
	int *corrections
	complex128_t ***effective
	long double ***effprocess
	long double ***levelOneChannels
	int *frames
	## Metrics
	int nmetrics
	char **metricsToCompute
	long double **metricValues
	## Syndrome sampling
	int nstats
	int nlevels
	int maxbin
	int importance
	long double *levelOneImpDist
	long double *levelOneImpCumul
	long double *outlierprobs
	long double **sampling
	long *statsperlevel
	int nbins
	int ****bins
	long double **sumsq
	long double **variance
	long double threshold
	## Running
	int cores
ctypedef simulation simul_t

# Allocate memory to the elements of the simulation structure that do not depend on the QECC.
cdef int AllocSimParams(simul_t *simul, int nstabs, int nlogs) nogil
# Free memory allocated to the elements of the simulation structure that do not depend on the QECC.
cdef int FreeSimParams(simul_t *simul, int nstabs, int nlogs) nogil

# Allocate memory to the elements of the simulation structure that depend upon the QECC.
cdef int AllocSimParamsQECC(simul_t *simul, int nqecc, int nstabs, int nlogs) nogil
# Free memory allocated to the elements of the simulation structure that depend on the QECC.
cdef int FreeSimParamsQECC(simul_t *simul, int nqecc, int nstabs, int nlogs) nogil

# Copy the elements of a simulation structure from an old to a new one.
from qecc cimport qecc_t
cdef int CopySimulation(simul_t *copyto, simul_t *copyfrom, qecc_t *qecc)

# Merge two simulation structures, including the results from the second simulation into the first. Add the average failure rates, bins, average logical channels, variance, statistics per level.
cdef int MergeSimulations(simul_t *parent, simul_t *child, int nlogs)

# Determine the number of independent logical channels at every level, that determine the logical channels of higher levels.
cdef int CountIndepLogicalChannels(int *chans, qecc_t **qecc, int nlevels) nogil

# Allocate and Free memory for the tree of lower-level channels which determine a logical channel.
cdef int MemManageChannels(long double ******channels, int nbatches, qecc_t **qecc, int nlevels, int importance, int tofree)

# Allocate and free memory for the input channels structure in ComputeLogicalChannels(...)
cdef int MemManageInputChannels(long double ****inputchannels, int nqecc, int nlogs, int importance, int tofree) nogil