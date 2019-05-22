#ifndef arrayfuns_H
#define arrayfuns_H

typedef double complex complex128_t;

// Populate a 4D array of a given shape
// The array entries are numbers between 1 and 100.
extern void Initialize4D(complex128_t ****arr, const int shape[4]);

// Populate a 2D array of a given shape
// The array entries are numbers between 1 and 100.
extern void Initialize2D(complex128_t **arr, const int shape[4]);

// Free memory allocated to a 4D array
extern void Free4D(complex128_t ****arr, const int shape[4]);

// Free memory allocated to a 2D array
extern void Free2D(complex128_t **arr, const int shape[4]);

// Print the contents of a 4D array
extern void Print4D(complex128_t ****arr, const int shape[4]);

// Print the contents of a 2D array
extern void Print2D(complex128_t **arr, const int shape[4]);

#endif /* ARRAYFUNS_H */