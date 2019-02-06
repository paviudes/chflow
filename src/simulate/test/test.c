/*
#include <stdlib.h>
#include <stdio.h>
// To compile: gcc -shared -o test.so -fPIC test.c

int *Offset(int *arr, int size, int offset){
	// Add an offset to each element of the array
	int i;
	printf("Size of the array is %d and the offset is %d.\n", size, offset);
	int *shifted = malloc(sizeof(int) * size);
	for (i = 0; i < size; i ++)
		shifted[i] = arr[i] + offset;
	return shifted;
}
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
// To compile: gcc -shared -o test.so -fPIC test.c

struct Output{
	int *inparr;
	int *outarr;
	double rtime;
};

struct Output Offset(int nrows, int ncols, int offset, int *arr){
	// Add an offset to each element of the array
	printf("Shape of the input array is %d x %d and the offset is %d.\n", nrows, ncols, offset);
	struct Output outshf;
	clock_t begin = clock();
	outshf.outarr = malloc(sizeof(int) * (nrows * ncols));
	outshf.inparr = malloc(sizeof(int) * (nrows * ncols));
	int i;
	for (i = 0; i < nrows * ncols; i ++){
		outshf.inparr[i] = arr[i];
		outshf.outarr[i] = arr[i] + offset;
	}
	clock_t end = clock();
	outshf.rtime = (double) (end - begin)/CLOCKS_PER_SEC;
	return outshf;
}