# Define the complex data type before this
ctypedef long double complex complex128_t


# This function is a wrapper for ComputeLogicalChannels(...)
# The inputs to this function are class objects, just like a python function.
# In this function, we will prepare the inputs to the pure C functions.
# Then call the C functions.
# Once done, we will write the output from the C structures, back to pure Python objects.
cpdef Benchmark(submit, rate, sample, physical, refchan)