'''
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

def WrapperOffset(arr, offset):
	# wrapper for the c function in test.c
	_test = ctypes.CDLL('test.so')
	_test.Offset.argtypes = (ndpointer(ctypes.c_int, flags = "C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int)
	_test.Offset.restype = (ndpointer(dtype = ctypes.c_int, shape = (arr.size,), flags = "C_CONTIGUOUS"))
	print("Calling C function: Offset(arr = {}, size = {}, offset = {})".format(arr, arr.size, offset))
	shifted = _test.Offset(arr, arr.size, offset)
	return shifted

if __name__ == '__main__':
	arr = np.array([2, 3, 4, 5, 6], dtype = np.int32)
	offset = 2
	shifted = WrapperOffset(arr, offset)
	print("Array: {} offset by {} resulted in {}.".format(arr, offset, shifted))
'''

import ctypes
import numpy as np

class Shift:
	"""
	Storing an array and its offset
	"""
	def __init__(self, nrows, ncols, offset):
		self.arr = np.random.randint(0, high=10, size=(nrows, ncols)).astype(np.int32)
		self.offset = offset

def GetOutput(nrows, ncols):
	class Output(ctypes.Structure):
		"""
		Output datatype containing the shifted array.
		"""
		_fields_ = [("inparr", np.ctypeslib.ndpointer(dtype = np.int32, shape=(nrows * ncols,), ndim = 1, flags = "C_CONTIGUOUS")),
					("outarr", np.ctypeslib.ndpointer(dtype = np.int32, shape=(nrows * ncols,), ndim = 1, flags = "C_CONTIGUOUS")),
					("rtime", ctypes.c_double)]
	return Output

def WrapperOffset(inpshf):
	# wrapper for the c function in test.c
	_test = ctypes.CDLL('test.so')
	_test.Offset.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(ctypes.c_int, ndim = 1, flags = "C_CONTIGUOUS"))
	# _test.Offset.restype = (ndpointer(dtype = ctypes.c_int, shape = (inpshf.arr.size,), flags = "C_CONTIGUOUS"))
	_test.Offset.restype = GetOutput(*inpshf.arr.shape)
	# print("Calling C function: Offset(nrows = {}, ncols = {}, offset = {}, arr = {})".format(inpshf.arr.shape[0], inpshf.arr.shape[1], inpshf.offset, inpshf.arr))
	outshf = _test.Offset(inpshf.arr.shape[0], inpshf.arr.shape[1], inpshf.offset, inpshf.arr.ravel())
	return outshf

if __name__ == '__main__':
	inpshf = Shift(3, 3, 12)
	outshf = WrapperOffset(inpshf)
	print("Array:\n{}\noffset by {} resulted in\n{}\nin {} seconds.".format(np.ctypeslib.as_array(outshf.inparr).reshape(*inpshf.arr.shape), inpshf.offset, np.ctypeslib.as_array(outshf.outarr).reshape(*inpshf.arr.shape), outshf.rtime))