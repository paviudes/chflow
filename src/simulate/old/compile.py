import os
import numpy as np
from distutils.core import setup
from distutils.extension import Extension

os.environ["CC"]="cc"
# os.environ["CFLAGS"] = "-lm -O3 -Wall -ffast-math -march=native -mfpmath=sse -fno-signed-zeros"

try:
	import cython
	USE_CYTHON = 1
	ext = ".pyx"
except:
	USE_CYTHON = 0
	ext = ".c"

extensions = [
	Extension('simulate.benchmark', ["simulate/benchmark" + ext], include_dirs = [np.get_include()], language = 'c'),
	Extension('simulate.printfuns', ["simulate/printfuns" + ext], language = 'c'),
	Extension('simulate.checks', ["simulate/checks" + ext], include_dirs = [np.get_include()], language = 'c'),
	Extension('simulate.memory', ["simulate/memory" + ext], language = 'c'),
	Extension('simulate.logmetrics', ["simulate/logmetrics" + ext], include_dirs = [np.get_include()], language = 'c'),
	Extension('simulate.effective', ["simulate/effective" + ext], language = 'c'),
	Extension('simulate.sampling', ["simulate/sampling" + ext], language = 'c'),
	Extension('simulate.qecc', ["simulate/qecc" + ext], language = 'c'),
	Extension('simulate.constants', ["simulate/constants" + ext], language = 'c')
]

if USE_CYTHON:
	from Cython.Build import cythonize
	from Cython.Distutils import build_ext
	extensions = cythonize(extensions)

setup(
	name = 'compiled',
	ext_modules = extensions,
	cmdclass = {'build_ext': build_ext}
)

# ext_modules = [
# 	Extension('benchmark', ['benchmark.pyx'], build_dir="build", include_dirs = [np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
# 	Extension('printfuns', ['printfuns.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
# 	Extension('checks', ['checks.pyx'], build_dir="build", include_dirs = [np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
# 	Extension('memory', ['memory.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
# 	Extension('logmetrics', ['logmetrics.pyx'], build_dir="build", include_dirs = [np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
# 	Extension('effective', ['effective.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
# 	Extension('sampling', ['sampling.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
# 	Extension('qecc', ['qecc.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
# 	Extension('constants', ['constants.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c')
# ]

# setup(
#     name='Flow',
#     ext_modules=cythonize(ext_modules),
#     cmdclass = {'build_ext': build_ext}
# )
