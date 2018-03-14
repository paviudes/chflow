import os
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
# os.environ["CC"]="clang"
os.environ["CC"]="gcc"
os.environ["CFLAGS"] = "-lm -O3 -Wall -ffast-math -march=native -mfpmath=sse -fno-signed-zeros"

ext_modules = [
	Extension('benchmark', ['benchmark.pyx'], build_dir="build", include_dirs = [np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('printfuns', ['printfuns.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('checks', ['checks.pyx'], build_dir="build", include_dirs = [np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('memory', ['memory.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('logmetrics', ['logmetrics.pyx'], build_dir="build", include_dirs = [np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('effective', ['effective.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('sampling', ['sampling.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('qecc', ['qecc.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('constants', ['constants.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c')
]

setup(
    name='Flow',
    ext_modules=cythonize(ext_modules),
    cmdclass = {'build_ext': build_ext}
)
