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
	Extension('simulate.benchmark', ['benchmark.pyx'], build_dir="build", include_dirs = [np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('simulate.printfuns', ['printfuns.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('simulate.checks', ['checks.pyx'], build_dir="build", include_dirs = [np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('simulate.memory', ['memory.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('simulate.logmetrics', ['logmetrics.pyx'], build_dir="build", include_dirs = [np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('simulate.effective', ['effective.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('simulate.sampling', ['sampling.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('simulate.qecc', ['qecc.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('simulate.constants', ['constants.pyx'], build_dir="build", extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c')
]

setup(
    name='Flow',
    ext_modules=cythonize(ext_modules),
    cmdclass = {'build_ext': build_ext}
)
