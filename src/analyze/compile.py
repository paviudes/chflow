import os
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
os.environ["CC"]="gcc"
os.environ["CFLAGS"] = "-lm -O3 -Wall -ffast-math -march=native -mfpmath=sse -fno-signed-zeros"

ext_modules = [
	Extension('bestfit', ['bestfit.pyx'], build_dir="build", include_dirs = [np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
	Extension('learning', ['learning.pyx'], build_dir="build", include_dirs = [np.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], language = 'c'),
]

setup(
    name='ML',
    ext_modules=cythonize(ext_modules),
    cmdclass = {'build_ext': build_ext}
)
