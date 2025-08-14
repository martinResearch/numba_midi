#!/usr/bin/env python3
"""
Setup script for building Cython extensions.
This handles the compilation of .pyx files to C extensions with proper NumPy includes.
"""

import os

from Cython.Build import cythonize
import numpy as np
from setuptools import setup
from setuptools.extension import Extension

# Define the extensions with proper NumPy include directories
extensions = [
    Extension(
        "numba_midi.cython.midi",
        sources=["src/numba_midi/cython/midi.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-ffast-math"] if os.name != "nt" else ["/O2"],
        language="c",
    ),
    Extension(
        "numba_midi.cython.engine2d",
        sources=["src/numba_midi/cython/engine2d.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-ffast-math"] if os.name != "nt" else ["/O2"],
        language="c",
    ),
    Extension(
        "numba_midi.cython.pianoroll",
        sources=["src/numba_midi/cython/pianoroll.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-ffast-math"] if os.name != "nt" else ["/O2"],
        language="c",
    ),
    Extension(
        "numba_midi.cython.draw",
        sources=["src/numba_midi/cython/draw.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-ffast-math"] if os.name != "nt" else ["/O2"],
        language="c",
    ),
    Extension(
        "numba_midi.cython.score",
        sources=["src/numba_midi/cython/score.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-ffast-math"] if os.name != "nt" else ["/O2"],
        language="c",
    ),
]

# Cythonize the extensions
ext_modules = cythonize(
    extensions,
    compiler_directives={
        "language_level": 3,
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "nonecheck": False,
        "embedsignature": True,
        "optimize.use_switch": True,
        "optimize.unpack_method_calls": True,
    },
    annotate=True,  # Enable annotation to generate HTML files
)

if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        zip_safe=False,
    )
