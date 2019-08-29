#============================================================================================
# Name        : compile_asym_linkage.py
# Author      : Samuel Marchal, Sebastian Szyller
# Version     : 1.0
# Copyright   : Copyright (C) Secure Systems Group, Aalto University {https://ssg.aalto.fi/}
# License     : This code is released under Apache 2.0 license
#============================================================================================


from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name='asym_linkage',
    ext_modules=cythonize(
        Extension(
            "asym_linkage",
            sources=["asym_linkage.pyx"],
            include_dirs=[np.get_include()]
        )
    ),
    install_requires=["numpy"]
)
