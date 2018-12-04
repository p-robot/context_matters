from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(  
    name='context_matters',
    version="0.1",
    description="Reinforcement learning example for foot-and-mouth disease control",
    author="Will Probert",
    author_email="willprobert@gmail.com",
    license="Apache License, Version 2.0",
    url="https://github.com/p-robot/context_matters",
    requires=['numpy','cython','pandas','matplotlib'],
    packages=["context_matters"],
    ext_modules = cythonize("context_matters/*.pyx"),
    classifiers = ['Development Status :: 3 - Alpha',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   'License :: OSI Approved :: Apache Software License',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Mathematics',
                   'Operating System :: OS Independent'],
    include_dirs = [np.get_include()]
    )
