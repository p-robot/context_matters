from distutils.core import setup
from Cython.Build import cythonize
import numpy as np


DESCRIPTION = "Reinforcement learning example for foot-and-mouth disease control"
NAME = "context_matters"
AUTHOR = "Will Probert"
AUTHOR_EMAIL = "william.probert@bdi.ox.ac.uk"
MAINTAINER = "Will Probert"
MAINTAINER_EMAIL = "william.probert@bdi.ox.ac.uk"
URL = "https://github.com/p-robot/context_matters"
DOWNLOAD_URL = "https://github.com/p-robot/context_matters"
LICENSE = "Apache License, Version 2.0"

import context_matters
VERSION = context_matters.__version__

setup(  
    name = NAME,
    version = VERSION,
    description = DESCRIPTION,
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    maintainer = MAINTAINER,
    maintainer_email = MAINTAINER_EMAIL,
    url = URL,
    download_url = DOWNLOAD_URL,
    license = LICENSE,
    requires=['numpy','cython','pandas','matplotlib'],
    packages=["context_matters", "context_matters.tests", "context_matters.data"],
    package_data={"context_matters": ["data/circular_3km_data_n4000_seed12.csv",]},
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
