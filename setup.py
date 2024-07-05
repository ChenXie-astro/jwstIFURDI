#!/usr/bin/env python
from setuptools import setup, find_packages
try:
    from distutils.config import ConfigParser
except ImportError:
    from configparser import ConfigParser

# Read configuration variables in from setup.cfg
conf = ConfigParser()
conf.read(['setup.cfg'])

# Get some config values
metadata = dict(conf.items('metadata'))
PACKAGENAME = metadata.get('package_name', 'packagename')
DESCRIPTION = metadata.get('description', '')
AUTHOR = metadata['author']
AUTHOR_EMAIL = metadata['author_email']
URL = metadata['url']
LICENSE = metadata['license']
VERSION = metadata['version']





# import numpy as np
# from astropy.io import fits

# from skimage.transform import rotate
# from time import time
# import os
# import numpy as np
# from scipy.interpolate import interp2d
# from scipy import ndimage
# from scipy import optimize

# # from scipy.integrate import simps
# from scipy.ndimage import fourier_shift
# from scipy.ndimage import shift as spline_shift
setup(
    name=PACKAGENAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    py_modules=["jwstIFURDI"],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    install_requires=['numpy', 'scipy', 'astropy', 'skimage'],
    packages=find_packages(),
    package_data={},
)
