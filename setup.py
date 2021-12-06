#!/usr/bin/env python
from __future__ import print_function

"""SimPEG: Simulation and Parameter Estimation in Geophysics

SimPEG is a python package for simulation and gradient based
parameter estimation in the context of geophysical applications.
"""

from distutils.core import setup
from setuptools import find_packages

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Natural Language :: English",
]

with open("README.md") as f:
    LONG_DESCRIPTION = "".join(f.readlines())

setup(
    name="MGeo",
    version="0.0.2",
    packages=find_packages(exclude=["tests*", "examples*", "tutorials*", "temp*", "pproc*", "front-end*"]),
    install_requires=[
        "numpy>=1.7",
        "scipy>=1.0.0",
        "scikit-learn>=0.22",
        "pymatsolver>=0.1.1",
        "matplotlib",
        "properties>=0.5.2",
        "vectormath>=0.2.0",
        "discretize>=0.7.0",
        "geoana>=0.0.4",
        "empymod",
        "pandas",
        "mkl",
        "pyvista",
    ],
    author="JMT",
    author_email="wavefrontgeo@gmail.com",
    description="MGEO package with modification to incorporate octree meshes with MT",
    long_description=LONG_DESCRIPTION,
    license="UFPA",
    keywords="geophysics",
    url="https://github.com/projetoemBR/mgeo",
    download_url="https://github.com/projetoemBR/mgeo",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3=False,
)