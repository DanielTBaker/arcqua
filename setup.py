#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 07:30:00 2024

@author: DanielTBaker
"""

from setuptools import setup

setup(
    name='arcqua',
    version='0.5',
    description='Tools for measuring ocean wave directions from CYGNSS and TRITON data',
    long_description="See: `github.com/DanielTBaker/arcqua \
                      <https://github.com/DanielTBaker/arcqua>`_.",
    classifiers=[
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='DDMs, CYGNSS, scintillation',
    url='https://github.com/DanielTBaker/arcqua',
    author='Daniel T. Baker',
    author_email='dbaker@asiaa.sinica.edu.tw',
    license='MIT',
    packages=['arcqua'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'astropy','mechanize'],
    include_package_data=True,
    zip_safe=False,
)