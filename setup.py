#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, Extension
import os
import sys

compiler_args=['-std=c++11','-fopenmp','-O3','-march=native','-DNDEBUG',]

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

setup(name='dtree',
      version='1.0.1',
      description='Adaptive Bootstrap Decistion Trees for Active Learning',
      packages=['dtree'],
      ext_modules=[Extension('dtree.dtreelib',
          ['src/dtreemodule.cpp'],
          libraries=['boost_python'],
          extra_compile_args=compiler_args,
          extra_link_args=['-fopenmp'])],
      url='xxx.yyy.com',
      author='Massimo Quadrana',
      author_email='massimo.quadrana@polimi.it'
    )
