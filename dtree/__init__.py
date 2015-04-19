#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .dtreelib import *

def ErrorTree(bu_reg=7, h_smooth=100, depth_max=6, ratings_min=200000, top_pop=0, num_threads=1, randomize=False, rand_coeff=10):
    return ErrorTreePy(bu_reg, h_smooth, depth_max, ratings_min, top_pop, num_threads, randomize, rand_coeff)

def RankNDCGTree(bu_reg=7, h_smooth=100, depth_max=6, ratings_min=200000, top_pop=0, num_threads=1, randomize=False, rand_coeff=10):
    return RankNDCGTreePy(bu_reg, h_smooth, depth_max, ratings_min, top_pop, num_threads, randomize, rand_coeff)

def ErrorTreeTraverser(tree):
    return ErrorTreeTraverserPy(tree)

def RankNDCGTreeTraverser(tree):
    return RankNDCGTreeTraverserPy(tree)
