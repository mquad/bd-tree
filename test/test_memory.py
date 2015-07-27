#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dtree import ErrorTree, ErrorTreeTraverser
from random import random, seed
import resource
import gc

def printMemoryUsage():
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


seed(1)
nusers = 1000
nitems = 4000
density = .005

printMemoryUsage()

trees = [(i, ErrorTree(ratings_min=1, num_threads=1, depth_max=7, randomize=False, cache_enabled=False, top_pop=200)) for i in xrange(0,4)]

printMemoryUsage()

for tid,tree in trees:
    tdata = [(int(random()*nusers), int(random()*nitems), int(random()*4+1)) for i in xrange(0, int(nusers*nitems*density))]
    tree.init(tdata)
    tree.build([i for i in xrange(0,nusers)], False)
    printMemoryUsage()

trees = []
gc.collect()
printMemoryUsage()


