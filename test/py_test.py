#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dtree import ErrorTree, ErrorTreeTraverser, RankNDCGTree, RankNDCGTreeTraverser
from random import random, seed

seed(1)
nusers = 3000
nitems = 4000
density = .0001

tree = ErrorTree(ratings_min=1, num_threads=4, depth_max=4, randomize=True, cache_enabled=True)
tdata = [(int(random()*nusers), int(random()*nitems), int(random()*4+1)) for i in xrange(0, int(nusers*nitems*density))]
tree.init(tdata)
tree.build([i for i in xrange(0,500)], True)

traverser = ErrorTreeTraverser(tree)
print traverser.current_query()
print traverser.at_leaf()
traverser.traverse_unknown()
print traverser.current_query()
print traverser.at_leaf()
traverser.traverse_unknown()
print traverser.current_query()
print traverser.at_leaf()
traverser.traverse_unknown()
print traverser.current_query()
print traverser.at_leaf()

tree = RankNDCGTree(ratings_min=1, num_threads=4, depth_max=4, randomize=True, cache_enabled=True)
tdata = [(int(random()*nusers), int(random()*nitems), int(random()*4+1)) for i in xrange(0, int(nusers*nitems*density))]
tree.init(tdata)
tree.build([i for i in xrange(0,500)], True)

traverser = RankNDCGTreeTraverser(tree)
print traverser.current_query()
print traverser.at_leaf()
traverser.traverse_unknown()
print traverser.current_query()
print traverser.at_leaf()
traverser.traverse_unknown()
print traverser.current_query()
print traverser.at_leaf()
traverser.traverse_unknown()
print traverser.current_query()
print traverser.at_leaf()

