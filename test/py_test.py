#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dtree import ErrorTree
from random import random

tree = ErrorTree(ratings_min=1, num_threads=4, depth_max=3, randomize=False)
tdata = [(int(random()*1000), int(random()*1000), int(random()*4+1)) for i in xrange(0,100000)]
tree.init(tdata)
tree.build()
