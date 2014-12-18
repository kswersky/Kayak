# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import numpy as np

from . import Differentiable

class Slice(Differentiable):
    __slots__ = ['A', 'inds']
    def __init__(self, A, inds):
        super(Slice, self).__init__([A])

        self.A = A
        self.inds = inds

    def _compute_value(self):
        return self.A.value[self.inds]

    def _local_grad(self, parent, d_out_d_self):
        g = np.zeros(self.A.shape)
        g[self.inds] = d_out_d_self

        return g