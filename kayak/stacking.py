# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import numpy        as np
import numpy.random as npr

from . import Differentiable

class Stack(Differentiable):
    __slots__ = ['A', 'B', 'axis']

    def __init__(self, A, B, axis):
        super(Stack, self).__init__([A, B])

        self.A    = A
        self.B    = B
        self.axis = axis

    def _compute_value(self):
        if self.axis == 0:
            return np.vstack((self.A.value, self.B.value))
        else:
            return np.hstack((self.A.value, self.B.value))

    def _local_grad(self, parent, d_out_d_self):
        if self.axis == 0:
            if parent == 0:
                return d_out_d_self[:self.A.shape[0],...]
            if parent == 1:
                return d_out_d_self[self.A.shape[0]:,...]
        else:
            if parent == 0:
                return d_out_d_self[...,:self.A.shape[-1]]
            if parent == 1:
                return d_out_d_self[...,self.A.shape[-1]:]

class Vstack(Stack):
    def __init__(self, A, B):
        super(Vstack, self).__init__(A,B,0)

class Hstack(Stack):
    def __init__(self, A, B):
        super(Hstack, self).__init__(A,B,1)
