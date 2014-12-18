# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.
# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import numpy as np
from . import Differentiable
import matrix_ops

class Elementwise(Differentiable):
    __slots__ = ['X']
    def __init__(self, X):
        super(Elementwise, self).__init__(X)
        self.X = X

    def _compute_shape(self, inputs=None):
        return self.X.shape

# Just an alias for matrix addition and elementwise multiplication.
ElemAdd = matrix_ops.MatAdd
ElemMult = matrix_ops.MatElemMult

class ElemExp(Elementwise):
    """
    Elementwise exponentiation of an array
    """
    __slots__ = ['A']
    def __init__(self, A):
        super(ElemExp, self).__init__([A])
        self.A = A

    def _compute_value(self):
        return np.exp(self.A.value)

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            return d_out_d_self * np.exp(self.A.value)
        else:
            raise Exception("Not a parent of me")

class ElemLog(Elementwise):
    """
    Elementwise logarithm of an array
    """
    __slots__ = ['A']
    def __init__(self, A):
        super(ElemLog, self).__init__([A])
        self.A = A

    def _compute_value(self):
        return np.log(self.A.value)

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            return d_out_d_self / self.A.value
        else:
            raise Exception("Not a parent of me")

class ElemPower(Elementwise):
    """
    Elementwise power of an array.

    NOTE: Fractional powers are only defined for positive bases.
          We do not check for this; numpy will throw a runtime exception.

    NOTE: This only supports A**p where p is a scalar value.
    """
    __slots__ = ['A', 'p']
    def __init__(self, A, p):
        if isinstance(p, Differentiable):
            assert p.value.size == 1, 'ElemPower only allows scalar powers.'
            super(ElemPower, self).__init__([A,p])
        else:
            assert np.isscalar(p), 'ElemPower only allows scalar powers.'
            super(ElemPower, self).__init__([A])

        self.A = A
        self.p = p

    def _compute_value(self):
        if isinstance(self.p, Differentiable):
            p = np.squeeze(self.p.value)
        else:
            p = self.p
            
        return self.A.value**p

    def _local_grad(self, parent, d_out_d_self):
        if isinstance(self.p, Differentiable):
            p = np.squeeze(self.p.value)
        else:
            if parent > 0:
                raise Exception('Not a parent of me.')
            p = self.p

        if parent == 0:
            return d_out_d_self * p * self.A.value**(p-1)
        else:
            if np.issubdtype(self.A.value.dtype, np.integer) and not (self.A.value & 0x1):
                return np.sum(d_out_d_self * self.value * np.log(np.abs(self.A.value)))
            else:
                return np.sum(d_out_d_self * self.value * np.log(self.A.value))


class ElemAbs(Elementwise):
    """
    Elementwise absolute value of an array.
    """
    __slots__ = ['A']
    def __init__(self, A):
        super(ElemAbs, self).__init__([A])
        self.A = A

    def _compute_value(self):
        return abs(self.A.value)

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            return d_out_d_self * np.sign(self.A.value)
        else:
            raise Exception("Not a parent of me")
