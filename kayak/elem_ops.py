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
from . import Differentiable, Parameter
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

    NOTE: This will not work for negative numbers raised to even powers.
    """
    __slots__ = ['A', 'p']
    def __init__(self, A, p):
        if not isinstance(p, Differentiable):
            p = Parameter(p)

        super(ElemPower, self).__init__([A,p])

        self.A = A
        self.p = p

    def _compute_value(self):
        return self.A.value**self.p.value

    def _local_grad(self, parent, d_out_d_self):
        p = self.p.value
        if parent == 0:
            inds = self.A.value == 0
            c    = self.A.value + inds
            return (1 - inds) * d_out_d_self * p * c**(p-1)
        else:
            c = self.A.value + (self.A.value == 0) # Special case subgradient if A is 0.
            ret = d_out_d_self * self.value * np.log(c)
            if self.p.value.size > 1:
                return ret
            else:
                return ret.sum()

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

class Maximum(Elementwise):
    """
    Elementwise maximum between two arrays.
    """
    __slots__ = ['A', 'B']
    def __init__(self, A, B):
        super(Maximum, self).__init__([A, B])
        self.A = A
        self.B = B

    def _compute_value(self):
        return np.maximum(self.A.value, self.B.value)

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            return d_out_d_self * (self.A.value >= self.B.value)
        else:
            return d_out_d_self * (self.A.value < self.B.value)

class Minimum(Elementwise):
    """
    Elementwise minimum between two arrays.
    """
    __slots__ = ['A', 'B']
    def __init__(self, A, B):
        super(Minimum, self).__init__([A, B])
        self.A = A
        self.B = B

    def _compute_value(self):
        return np.minimum(self.A.value, self.B.value)

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            return d_out_d_self * (self.A.value <= self.B.value)
        else:
            return d_out_d_self * (self.A.value > self.B.value)

Log = ElemLog
Exp = ElemExp

def Sqrt(X):
    p = Parameter(0.5)
    return ElemPower(X, p)
