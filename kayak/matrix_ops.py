# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import collections
import numpy as np
import scipy.linalg as spla
from . import Differentiable

class MatMult(Differentiable):
    __slots__ = ['A', 'B']
    def __init__(self, A, B, *args):
        # Recurse to handle lists of arguments.
        if len(args) > 0:
            B = MatMult(B, *args)
        super(MatMult, self).__init__((A, B))
        self.A = A
        self.B = B

    def _compute_value(self):
        A_val, B_val = self.A.value, self.B.value
        if A_val.ndim > 2 or B_val.ndim > 2:
            raise Exception("Inputs of shape %s and %s are not matrices or vectors" % (self.A.shape))
        if A_val.shape[-1] != B_val.shape[0]:
            raise Exception("Cannot multiply %s by %s matrices." % (self.A.shape, self.B.shape))

        return np.dot(self.A.value, self.B.value)

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            B_val = self.B.value
            if B_val.ndim == 2:
                return np.dot(d_out_d_self, B_val.T)
            else:
                return np.outer(d_out_d_self, B_val).reshape(self.A.shape)
        elif parent == 1:
            A_val = self.A.value
            if A_val.ndim ==2:
                return np.dot(A_val.T, d_out_d_self)
            else:
                return np.outer(A_val, d_out_d_self).reshape(self.B.shape)
        else:
            raise Exception("Not a parent of me")

class MatSum(Differentiable):
    __slots__ = ['A', 'axis', 'keepdims']
    def __init__(self, A, axis=None, keepdims=True):
        super(MatSum, self).__init__((A,))
        if axis is not None and type(axis) != int:
            raise Exception("Can only sum over one axis at a time.")
        self.A    = A
        self.axis = axis
        self.keepdims = keepdims

    def _compute_value(self):
        return np.sum(self.A.value, axis=self.axis, keepdims=self.keepdims)

    def _local_grad(self, parent, d_out_d_self):
        # If self.keepdims == False then we need to
        # broadcast d_out_d_self along the summation axis
        if not self.keepdims and self.axis is not None:
            expanded_d_out_d_self = np.expand_dims(d_out_d_self, self.axis)
            return expanded_d_out_d_self * np.ones(self.A.shape)
        else:
            return d_out_d_self * np.ones(self.A.shape)

class MatMean(Differentiable):
    __slots__ = ['A', 'axis', 'keepdims']
    def __init__(self, A, axis=None, keepdims=True):
        super(MatMean, self).__init__((A,))
        if axis is not None and type(axis) != int:
            raise Exception("Can only take the mean over one axis at a time.")
        self.A    = A
        self.axis = axis
        self.keepdims = keepdims

    def _compute_value(self):
        return np.mean(self.A.value, axis=self.axis, keepdims=self.keepdims)

    def _local_grad(self, parent, d_out_d_self):
        # If self.keepdims == False then we need to
        # broadcast d_out_d_self along the summation axis
        N = float(self.A.value.size) if self.axis is None else float(self.A.shape[self.axis])
        if not self.keepdims and self.axis is not None:
            expanded_d_out_d_self = np.expand_dims(d_out_d_self, self.axis)
            return expanded_d_out_d_self * 1.0/N * np.ones(self.A.shape)
        else:
            return d_out_d_self * 1.0/N * np.ones(self.A.shape)

# class MatAdd(Differentiable):
#     __slots__ = ['A','B']
#     def __init__(self, A, B):
#         super(MatAdd, self).__init__((A,B))
#         self.A = A
#         self.B = B

#     def _compute_value(self):
#         return self.A.value + self.B.value

#     def _local_grad(self, parent, d_out_d_self):
#         if parent == 0:
#             if self.A.value.ndim >= self.B.value.ndim:
#                 # A has equal or more dimensions than B so post-broadcasting
#                 # we will sum over the singleton dimensions of A.
#                 singleton_inds = np.array(self.A.shape) == 1
#                 sum_dims = tuple(np.arange(self.A.value.ndim)[singleton_inds])
#             else:
#                 # A has fewer dimensions than B so post-broadcasting
#                 # we will sum over the extra dimensions at the beginning of the array.
#                 sum_dims = tuple(range(0, self.B.value.ndim - self.A.value.ndim))

#             if sum_dims:
#                 return np.sum(d_out_d_self, sum_dims).reshape(self.A.shape)
#             else:
#                 return d_out_d_self
#         elif parent == 1:
#             if self.B.value.ndim >= self.A.value.ndim:
#                 # B has equal or more dimensions than A so post-broadcasting
#                 # we will sum over the singleton dimensions of B.
#                 singleton_inds = np.array(self.B.shape) == 1
#                 sum_dims = tuple(np.arange(self.B.value.ndim)[singleton_inds])
#             else:
#                 # B has fewer dimensions than A so post-broadcasting
#                 # we will sum over the extra dimensions at the beginning of the array.
#                 sum_dims = tuple(range(0, self.A.value.ndim - self.B.value.ndim))
#             if sum_dims:    
#                 return np.sum(d_out_d_self, sum_dims).reshape(self.B.shape)
#             else:
#                 return d_out_d_self

# class MatElemMult(Differentiable):
#     __slots__ = ['A','B']
#     def __init__(self, A, B):
#         super(MatElemMult, self).__init__((A,B))
#         self.A = A
#         self.B = B

#     def _compute_value(self):
#         return self.A.value * self.B.value

#     def _local_grad(self, parent, d_out_d_self):
#         if parent == 0:
#             if self.A.value.ndim >= self.B.value.ndim:
#                 # A has equal or more dimensions than B so post-broadcasting
#                 # we will sum over the singleton dimensions of A.
#                 singleton_inds = np.array(self.A.shape) == 1
#                 sum_dims = tuple(np.arange(self.A.value.ndim)[singleton_inds])
#             else:
#                 # A has fewer dimensions than B so post-broadcasting
#                 # we will sum over the extra dimensions at the beginning of the array.
#                 sum_dims = tuple(range(0, self.B.value.ndim - self.A.value.ndim))

#             if sum_dims:
#                 return np.sum(d_out_d_self*self.B.value, sum_dims).reshape(self.A.shape)
#             else:
#                 return d_out_d_self*self.B.value
#         elif parent == 1:
#             if self.B.value.ndim >= self.A.value.ndim:
#                 # B has equal or more dimensions than A so post-broadcasting
#                 # we will sum over the singleton dimensions of B.
#                 singleton_inds = np.array(self.B.shape) == 1
#                 sum_dims = tuple(np.arange(self.B.value.ndim)[singleton_inds])
#             else:
#                 # B has fewer dimensions than A so post-broadcasting
#                 # we will sum over the extra dimensions at the beginning of the array.
#                 sum_dims = tuple(range(0, self.A.value.ndim - self.B.value.ndim))

#             if sum_dims:
#                 return np.sum(d_out_d_self*self.A.value, sum_dims).reshape(self.B.shape)
#             else:
#                 return d_out_d_self*self.A.value

class MatAdd(Differentiable):
    __slots__ = []
    def __init__(self, *args):
        super(MatAdd, self).__init__(args)

    def _compute_value(self):
        return sum([p.value for p in self._parents])

    def _local_grad(self, parent, d_out_d_self):
        parent_shape = self._parents[parent].shape
        num_singletons = len(d_out_d_self.shape) - len(parent_shape)
        if num_singletons > 0:
            extra_singletons = tuple(range(num_singletons))
            result = np.sum(d_out_d_self, axis=extra_singletons, keepdims=False)
        else:
            result = d_out_d_self

        assert len(result.shape) == len(parent_shape)
        original_singletons = tuple(np.where(np.array(parent_shape) == 1)[0])
        return np.sum(result, axis=original_singletons, keepdims=True).reshape(parent_shape)

class MatElemMult(Differentiable):
    """
    Elementwise multiplication of two arrays of the same size.
    Note: This does not support broadcasting yet. Look at MatAdd for ideas.
    """
    __slots__ = ['A', 'B']
    def __init__(self, A, B, *args):
        # Recurse to handle lists of arguments.
        if len(args) > 0:
            B = MatElemMult(B, *args)

        super(MatElemMult, self).__init__((A,B))

        self.A = A
        self.B = B

    def _compute_value(self):
        return self.A.value * self.B.value

    def _local_grad(self, parent, d_out_d_self):
        """
        For element-wise multiplication d(A*B)/dA = d_out_d_self * B.
        However, to support  broadcasting, we need to sum over the broadcast dimensions.
        For  example, d(A*x)/dx, where A is a matrix and x is a scalar, is
        given by \sum_{d1} \ldots \sum_{dD} (d_out_d_self * A)[d1,...,dD]
        """
        parent_shape = self._parents[parent].shape
        other_parent = 1 if parent == 0 else 0
        other_parent_value = self._parents[other_parent].value

        # Compute how many dimensions was parent broadcast along
        num_singletons = len(d_out_d_self.shape) - len(parent_shape)
        if num_singletons > 0:
            extra_singletons = tuple(range(num_singletons))
            # Sum out the broadcast dimensions
            result = np.sum(d_out_d_self*other_parent_value, axis=extra_singletons, keepdims=False)
        else:
            result = d_out_d_self*other_parent_value

        if parent == 0:
            final_shape = self.A.shape
        elif parent == 1:
            final_shape = self.B.shape

        # In mutliplying, we may have broadcast the parent.
        # Sum out those dimensions as well.
        assert len(result.shape) == len(parent_shape)
        original_singletons = tuple(np.where(np.array(parent_shape) == 1)[0])
        return np.sum(result, axis=original_singletons, keepdims=True).reshape(final_shape)

class MatDet(Differentiable):
    __slots__ = ['A']
    def __init__(self, A, axis=None, keepdims=True):
        super(MatDet, self).__init__((A,))
        self.A = A

    def _compute_value(self):
        return np.linalg.det(self.A.value)

    def _local_grad(self, parent, d_out_d_self):
        det = self._compute_value()
        return d_out_d_self * det * np.linalg.inv(self.A.value).T

class MatLogDet(Differentiable):
    __slots__ = ['A','L']
    def __init__(self, A):
        super(MatLogDet, self).__init__((A,))
        self.A = A
        self.L = None

    def _compute_value(self):
        self.L = spla.cholesky(self.A.value,lower=True)
        return 2*np.sum(np.log(np.diag(self.L)))

    def _local_grad(self, parent, d_out_d_self):
        val = self.value # Use cached cholesky if possible
        return spla.cho_solve((self.L,True),np.eye(self.A.shape[0]).dot(d_out_d_self))

class MatTrace(Differentiable):
    __slots__ = ['A']
    def __init__(self, A):
        super(MatTrace, self).__init__((A,))
        self.A = A

    def _compute_value(self):
        return np.trace(self.A.value)

    def _local_grad(self, parent, d_out_d_self):
        return d_out_d_self*np.eye(self.A.shape[0])

class MatInv(Differentiable):
    __slots__ = ['A']
    def __init__(self, A):
        super(MatInv, self).__init__((A,))
        self.A = A

    def _compute_value(self):
        return np.linalg.inv(self.A.value)

    def _local_grad(self, parent, d_out_d_self):
        return -self.value.T.dot(d_out_d_self).dot(self.value.T)

class MatDiag(Differentiable):
    __slots__ = ['A']
    def __init__(self, A):
        super(MatDiag, self).__init__((A,))
        self.A = A

    def _compute_value(self):
        return np.diag(self.A.value)

    def _local_grad(self, parent, d_out_d_self):
        return np.diag(d_out_d_self)

class MatEye(Differentiable):
    __slots__ = ['size']
    def __init__(self, size):
        super(MatEye, self).__init__(())
        self.size = size

    def _compute_value(self):
        return np.eye(self.size)

    def _local_grad(self, parent, d_out_d_self):
        return np.zeros(self.size)

class MatShape(Differentiable):
    __slots__ = ['A', 'dims']
    def __init__(self, A, dims=None):
        super(MatShape, self).__init__((A,))
        self.A = A
        self.dims = dims

    def _compute_value(self):
        if self.dims is None:
            return self.A.shape
        else:
            if isinstance(self.dims, collections.Iterable):
                return np.array([self.A.shape[i] for i in self.dims])
            else:
                return np.array(self.A.shape[self.dims])

    def _local_grad(self, parent, d_out_d_self):
        return np.zeros(self.A.shape)

class MatKron(Differentiable):
    """
    Currently only guaranteed to work for 2D arrays.
    """
    __slots__ = ['A', 'B']
    def __init__(self, A, B):
        super(MatKron, self).__init__([A,B])
        self.A = A
        self.B = B

    def _compute_value(self):
        return np.kron(self.A.value, self.B.value)

    def _local_grad(self, parent, d_out_d_self):
        # This is just an exercise in numpy array manipulation...
        if parent == 0:
            O = np.ones(self.A.shape)
            temp = np.add.reduceat(d_out_d_self*np.kron(O,self.B.value),np.arange(0,self.value.shape[1],self.B.shape[1]),axis=1)
            return np.add.reduceat(temp,np.arange(0,self.value.shape[0],self.B.shape[0]),axis=0)
        elif parent == 1:
            # The above case handles kron(A,B), so we first rotate d_out_d_self
            # to look like d_kron(B,A) and then take the derivative wrt B.
            temp = np.take(d_out_d_self,np.arange(self.value.shape[1]).reshape((self.A.shape[1],self.B.shape[1])).T.ravel(),axis=1)
            d_temp = np.take(temp,np.arange(self.value.shape[0]).reshape((self.A.shape[0],self.B.shape[0])).T.ravel(),axis=0)
            O = np.ones(self.B.shape)
            temp = np.add.reduceat(d_temp*np.kron(O,self.A.value),np.arange(0,self.value.shape[1],self.A.shape[1]),axis=1)
            return np.add.reduceat(temp,np.arange(0,self.value.shape[0],self.A.shape[0]),axis=0)

class Reshape(Differentiable):
    __slots__ = ['A', 'new_shape']

    def __init__(self, A, new_shape):
        super(Reshape, self).__init__((A,))
        self.A         = A
        self.new_shape = new_shape

    def _compute_value(self):
        return np.reshape(self.A.value, self.new_shape)

    def _local_grad(self, parent, d_out_d_self):
        return np.reshape(d_out_d_self, self.A.shape)

class ExpandDims(Differentiable):
    __slots__ = ['A', 'axis']

    def __init__(self, A, axis):
        super(ExpandDims, self).__init__((A,))
        self.A    = A
        self.axis = axis

    def _compute_value(self):
        return np.expand_dims(self.A.value, self.axis)

    def _local_grad(self, parent, d_out_d_self):
        return np.squeeze(d_out_d_self, self.axis)

class Concatenate(Differentiable):
    __slots__ = ['axis']
    def __init__(self, axis, *args):
        super(Concatenate, self).__init__(args)
        self.axis = axis

    def _compute_value(self):
        return np.concatenate([p.value for p in self._parents], axis=self.axis)

    def _local_grad(self, parent_ix, d_out_d_self):
        # Return the gradient only w.r.t. the matrix indexed by parent.
        start_ix = sum([p.shape[self.axis] for p in self._parents[0:parent_ix]])
        end_ix = start_ix + self._parents[parent_ix].shape[self.axis]
        return index_along_axis(d_out_d_self, self.axis, start_ix, end_ix)

class ListToArray(Differentiable):
    """Build an array out of a list of numbers."""
    __slots__ = []
    def __init__(self, *args):
        super(ListToArray, self).__init__(args)

    def _compute_value(self):
        return np.array([p.value for p in self._parents])

    def _local_grad(self, parent_ix, d_out_d_self):
        return d_out_d_self[parent_ix]

def index_along_axis(array, axis, start, end):
    """Return everything up to but not including end.

    For example:
    >>> index_along_axis(np.randn(10,20), 0, 10, 12).shape
    (2, 20)
    """
    full_slice = [slice(None),] * array.ndim
    full_slice[axis] = slice(start,end)
    return array[full_slice]

class TensorMult(Differentiable):
    __slots__ = ['axes']
    def __init__(self, A, B, axes):
        super(TensorMult, self).__init__((A, B))
        self.axes = axes

    def _compute_value(self):
        A = self._parents[0].value
        B = self._parents[1].value
        return safe_tensordot(A, B, self.axes)

    def _local_grad(self, parent, d_out_d_self):
        diff = lambda A, B : [a for a in A if a not in B]
        rank = lambda L : list(np.argsort(np.argsort(L)))
        val = [p.value for p in self._parents]
        axes = self.axes
        n_axes = len(axes[0])
        ignore_dims = [diff(range(val[i].ndim), axes[i]) for i in (0, 1)]
        ignore_ndims = [len(x) for x in ignore_dims]
        output_dims = (range(ignore_ndims[0]),
                       range(ignore_ndims[0], ignore_ndims[0] + ignore_ndims[1]))
        X, Y = parent, 1 - parent
        wrong_order = safe_tensordot(val[Y], d_out_d_self, (ignore_dims[Y], output_dims[Y]))
        permutation = [None] * val[X].ndim
        for final, cur in zip(list(axes[X]) + ignore_dims[X],
                              rank(axes[Y]) + range(n_axes, val[X].ndim)):
            permutation[final] = cur

        return np.transpose(wrong_order, permutation)

def safe_tensordot(A, B, axes):
    """Allows dimensions of length zero"""
    Adims, Bdims = list(A.shape), list(B.shape)
    if np.any([d is 0 for d in Adims + Bdims]):
        Anewdims = [d for i, d in enumerate(Adims) if i not in axes[0]]
        Bnewdims = [d for i, d in enumerate(Bdims) if i not in axes[1]]
        return np.zeros(Anewdims + Bnewdims)
    else:
        return np.tensordot(A, B, axes)

class Identity(Differentiable):
    __slots__ = []
    def __init__(self, A):
        super(Identity, self).__init__((A,))

    def _compute_value(self):
        return self._parents[0].value

    def _local_grad(self, parent_ix, d_out_d_self):
        return d_out_d_self

class Comparison(Differentiable):
    __slots__ = ['A', 'B', 'op']
    def __init__(self, A, B, op):
        super(Comparison, self).__init__((A, B))
        self.A  = A
        self.B  = B
        self.op = op

    def _compute_value(self):
        return self.op(self.A.value, self.B.value)

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            return np.zeros(self.A.shape)
        else:
            return np.zeros(self.B.shape)
