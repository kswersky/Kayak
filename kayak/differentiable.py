# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import weakref
import numpy as np

from Queue import Queue as SimpleQueue


class ndarray(np.ndarray):

    def __new__(cls, input_array, parent=None):
        obj = np.asarray(input_array).view(cls)
        obj.parent = parent
        return obj

    def __getitem__(self, index):
        return super(ndarray, self).__getitem__(index)

    def __setitem__(self, index, value):
        super(ndarray, self).__setitem__(index, value)
        if hasattr(self, 'parent') and self.parent is not None:
            self.parent._clear_value_cache()
            self.parent._clear_grad_cache()
            self.parent._value = np.atleast_1d(self)

class Differentiable(object):
    __slots__ = ['_value', '_grad', '_loss', '_parents', '_children', '_parent_indices', '_T', '__weakref__', 'stuff']
    def __init__(self, parents=()):
        self._value = None # Cached value
        self._grad  = None # Cached grad
        self._loss  = None # Loss we are caching with respect to
        self._T     = None

        self._parents  = tuple(parents)
        for parent_index, parent in enumerate(parents):
            parent._add_child(self, parent_index)

        self._children = weakref.WeakValueDictionary()

    @property
    def T(self):
        if self._T is None:
            self._T = Transpose(self)
        
        return self._T

    @property
    def value(self):
        """Compute the value of the function.  This walks up the
        dependency graph and finds all of the Kayak objects with known
        values (such as Inputs and Targets, perhaps modulated by a
        Batcher) and then propagates their values forward through the
        modular computations of Differentiable subclasses.  There are
        some subtleties to this behavior, which are determined by the
        method arguments.
        """
        # If the value is not yet cached, compute it.
        if self._value is None:
            self._value = self._compute_value()

        return np.array(self._value)

    @value.setter
    def value(self, new_value):
        self._clear_value_cache()
        if not isinstance(new_value, np.ndarray):
            new_value = np.array(new_value)
        self._value = ndarray(np.atleast_1d(new_value),parent=self)

    @property
    def _children_with_parent_indices(self):
        return [(self._children[key], key[1]) for key in self._children.keys()]

    def _clear_value_cache(self):
        """
        Sets the new value and clears the values of any dependencies. We
        maintain the invariant that cached values are never wrong relative
        to their parents' values.
        """
        if self._value is not None:
            [child._clear_value_cache() for child in self._children.values()]
            self._clear_grad_cache()
            self._value = None

    def _clear_grad_cache(self):
        if self._grad is not None:
            [parent._clear_grad_cache() for parent in self._parents]
            self._grad = None
            self._loss = None
        
    def grad(self, other):
        """Compute the gradient of this module in terms of another
        module.  One of the main points of the Kayak setup is to
        easily compute gradients in terms of parameters.  This is the
        interface for doing so.  You call the grad() method on
        something that produces a scalar, providing as an argument
        some other object that appears deeper in the graph.  You get
        out an array of the same shape as the deeper object, but which
        is the gradient.

        Arguments:

          other: (Kayak object) The other object, in terms of which
                 you'd like to take this thing's gradient.
        """
        grad = other._d_out_d_self(self)
        if grad is 0:
            # Make sure the output has the expected shape
            grad = np.zeros(other.shape)

        return grad

    @property
    def shape(self):
        return self.value.shape

    @property
    def size(self):
        return self.value.size

    def sum(self, axis=None):
        from . import MatSum
        return MatSum(self, axis=axis)

    def mean(self, axis=None):
        from . import MatMean
        return MatMean(self, axis=axis)

    def dot(self, B):
        from . import MatMult
        return MatMult(self, B)

    def _d_out_d_self(self, out):
        # Cached grad is not valid or refers to a different loss,
        # so we need to recompute compute the gradient
        if self._grad is None or self._loss is not out:
            if self is out:
                grad = np.ones(self.shape)
            elif not self._children:
                grad = 0
            else:
                grad = sum(child._d_out_d_parent(out, parent_index)
                           for child, parent_index in self._children_with_parent_indices)

            self._loss = out
            self._grad = grad

        return self._grad

    def _d_out_d_parent(self, out, parent):
        d_out_d_self = self._d_out_d_self(out)
        if d_out_d_self is 0:
            # This avoid calling local_grad for paths that don't end in 'out'
            return 0
        else:
            return self._local_grad(parent, d_out_d_self)

    def _bfs(self):
        q = SimpleQueue()
        q.put(self)
        bfs_list = [self]

        visited = set()
        while not q.empty():
            node = q.get(block=False)
            for parent in node._parents:
                if not parent in visited:
                    visited.add(parent)
                    q.put(parent)
                    bfs_list.append(parent)

        return bfs_list

    def bfs_grad(self, other):
        bfs_list = self._bfs()
        for node in bfs_list:
            if node is self:
                node.stuff = np.ones(self.shape)
            elif not node._children:
                node.stuff = 0
            else:
                node.stuff = sum(child._local_grad(parent_index, child.stuff)
                         for child, parent_index in node._children_with_parent_indices
                         if child.stuff is not 0)

        return other.stuff

    def _add_child(self, child, parent_index):
        """Parent_index is an int that tells out child which parent we are."""
        self._children[(id(child), parent_index)] = child

    def _local_grad(self, parent, d_out_d_self):
        """Return d_out_d_self * d_self_d_parent"""
        raise Exception("Class 'Differentiable' is abstract.")

    def _compute_value(self):
        raise Exception("Class 'Differentiable' is abstract.")


    def __getitem__(self, index):
        from . import Slice
        return Slice(self, index)

    def __setitem__(self, index, new_val):
        self.value[index] = new_val

    # Overload plus and times operators with elementwise operations
    # To avoid circular imports, we wait until the operator is called
    # to import the subclasses of Differentiable
    def __add__(self, other):
        from . import ElemAdd, Constant

        # If other is not a Differentiable object,
        # try to cast it as a constant.
        if not isinstance(other, Differentiable):
            other = Constant(other)
        return ElemAdd(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __mul__(self, other):
        from . import ElemMult, Constant
        # If other is not a Differentiable object,
        # try to cast it as a constant.
        if not isinstance(other, Differentiable):
            other = Constant(other)
        return ElemMult(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    # NOTE: Assuming Python 2.x syntax for div
    def __div__(self, other):
        from . import ElemPower, Constant
        if not isinstance(other, Differentiable):
            other = Constant(other)
        return self * ElemPower(other, -1)

    def __rdiv__(self, other):
        from . import ElemPower, Constant
        if not isinstance(other, Differentiable):
            other = Constant(other)
        return other * ElemPower(self, -1)

    def __neg__(self):
        from . import ElemMult, Constant
        return ElemMult(Constant(-1.), self)

    def __pow__(self, power, modulo=None):
        from . import ElemPower
        return ElemPower(self, power)

    def __abs__(self):
        from . import ElemAbs
        return ElemAbs(self)

    def __eq__(self, other):
        from . import Comparison
        return Comparison(self, other, np.equal)

    def __ne__(self, other):
        from . import Comparison
        return Comparison(self, other, np.not_equal)

    def __gt__(self, other):
        from . import Comparison
        return Comparison(self, other, np.greater)

    def __lt__(self, other):
        from . import Comparison
        return Comparison(self, other, np.less)

    def __ge__(self, other):
        from . import Comparison
        return Comparison(self, other, np.greater_equal)

    def __le__(self, other):
        from . import Comparison
        return Comparison(self, other, np.less_equal)

class Transpose(Differentiable):
    __slots__ = ['A', 'axes']
    def __init__(self, A, axes=None):
        super(Transpose, self).__init__((A,))
        self.A    = A
        self.axes = axes

    @property
    def T(self):
        return self.A

    def _compute_value(self):
        return np.transpose(self.A.value, axes=self.axes)

    def _local_grad(self, parent, d_out_d_self):
        if self.axes is None:
            return np.transpose(d_out_d_self)
        else:
            return np.transpose(d_out_d_self, axes=np.argsort(self.axes))
