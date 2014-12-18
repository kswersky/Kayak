# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import numpy as np

from . import Differentiable
from scipy.misc import logsumexp

class Loss(Differentiable):
    __slots__ = ['preds', 'targs']
    def __init__(self, predictions, targets):
        super(Loss, self).__init__((predictions, targets))
        self.preds  = predictions
        self.targs  = targets

class L2(Loss):
    def __init__(self, predictions, targets):
        super(L2, self).__init__(predictions, targets)

    def _compute_value(self):
        return (self.preds.value - self.targs.value)**2

    def _local_grad(self, parent, d_out_d_self):
        assert parent is 0, "Shouldn't be taking derivative wrt targets"
        return 2 * (self.preds.value - self.targs.value) * d_out_d_self

class L1Loss(Loss):
    def __init__(self, predictions, targets):
        super(L1Loss, self).__init__(predictions, targets)

    def _compute_value(self):
        return np.abs(self.preds.value - self.targs.value)

    def _local_grad(self, parent, d_out_d_self):
        assert parent is 0, "Shouldn't be taking derivative wrt targets"
        return np.sign(self.preds.value - self.targs.value) * d_out_d_self

class LogMultinomial(Loss):
    def __init__(self, predictions, targets, axis=1, keepdims=True):
        # Predictions are log probabilities and targets are counts.
        super(LogMultinomial, self).__init__(predictions, targets)

    def _compute_value(self):
        return -self.targs.value*self.preds.value

    def _local_grad(self, parent, d_out_d_self):
        assert parent is 0, "Shouldn't be taking derivative wrt targets"
        return -d_out_d_self*self.targs.value

class CrossEntropy(Loss):
    def __init__(self, predictions, targets):
        # Predictions are probabilities and targets are binary labels.
        super(CrossEntropy, self).__init__(predictions, targets)

    def _compute_value(self):
        return -(self.targs.value * np.log(self.preds.value+1e-16) + (1.0-self.targs.value)*np.log(1.0-self.preds.value+1e-16))

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            num = self.preds.value - self.targs.value
            den = self.preds.value*(1.0-self.preds.value)

            factor = num/den
            return d_out_d_self * factor
        else:
            g = -np.log(self.preds.value) + np.log(1.0-self.preds.value)
            return g*d_out_d_self

class LatentCrossEntropy(Loss):
    def __init__(self, latent_preds, targets):
        # Latent predictions are log probabilities and targets are binary labels.
        super(LatentCrossEntropy, self).__init__(latent_preds, targets)

    def _sigmoid(self, x):
        return np.exp(-logsumexp(np.broadcast_arrays(0,-x),0))

    def _compute_value(self):
        return (self.targs.value*logsumexp(np.broadcast_arrays(0,-self.preds.value),0)
                + (1-self.targs.value)*logsumexp(np.broadcast_arrays(0,self.preds.value),0))

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            probs = self._sigmoid(self.preds.value)
            nprobs = self._sigmoid(-self.preds.value)
            return d_out_d_self * (-self.targs.value*nprobs + (1-self.targs.value)*probs)
        else:
            g = (logsumexp(np.broadcast_arrays(0,-self.preds.value),0)
                 - logsumexp(np.broadcast_arrays(0,self.preds.value),0))
            return g*d_out_d_self


