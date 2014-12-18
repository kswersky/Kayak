# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import numpy as np

from . import Differentiable

class Regularizer(Differentiable):
    __slots__ = ['X', 'weight']
    def __init__(self, X, weight):
        super(Regularizer, self).__init__([X])
        self.X      = X
        self.weight = weight

class L2Norm(Regularizer):
    __slots__ = []
    def __init__(self, X, weight=1.0):
        super(L2Norm, self).__init__(X, weight)

    def _compute_value(self):
        return self.weight * np.sum(self.X.value**2)

    def _local_grad(self, parent, d_out_d_self):
        return self.weight * 2.0 * self.X.value * d_out_d_self

class L1Norm(Regularizer):
    __slots__ = []
    def __init__(self, X, weight=1.0):
        super(L1Norm, self).__init__(X, weight)

    def _compute_value(self):
        return self.weight * np.sum(np.abs(self.X.value))

    def _local_grad(self, parent, d_out_d_self):
        return self.weight * np.sign(self.X.value) * d_out_d_self

class Horseshoe(Regularizer):
    __slots__ = []
    def __init__(self, X, weight=1.0):
        super(Horseshoe, self).__init__(X, weight)

    def _compute_value(self):
        return -self.weight * np.sum(np.log(np.log(1.0 + self.X.value**(-2))))

    def _local_grad(self, parent, d_out_d_self):
        return -(self.weight * d_out_d_self * (1 / (np.log(1.0 + self.X.value**(-2))))
                 * (1.0/(1 + self.X.value**(-2))) * (-2*self.X.value**(-3)))

class NExp(Regularizer):
    __slots__ = []
    def __init__(self, X, weight=1.0):
        super(NExp, self).__init__(X, weight)

    def _compute_value(self):
        return self.weight * np.sum(1.0 - np.exp(-np.abs(self.X.value)))

    def _local_grad(self, parent, d_out_d_self):
        return self.weight * d_out_d_self * np.exp(-np.abs(self.X.value)) * np.sign(self.X.value)

class KLSparsity(Regularizer):
    __slots__ = ['target']
    def __init__(self, X, weight=1.0, target=0.2):
        super(KLSparsity, self).__init__(X, weight)
        self.target = target

    def _compute_value(self):
        acts = self.X.value.mean(0)
        return -self.weight * (self.target*np.log(acts) + (1-self.target)*(np.log(1-acts)))

    def _local_grad(self, parent, d_out_d_self):
        acts = self.X.value.mean(0)
        return -self.weight*d_out_d_self*((self.target / acts) - ((1-self.target) / (1-acts)))*(np.ones(self.X.shape)/self.X.shape[0])

class RunningAvgKLSparsity(Regularizer):
    __slots__ = ['target', 'running_avg', 'mo']
    def __init__(self, X, weight=1.0, target=0.2, mo=0.9):
        super(RunningAvgKLSparsity, self).__init__(X, weight)
        self.target      = target
        self.running_avg = 0
        self.mo          = mo

    def reset(self):
        self.running_avg = 0

    def add_to_avg(self, acts):
        if np.sum(self.running_avg) == 0:
            self.running_avg = acts.mean(0)
        else:
            self.running_avg = self.mo*self.running_avg + (1-self.mo)*acts.mean(0)

    def _compute_value(self):
        if np.sum(self.running_avg) == 0:
            mo = 0
        else:
            mo = self.mo

        acts = mo*self.running_avg + (1-mo)*self.X.value.mean(0)
        return -(self.weight/((1-mo))) * np.sum((self.target*np.log(acts) + (1-self.target)*(np.log(1-acts))))

    def _local_grad(self, parent, d_out_d_self):
        if np.sum(self.running_avg) == 0:
            mo = 0
        else:
            mo = self.mo

        acts = mo*self.running_avg + (1-mo)*self.X.value.mean(0)
        return -(self.weight/((1-mo)))*d_out_d_self*((self.target / acts) - ((1-self.target) / (1-acts)))*(1-mo)*(np.ones(self.X.shape)/self.X.shape[0])

class AsymRunningAvgKLSparsity(Regularizer):
    __slots__ = ['target', 'running_avg', 'mo']
    def __init__(self, X, weight=1.0, target=0.1, mo=0.9):
        super(AsymRunningAvgKLSparsity, self).__init__(X, weight)
        self.target      = target
        self.running_avg = 0
        self.mo          = mo

    def reset(self):
        self.running_avg = 0

    def add_to_avg(self, acts):
        if np.sum(self.running_avg) == 0:
            self.running_avg = acts.mean(0)
        else:
            self.running_avg = self.mo*self.running_avg + (1-self.mo)*acts.mean(0)

    def _compute_value(self):
        if np.sum(self.running_avg) == 0:
            mo = 0
        else:
            mo = self.mo

        acts = mo*self.running_avg + (1-mo)*self.X.value.mean(0)
        acts[acts > self.target] = self.target
        p1 = -(self.weight/((1-mo))) * np.sum((self.target*np.log(acts) + (1-self.target)*(np.log(1-acts))))

        acts = mo*self.running_avg + (1-mo)*self.X.value.mean(0)
        acts[acts < 1-self.target] = 1-self.target
        p2 = -(self.weight/((1-mo))) * np.sum((self.target*np.log(1-acts) + (1-self.target)*(np.log(acts))))

        return p1 + p2

    def _local_grad(self, parent, d_out_d_self):
        if np.sum(self.running_avg) == 0:
            mo = 0
        else:
            mo = self.mo
           
        acts = mo*self.running_avg + (1-mo)*self.X.value.mean(0)
        acts[acts > self.target] = self.target
        g1 = -(self.weight/((1-mo)))*d_out_d_self*((self.target / acts) - ((1-self.target) / (1-acts)))*(1-mo)*(np.ones(self.X.shape)/self.X.shape[0])
        
        acts = mo*self.running_avg + (1-mo)*self.X.value.mean(0)
        acts[acts < 1-self.target] = 1-self.target
        g2 = -(self.weight/((1-mo)))*d_out_d_self*(-(self.target / (1-acts)) + ((1-self.target) / acts))*(1-mo)*(np.ones(self.X.shape)/self.X.shape[0])

        return g1 + g2
        
