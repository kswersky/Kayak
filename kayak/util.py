# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import numpy        as np
import itertools    as it
from scipy.misc import logsumexp as sc_logsumexp

from . import EPSILON

def checkgrad(variable, output, epsilon=1e-4, verbose=False):
#    if not isinstance(variable, Parameter):
#        raise Exception("Cannot evaluate gradient in terms of non-Parameter type %s", (type(variable)))

    an_grad = output.grad(variable)
    fd_grad = np.zeros(variable.shape)
    
    base_value = variable.value.copy()
    for in_dims in it.product(*map(range, variable.shape)):
        small_array = np.zeros(variable.shape)
        small_array[in_dims] = epsilon

        variable.value = base_value - 2*small_array
        fn_l2 = output.value
        variable.value = base_value - small_array
        fn_l1 = output.value
        variable.value = base_value + small_array
        fn_r1 = output.value
        variable.value = base_value + 2*small_array
        fn_r2 = output.value

        fd_grad[in_dims] = ((fn_l2 - fn_r2)/12. + (- fn_l1 + fn_r1)*2./3.) /epsilon # 2nd order method
        # fd_grad[in_dims] = (- fn_l1/2. + fn_r1/2.) /epsilon # 1st order method

        if verbose:
            print np.abs((an_grad[in_dims] - fd_grad[in_dims])/(fd_grad[in_dims]+EPSILON)), an_grad[in_dims], fd_grad[in_dims]

    variable.value = base_value
    print "Mean finite difference", np.mean(np.abs((an_grad - fd_grad)/(fd_grad+EPSILON)))
    return np.mean(np.abs((an_grad - fd_grad)/(fd_grad+EPSILON)))

def checkgrad_symmetric_matrix(variable, output, epsilon=1e-4, verbose=False):
#    if not isinstance(variable, Parameter):
#        raise Exception("Cannot evaluate gradient in terms of non-Parameter type %s", (type(variable)))

    assert variable.value.ndim == 2 and np.allclose(variable.value, variable.value.T), "Only works with 2D, symmetric matrices."

    an_grad = output.grad(variable)
    an_grad = 0.5*(an_grad+an_grad.T)
    fd_grad = np.zeros(variable.shape)
    base_value = variable.value.copy()

    for i in xrange(variable.shape[0]):
        for j in xrange(i+1):
            small_array = np.zeros(variable.shape)
            small_array[i,j] = epsilon
            small_array[j,i] = epsilon

            variable.value = base_value - 2*small_array
            fn_l2 = output.value
            variable.value = base_value - small_array
            fn_l1 = output.value
            variable.value = base_value + small_array
            fn_r1 = output.value
            variable.value = base_value + 2*small_array
            fn_r2 = output.value

            if i == j:
                fd_grad[i,i] = ((fn_l2 - fn_r2)/12. + (- fn_l1 + fn_r1)*2./3.) /epsilon # 2nd order method
            else:
                fd_grad[i,j] = 0.5*((fn_l2 - fn_r2)/12. + (- fn_l1 + fn_r1)*2./3.) /epsilon # 2nd order method
                fd_grad[j,i] = 0.5*((fn_l2 - fn_r2)/12. + (- fn_l1 + fn_r1)*2./3.) /epsilon # 2nd order method

            if verbose:
                print np.abs((an_grad[i,j] - fd_grad[i,j])/(fd_grad[i,j]+EPSILON)), an_grad[i,j], fd_grad[i,j]

    variable.value = base_value
    print "Mean finite difference", np.mean(np.abs((an_grad - fd_grad)/(fd_grad+EPSILON)))
    return np.mean(np.abs((an_grad - fd_grad)/(fd_grad+EPSILON)))

def logsumexp(X, axis=None, keepdims=False):
    if keepdims:
        return np.expand_dims(sc_logsumexp(X,axis),axis)
    else:
        return sc_logsumexp(X,axis)

def onehot(T, num_labels=None):
    if num_labels is None:
        num_labels = np.max(T)+1
    labels = np.zeros((T.shape[0], num_labels), dtype=bool)
    labels[np.arange(T.shape[0], dtype=int), T] = 1
    return labels

