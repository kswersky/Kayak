# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import sys
import hashlib
import numpy as np

EPSILON = sys.float_info.epsilon

from differentiable import Differentiable, Transpose
from root_nodes     import Constant, Parameter, DataNode, Inputs, Targets
from slicing        import Slice
from batcher        import Batcher
from matrix_ops     import MatAdd, MatMult, MatElemMult, MatSum, MatMean, MatEye, MatDiag, Reshape, Concatenate, Identity, TensorMult, ListToArray, MatDet, ExpandDims, Comparison
from elem_ops       import ElemAdd, ElemMult, ElemExp, ElemLog, ElemPower, ElemAbs, Maximum, Minimum, Log, Exp, Sqrt
from nonlinearities import SoftReLU, HardReLU, LogSoftMax, TanH, Logistic, InputSoftMax, SoftMax
from losses         import L2, LogMultinomial
from dropout        import Dropout
from regularizers   import L2Norm, L1Norm, Horseshoe, NExp, RunningAvgKLSparsity
from crossval       import CrossValidator
from convolution    import Convolve1d
from indexing       import Take
from stacking       import Hstack, Vstack
from generic_ops    import Blank

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)

