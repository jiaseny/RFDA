from __future__ import division
import cPickle
import itertools as it
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import ceil, floor
import numpy as np
import scipy as sp
import os
import pandas as pd
import seaborn as sns
import sys
import warnings
from collections import Counter
from copy import deepcopy
from itertools import chain, izip
from numpy import random as rand
from operator import itemgetter
from scipy.linalg import det, eig, eigvals, norm, expm, inv, pinv, qr, svd, \
    hadamard
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, diags, identity, \
    issparse
from sklearn.neighbors import KNeighborsClassifier

# warnings.simplefilter("error", RuntimeWarning)
# warnings.simplefilter("ignore", RuntimeWarning)
# warnings.simplefilter('always', DeprecationWarning)

np.set_printoptions(precision=3, suppress=True)
np.core.arrayprint._line_width = 120  # Display column width

_EPS = 1e-6
_INF = 1e6

sns.set_style('whitegrid')


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def assert_eq(a, b, message=""):
    """Check if a and b are equal."""
    assert a == b, "Error: %s != %s ! %s" % (a, b, message)
    return


def assert_le(a, b, message=""):
    """Check if a and b are equal."""
    assert a <= b, "Error: %s > %s ! %s" % (a, b, message)
    return


def assert_ge(a, b, message=""):
    """Check if a and b are equal."""
    assert a >= b, "Error: %s < %s ! %s" % (a, b, message)
    return


def assert_len(l, length, message=""):
    """Check list/array l is of shape shape."""
    assert_eq(len(l), length)
    return


def assert_shape(A, shape, message=""):
    """Check array A is of shape shape."""
    assert_eq(A.shape, shape)
    return


def check_prob_vector(p):
    """
    Check if a vector is a probability vector.

    Args:
        p, array/list.
    """
    assert np.all(p >= 0), p
    assert np.isclose(np.sum(p), 1), p

    return True


def accuracy(pred_labels, true_labels):
    """
    Computes prediction accuracy.

    Args:
    """
    assert len(pred_labels) == len(true_labels)
    num = len(pred_labels)
    num_correct = sum([pred_labels[i] == true_labels[i] for i in xrange(num)])

    acc = num_correct / float(num)

    return acc


# --------------------------------------------------------------------------- #
# Plotting setup
# --------------------------------------------------------------------------- #

font = {'size': 15}
mpl.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
# Do not use Type 3 fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def pckl_write(data, filename):
    with open(filename, 'w') as f:
        cPickle.dump(data, f)

    return


def pckl_read(filename):
    with open(filename, 'r') as f:
        data = cPickle.load(f)

    return data
