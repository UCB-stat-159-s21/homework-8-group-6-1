"""
Utilities and helper functions.

These functions can be used to construct 2-sided 1 - alpha confidence
bounds for the average treatment effect in a randomized experiment with binary
outcomes and two treatments.
"""

from itertools import combinations
from math import comb, floor
import numpy as np


def nchoosem(n, m):
    """
    Exact re-randomization matrix for small n choose m.

    Parameters
    ----------
    n: int
        total number of subjects
    m: int
        number of subjects with treatment
    """
    c = comb(n, m)
    trt = combinations(np.arange(1, n+1), m)
    Z = [[None]*n for i in np.arange(c)]
    for i in np.arange(c):
        co = next(trt)
        Z[i] = [1 if j in co else 0 for j in np.arange(1, n+1)]
    return Z
