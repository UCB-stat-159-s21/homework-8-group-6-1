"""
Test cibin.utils functions.

Unless otherwise noted, expected returns are calculated using the equivalent
functions in R.
"""

import pytest
import numpy as np
from ..utils import *


def test_nchoosem():
    """Test nchoosem returns correct list."""
    n = 5
    m = 3
    Z = nchoosem(n, m)
    expected_Z = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 1, 0],
                           [1, 1, 0, 0, 1], [1, 0, 1, 1, 0],
                           [1, 0, 1, 0, 1], [1, 0, 0, 1, 1],
                           [0, 1, 1, 1, 0], [0, 1, 1, 0, 1],
                           [0, 1, 0, 1, 1], [0, 0, 1, 1, 1]])
    np.testing.assert_array_equal(Z, expected_Z)


def test_combs():
    """Test rows of list have correct number of ones."""
    n = 5
    m = 3
    nperm = 10
    Z = combs(n, m, nperm)
    Z_sum = np.sum(Z, axis=1)
    expected_Z_sum = np.full(nperm, m)
    np.testing.assert_equal(Z_sum, expected_Z_sum)


def test_pval_one_lower():
    """Test pval_one_lower returns correct p-value."""
    n11 = 6
    n10 = 4
    n01 = 4
    n00 = 6
    n = n11+n10+n01+n00
    m = n11+n10
    N01 = 0
    N10 = 0
    N11 = 2
    N = np.array([N11, N10, N01, n-(N11+N10+N01)])
    Z_all = nchoosem(n, m)
    tau_obs = n11/m - n01/(n-m)
    pval = pval_one_lower(n, m, N, Z_all, tau_obs)
    expected_pval = 0.23684210526315788
    np.testing.assert_equal(pval, expected_pval)


def test_pval_two():
    """Test pval_two returns correct p-value."""
    n11 = 6
    n10 = 4
    n01 = 4
    n00 = 6
    n = n11+n10+n01+n00
    m = n11+n10
    N01 = 0
    N10 = 0
    N11 = 2
    N = np.array([N11, N10, N01, n-(N11+N10+N01)])
    Z_all = nchoosem(n, m)
    tau_obs = n11/m - n01/(n-m)
    pval = pval_two(n, m, N, Z_all, tau_obs)
    expected_pval = 0.47368421052631576
    np.testing.assert_equal(pval, expected_pval)


def test_check_compatible():
    """Check check_compatible returns correct list of booleans."""
    n11 = 6
    n10 = 4
    n01 = 4
    n00 = 6
    N11 = np.array([5, 6, 7])
    N10 = np.array([6, 7, 8])
    N01 = np.array([5, 7, 9])
    compatible = check_compatible(n11, n10, n01, n00, N11, N10, N01)
    expected_compatible = np.array([True, True, False])
    np.testing.assert_array_equal(compatible, expected_compatible)


def test_tau_lower_N11_oneside():
    """Test tau_lower_N11_oneside returns correct tau_min and N_accept."""
    n11 = 6
    n10 = 4
    n01 = 4
    n00 = 6
    N11 = 10
    n = n11+n10+n01+n00
    m = n11+n10
    Z_all = nchoosem(n, m)
    alpha = 0.05
    N11_oneside = tau_lower_N11_oneside(n11, n10, n01, n00, N11, Z_all, alpha)
    expected_N11_oneside = (-0.15, np.array([10, 0, 3, 7]))
    np.testing.assert_equal(N11_oneside[0], expected_N11_oneside[0])
    np.testing.assert_array_equal(N11_oneside[1], expected_N11_oneside[1])


def test_tau_lower_oneside():
    """Test tau_lower_oneside returns correct tau_lower and tau_upper."""
    n11 = 1
    n10 = 1
    n01 = 1
    n00 = 13
    alpha = 0.05
    nperm = 1000
    lower_oneside = tau_lower_oneside(n11, n10, n01, n00, alpha, nperm)
    expected_lower_oneside = (-0.0625, 0.875, np.array([1, 0, 1, 14]))
    np.testing.assert_equal(lower_oneside[0], expected_lower_oneside[0])
    np.testing.assert_equal(lower_oneside[1], expected_lower_oneside[1])
    np.testing.assert_array_equal(lower_oneside[2], expected_lower_oneside[2])


def test_tau_lower_N11_twoside():
    """Test tau_lower_N11_twoside returns correct taus and N_accepts."""
    n11 = 6
    n10 = 4
    n01 = 4
    n00 = 6
    n = n11+n10+n01+n00
    m = n11+n10
    N11 = 10
    Z_all = nchoosem(n, m)
    alpha = 0.05
    N11_twoside = tau_lower_N11_twoside(n11, n10, n01, n00, N11, Z_all, alpha)
    expected_N11_twoside = (-0.2, 0.2, np.array([10, 0, 4, 6]),
                            np.array([10, 4, 0, 6]), 11)
    np.testing.assert_equal(N11_twoside[0], expected_N11_twoside[0])
    np.testing.assert_array_equal(N11_twoside[1], expected_N11_twoside[1])
    np.testing.assert_equal(N11_twoside[2], expected_N11_twoside[2])
    np.testing.assert_array_equal(N11_twoside[3], expected_N11_twoside[3])


def test_tau_twoside_lower():
    """Test tau_twoside_lower returns correct taus and N_accepts."""
    n11 = 1
    n10 = 1
    n01 = 1
    n00 = 13
    n = n11+n10+n01+n00
    m = n11+n10
    alpha = 0.05
    Z_all = nchoosem(n, m)
    exact = True
    reps = 1
    twoside_lower_exact = tau_twoside_lower(n11, n10, n01, n00, alpha, Z_all,
                                            exact, reps)
    expected_twoside_lower_exact = (-0.0625, np.array([1, 0, 1, 14]), 0.375,
                                    np.array([0, 7, 1, 8]), 48)
    exact = False
    reps = 20
    twoside_lower_notexact = tau_twoside_lower(n11, n10, n01, n00, alpha,
                                               Z_all, exact, reps)
    expected_twoside_lower_notexact = (-0.0625, np.array([1, 0, 1, 14]), 0.375,
                                       np.array([0, 7, 1, 8]), 33)
    np.testing.assert_equal(twoside_lower_exact[0],
                            expected_twoside_lower_exact[0])
    np.testing.assert_array_equal(twoside_lower_exact[1],
                                  expected_twoside_lower_exact[1])
    np.testing.assert_equal(twoside_lower_exact[2],
                            expected_twoside_lower_exact[2])
    np.testing.assert_array_equal(twoside_lower_exact[3],
                                  expected_twoside_lower_exact[3])
    np.testing.assert_equal(twoside_lower_notexact[0],
                            expected_twoside_lower_notexact[0])
    np.testing.assert_array_equal(twoside_lower_notexact[1],
                                  expected_twoside_lower_notexact[1])
    np.testing.assert_equal(twoside_lower_notexact[2],
                            expected_twoside_lower_notexact[2])
    np.testing.assert_array_equal(twoside_lower_notexact[3],
                                  expected_twoside_lower_notexact[3])


def test_tau_twoside_less_treated():
    """Test tau_twoside_less_treated returns correct taus and N_accepts."""
    n11 = 1
    n10 = 1
    n01 = 1
    n00 = 13
    alpha = 0.05
    exact = True
    max_combinations = 120
    reps = 1
    twoside_less_treated_exact = tau_twoside_less_treated(n11, n10, n01, n00,
                                                          alpha, exact,
                                                          max_combinations,
                                                          reps)
    expected_twoside_less_treated_exact = (-0.0625, 0.875,
                                           np.array([1, 0, 1, 14]),
                                           np.array([1, 14, 0, 1]), 103)
    exact = False
    reps = 20
    twoside_less_treated_notexact = tau_twoside_less_treated(n11, n10, n01,
                                                             n00, alpha, exact,
                                                             max_combinations,
                                                             reps)
    expected_twoside_less_treated_notexact = (-0.0625, 0.875,
                                              np.array([1, 0, 1, 14]),
                                              np.array([1, 14, 0, 1]), 60)
    np.testing.assert_equal(twoside_less_treated_exact[0],
                            expected_twoside_less_treated_exact[0])
    np.testing.assert_equal(twoside_less_treated_exact[1],
                            expected_twoside_less_treated_exact[1])
    np.testing.assert_array_equal(twoside_less_treated_exact[2],
                                  expected_twoside_less_treated_exact[2])
    np.testing.assert_array_equal(twoside_less_treated_exact[3],
                                  expected_twoside_less_treated_exact[3])
    np.testing.assert_equal(twoside_less_treated_notexact[0],
                            expected_twoside_less_treated_notexact[0])
    np.testing.assert_equal(twoside_less_treated_notexact[1],
                            expected_twoside_less_treated_notexact[1])
    np.testing.assert_array_equal(twoside_less_treated_notexact[2],
                                  expected_twoside_less_treated_notexact[2])
    np.testing.assert_array_equal(twoside_less_treated_notexact[3],
                                  expected_twoside_less_treated_notexact[3])
    with pytest.raises(Exception):
        tau_twoside_less_treated(n11, n10, n01, n00, alpha, True, 100, reps)


def test_tau_twosided_ci():
    """Test tau_twosided_ci returns correct taus and N_accepts."""
    exact1 = tau_twosided_ci(1, 1, 1, 13, 0.05, True, 120, 1)
    exact2 = tau_twosided_ci(2, 6, 8, 0, 0.05, True, 12870, 1)
    exact3 = tau_twosided_ci(6, 0, 11, 3, 0.05, True, 38760, 1)
    exact4 = tau_twosided_ci(6, 4, 4, 6, 0.05, True, 184756, 1)
    exact5 = tau_twosided_ci(1, 1, 3, 19, 0.05, True, 276, 1)
    expected_exact1 = ([-1, 14], [[1, 0, 1, 14], [1, 14, 0, 1]], [120, 103])
    expected_exact2 = ([-14, -5], [[2, 0, 14, 0], [8, 0, 5, 3]], [12870, 113])
    expected_exact3 = ([-4, 8], [[13, 0, 4, 3], [11, 8, 0, 1]], [38760, 283])
    expected_exact4 = ([-4, 10], [[5, 1, 5, 9], [5, 11, 1, 3]], [184756, 308])
    expected_exact5 = ([-3, 20], [[1, 0, 3, 20], [3, 20, 0, 1]], [276, 251])
    assert exact1 == expected_exact1
    assert exact2 == expected_exact2
    assert exact3 == expected_exact3
    assert exact4 == expected_exact4
    assert exact5 == expected_exact5
    notexact1 = tau_twosided_ci(1, 1, 1, 13, 0.05, False, 1, 5000)
    notexact2 = tau_twosided_ci(2, 6, 8, 0, 0.05, False, 1, 5000)
    notexact3 = tau_twosided_ci(6, 0, 11, 3, 0.05, False, 1, 5000)
    notexact4 = tau_twosided_ci(6, 4, 4, 6, 0.05, False, 1, 5000)
    notexact5 = tau_twosided_ci(1, 1, 3, 19, 0.05, False, 1, 5000)
    assert exact1[:2] == notexact1[:2]
    assert exact2[:2] == notexact2[:2]
    assert exact3[:2] == notexact3[:2]
    assert exact4[:2] == notexact4[:2]
    assert exact5[:2] == notexact5[:2]
    with pytest.raises(Exception):
        tau_twosided_ci(2, 6, 8, 0, 0.05, True, 100, 1)


def test_ind():
    """Test that ind returns correct boolean."""
    assert ind(5, 4, 6)
    assert not ind(4, 5, 6)


def test_lci():
    """Test that lci returns correct lower bounds."""
    N = 50
    n = 10
    xx = np.arange(n+1)
    alpha = 0.05
    lcis = lci(xx, n, N, alpha)
    expected_lcis = np.array([0, 1, 3, 6, 9, 13, 17, 21, 27, 32, 39])
    np.testing.assert_array_equal(lcis, expected_lcis)


def test_uci():
    """Test that uci returns correct upper bounds."""
    N = 50
    n = 10
    xx = np.arange(n+1)
    alpha = 0.05
    ucis = uci(xx, n, N, alpha)
    expected_ucis = np.array([11, 18, 23, 29, 33, 37, 41, 44, 47, 49, 50])
    np.testing.assert_array_equal(ucis, expected_ucis)


def test_exact_CI_odd():
    """Test exact_CI_odd returns correct CI."""
    N = 50
    n = 15
    x = 10
    alpha = 0.05
    CI_odd = exact_CI_odd(N, n, x, alpha)
    expected_CI_odd = (23, 41)
    assert CI_odd == expected_CI_odd


def test_exact_CI_even():
    """Test exact_CI_odd returns correct CI."""
    N = 50
    n = 14
    x = 10
    alpha = 0.05
    CI_even = exact_CI_even(N, n, x, alpha)
    expected_CI_even = (24, 43)
    assert CI_even == expected_CI_even


def test_exact_CI():
    """Test exact_CI returns correct CI."""
    N = 50
    n = 15
    x = 10
    alpha = 0.05
    CI_odd = exact_CI(N, n, x, alpha)
    expected_CI_odd = (23, 41)
    assert CI_odd == expected_CI_odd
    n = 14
    CI_even = exact_CI(N, n, x, alpha)
    expected_CI_even = (24, 43)
    assert CI_even == expected_CI_even


def test_combin_exact_CI():
    """Test that combin_exact_CI returns correct CI."""
    n11 = 6
    n10 = 4
    n01 = 4
    n00 = 6
    alpha = .05
    exact_CI = combin_exact_CI(n11, n10, n01, n00, alpha)
    expected_exact_CI = (-0.3, 0.6)
    assert exact_CI == expected_exact_CI


def test_N_plus1_exact_CI():
    """Test that N_plus1_exact_CI returns correct CI."""
    n11 = 6
    n10 = 4
    n01 = 4
    n00 = 6
    alpha = .05
    Nplus1_exact_CI = N_plus1_exact_CI(n11, n10, n01, n00, alpha)
    expected_Nplus1_exact_CI = (-0.3, 0.55)
    assert Nplus1_exact_CI == expected_Nplus1_exact_CI
