"""Significance tests validated against known-answer cases."""

import numpy as np
import pytest

from agentstat.core.significance import permutation_test, fdr_correct


def test_permutation_detects_clear_paired_difference():
    rng = np.random.default_rng(0)
    n = 150
    item = rng.normal(0, 1, n)
    a = item + 0.5 + rng.normal(0, 0.3, n)
    b = item + 0.0 + rng.normal(0, 0.3, n)
    p = permutation_test(a, b, n_perm=5000, paired=True, seed=1)
    assert p < 0.01


def test_permutation_no_difference_gives_large_p():
    rng = np.random.default_rng(2)
    n = 150
    shared = rng.normal(0, 1, n)
    a = shared + rng.normal(0, 0.3, n)
    b = shared + rng.normal(0, 0.3, n)
    p = permutation_test(a, b, n_perm=5000, paired=True, seed=3)
    assert p > 0.2


def test_permutation_identical_arms_p_near_one():
    x = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    p = permutation_test(x, x, n_perm=2000, paired=True, seed=0)
    # Zero difference everywhere -> the observed statistic is the most central;
    # p-value should be ~1.
    assert p == pytest.approx(1.0, abs=1e-9)


def test_permutation_unpaired_path():
    rng = np.random.default_rng(4)
    a = rng.normal(0.7, 0.1, 100)
    b = rng.normal(0.5, 0.1, 100)
    p = permutation_test(a, b, n_perm=5000, paired=False, seed=5)
    assert p < 0.01


def test_permutation_paired_length_mismatch():
    with pytest.raises(ValueError):
        permutation_test([1.0, 0.0, 1.0], [1.0, 0.0], paired=True)


# ---- FDR ----

def test_fdr_rejects_small_keeps_large():
    # Three tiny p-values (real), three large (noise).
    pvals = [0.001, 0.002, 0.003, 0.4, 0.6, 0.9]
    res = fdr_correct(pvals, alpha=0.05)
    assert res.reject[:3] == [True, True, True]
    assert res.reject[3:] == [False, False, False]
    assert res.n_significant == 3


def test_fdr_all_null_rejects_none():
    pvals = [0.2, 0.5, 0.7, 0.9, 0.95]
    res = fdr_correct(pvals, alpha=0.05)
    assert res.n_significant == 0


def test_fdr_corrected_pvalues_monotone_and_bounded():
    pvals = [0.01, 0.02, 0.03, 0.5]
    res = fdr_correct(pvals)
    for pc in res.pvals_corrected:
        assert 0.0 <= pc <= 1.0
    # BH-corrected p-values are >= raw p-values.
    for raw, cor in zip(pvals, res.pvals_corrected):
        assert cor >= raw - 1e-12


def test_fdr_rejects_bad_input():
    with pytest.raises(ValueError):
        fdr_correct([])
    with pytest.raises(ValueError):
        fdr_correct([0.1, 1.5])
