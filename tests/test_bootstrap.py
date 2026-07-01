"""Bootstrap CIs validated against synthetic ground truth and against scipy itself.

Two kinds of check:
  1. Statistical: on data with a known mean, the CI covers it and brackets the point.
  2. Correctness: our BCa wrapper must reproduce scipy.stats.bootstrap exactly
     (same seed) — scipy is the reference; we only adapt inputs/outputs.
"""

import numpy as np
import pytest
from scipy import stats

from agentstat.core.bootstrap import bootstrap_ci, paired_bootstrap_diff


def test_ci_brackets_point_and_covers_true_mean():
    rng = np.random.default_rng(0)
    # Continuous scores, true mean 5.0.
    scores = rng.normal(5.0, 1.0, size=300)
    res = bootstrap_ci(scores, n_boot=2000, method="bca", seed=1)
    assert res.low < res.point < res.high
    assert res.low < 5.0 < res.high
    assert res.confidence == 0.95


def test_wrapper_matches_scipy_bca_exactly():
    # The correctness guard: same seed -> identical bounds to scipy's own BCa.
    rng = np.random.default_rng(0)
    scores = rng.normal(0.6, 0.2, size=150)

    ours = bootstrap_ci(scores, n_boot=5000, method="bca", alpha=0.05, seed=42)

    ref = stats.bootstrap(
        (scores,),
        np.mean,
        n_resamples=5000,
        method="bca",
        confidence_level=0.95,
        random_state=np.random.default_rng(42),
        vectorized=False,
    ).confidence_interval

    assert ours.low == pytest.approx(ref.low, rel=1e-12)
    assert ours.high == pytest.approx(ref.high, rel=1e-12)


@pytest.mark.parametrize("method", ["bca", "percentile", "basic"])
def test_methods_all_bracket_point(method):
    rng = np.random.default_rng(3)
    scores = rng.binomial(1, 0.7, size=200).astype(float)  # binary pass/fail
    res = bootstrap_ci(scores, n_boot=2000, method=method, seed=7)
    assert 0.0 <= res.low <= res.point <= res.high <= 1.0


def test_coverage_is_approximately_nominal():
    # Frequentist coverage check: a 95% CI should cover the truth ~95% of the time.
    rng = np.random.default_rng(123)
    true_p = 0.5
    covered = 0
    trials = 200
    for i in range(trials):
        scores = rng.binomial(1, true_p, size=120).astype(float)
        res = bootstrap_ci(scores, n_boot=1000, method="bca", seed=int(i))
        if res.low <= true_p <= res.high:
            covered += 1
    # Loose bound — this is a stochastic check, not exact.
    assert 0.88 <= covered / trials <= 0.99


def test_degenerate_all_equal_scores():
    res = bootstrap_ci([1.0, 1.0, 1.0, 1.0], method="bca")
    assert res.point == res.low == res.high == 1.0


def test_bootstrap_ci_rejects_bad_input():
    with pytest.raises(ValueError):
        bootstrap_ci([1.0])  # need >= 2
    with pytest.raises(ValueError):
        bootstrap_ci(np.zeros((2, 2)))  # not 1-D


def test_as_tuple_signature():
    res = bootstrap_ci(np.linspace(0, 1, 50), seed=0)
    point, low, high = res.as_tuple()
    assert low <= point <= high


# ---- paired bootstrap difference ----

def test_paired_diff_detects_real_gap():
    rng = np.random.default_rng(5)
    n = 200
    # Config a is genuinely better on the same items: shared item effect + a bump.
    item_effect = rng.normal(0, 1, n)
    a = item_effect + 0.5 + rng.normal(0, 0.3, n)
    b = item_effect + 0.0 + rng.normal(0, 0.3, n)
    res = paired_bootstrap_diff(a, b, n_boot=5000, seed=1)
    assert res.diff > 0
    assert res.low > 0            # CI excludes zero -> real difference
    assert res.prob_a_gt_b > 0.95
    assert res.paired


def test_paired_diff_finds_no_gap_when_none():
    rng = np.random.default_rng(9)
    n = 200
    shared = rng.normal(0, 1, n)
    a = shared + rng.normal(0, 0.3, n)
    b = shared + rng.normal(0, 0.3, n)
    res = paired_bootstrap_diff(a, b, n_boot=5000, seed=2)
    assert res.low < 0 < res.high        # CI includes zero
    assert 0.3 < res.prob_a_gt_b < 0.7   # near coin-flip


def test_paired_diff_requires_equal_length():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0, 0.0, 1.0], [1.0, 0.0])


def test_paired_diff_ties_give_half_credit():
    # Identical arms -> every bootstrap diff is exactly 0 -> P(a>b) == 0.5.
    x = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    res = paired_bootstrap_diff(x, x, n_boot=2000, seed=0)
    assert res.diff == 0.0
    assert res.prob_a_gt_b == pytest.approx(0.5)
