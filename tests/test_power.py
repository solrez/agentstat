"""Power analysis validated against known textbook answers."""

import pytest

import numpy as np

from agentstat.core.power import (
    required_n,
    achieved_power,
    required_n_proportions,
    pairwise_power,
)


def test_required_n_matches_textbook_cohens_d():
    # Cohen's d = 0.5 (medium effect), alpha=.05, power=.8, two-sided:
    # the classic answer is ~64 per group.
    n = required_n(effect_size=0.5, variance=1.0, alpha=0.05, power=0.8)
    assert 62 <= n <= 66


def test_required_n_large_effect_needs_fewer():
    small_effect = required_n(effect_size=0.2, variance=1.0)
    large_effect = required_n(effect_size=0.8, variance=1.0)
    assert large_effect < small_effect


def test_required_n_more_variance_needs_more():
    low_var = required_n(effect_size=0.5, variance=1.0)
    high_var = required_n(effect_size=0.5, variance=4.0)
    assert high_var > low_var


def test_achieved_power_round_trips_with_required_n():
    # If required_n says n suffices for 80% power, achieved_power at that n
    # should be >= 0.8.
    n = required_n(effect_size=0.5, variance=1.0, power=0.8)
    power = achieved_power(n, effect_size=0.5, variance=1.0)
    assert power >= 0.8


def test_achieved_power_increases_with_n():
    p_small = achieved_power(20, effect_size=0.5, variance=1.0)
    p_large = achieved_power(200, effect_size=0.5, variance=1.0)
    assert p_large > p_small
    assert 0.0 <= p_small <= 1.0 and 0.0 <= p_large <= 1.0


def test_required_n_proportions_sane():
    # Distinguishing 70% vs 60% pass rate needs a few hundred per group.
    n = required_n_proportions(0.7, 0.6, alpha=0.05, power=0.8)
    assert 300 <= n <= 450


def test_required_n_proportions_bigger_gap_needs_fewer():
    close = required_n_proportions(0.55, 0.50)
    far = required_n_proportions(0.90, 0.50)
    assert far < close


def test_power_rejects_bad_input():
    with pytest.raises(ValueError):
        required_n(effect_size=0.0, variance=1.0)
    with pytest.raises(ValueError):
        required_n(effect_size=0.5, variance=0.0)
    with pytest.raises(ValueError):
        achieved_power(1, effect_size=0.5, variance=1.0)
    with pytest.raises(ValueError):
        required_n_proportions(0.5, 0.5)


# ---- pairwise (paired) power ----

def test_pairwise_power_big_stable_gap_is_powered():
    # A large, consistent per-item advantage -> high power, small required_n.
    rng = np.random.default_rng(0)
    n = 300
    item = rng.normal(0, 1, n)
    a = item + 1.0 + rng.normal(0, 0.2, n)
    b = item + 0.0 + rng.normal(0, 0.2, n)
    pw = pairwise_power(a, b)
    assert pw.observed_diff > 0
    assert pw.achieved_power > 0.8
    assert not pw.underpowered
    assert pw.required_n < n


def test_pairwise_power_tiny_gap_is_underpowered():
    # A tiny gap swamped by per-item noise -> low power, huge required_n.
    rng = np.random.default_rng(1)
    n = 200
    item = rng.normal(0, 1, n)
    a = item + 0.02 + rng.normal(0, 0.5, n)
    b = item + 0.00 + rng.normal(0, 0.5, n)
    pw = pairwise_power(a, b)
    assert pw.underpowered
    assert pw.required_n > n


def test_pairwise_power_more_items_needed_than_run_when_low_power():
    # Sanity: if achieved power < target, required_n must exceed n_items.
    rng = np.random.default_rng(2)
    n = 100
    a = rng.binomial(1, 0.52, n).astype(float)
    b = rng.binomial(1, 0.50, n).astype(float)
    pw = pairwise_power(a, b)
    if pw.underpowered and pw.observed_diff != 0:
        assert pw.required_n > pw.n_items


def test_pairwise_power_identical_arms():
    x = [1.0, 0.0, 1.0, 1.0, 0.0]
    pw = pairwise_power(x, x)
    assert pw.observed_diff == 0.0
    assert pw.achieved_power == 0.0        # can't power a zero effect
    assert pw.required_n >= 10**9


def test_pairwise_power_length_mismatch():
    with pytest.raises(ValueError):
        pairwise_power([1.0, 0.0, 1.0], [1.0, 0.0])
