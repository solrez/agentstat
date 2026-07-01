"""Ranking stability validated on constructed stable / unstable leaderboards."""

import numpy as np
import pytest

from agentstat.data.schema import EvalResult
from agentstat.ranking.stability import ranking_stability


def _make_results(config_hits: dict[str, int], n_items: int, seed: int):
    """One binary EvalResult per (config, item), with a *controlled realized* pass
    count per config.

    We assert on realized structure, so we set it directly rather than sampling a
    binomial and hoping the nominal rate survives: each config passes exactly
    ``config_hits[cfg]`` of ``n_items``, at item positions permuted per config so
    the configs aren't trivially nested.
    """
    rng = np.random.default_rng(seed)
    results = []
    for cfg, hits in config_hits.items():
        scores = np.zeros(n_items)
        scores[:hits] = 1.0
        rng.shuffle(scores)
        for j in range(n_items):
            results.append(
                EvalResult(
                    config_id=cfg,
                    item_id=f"item_{j}",
                    score=float(scores[j]),
                    seed=seed,
                )
            )
    return results


def test_dominant_config_is_stable():
    # A clearly beats B, B clearly beats C -> the ordering almost never changes.
    results = _make_results({"A": 170, "B": 120, "C": 70}, n_items=200, seed=0)
    res = ranking_stability(results, n_boot=1000, seed=1)
    assert res.point_ranking[0] == "A"
    assert res.top_flip_prob < 0.05
    assert res.expected_kendall_tau > 0.7


def test_near_tie_is_unstable():
    # A and B tie exactly (same realized pass count) -> the winner flips often.
    # This is the headline case: "A > B" is not supported by the data.
    results = _make_results({"A": 90, "B": 89}, n_items=150, seed=2)
    res = ranking_stability(results, n_boot=1000, seed=3)
    # A 1-item gap on 150 items is deep inside the resampling noise -> flips a lot.
    assert 0.2 < res.top_flip_prob < 0.5


def test_pairwise_win_prob_reflects_gap():
    results = _make_results({"A": 160, "B": 100}, n_items=200, seed=4)
    res = ranking_stability(results, n_boot=1000, seed=5)
    # A ranks above B almost always.
    assert res.pairwise_win_prob[("A", "B")] > 0.95
    # Complementary (no ties possible between two distinct-score configs here).
    assert res.pairwise_win_prob[("B", "A")] < 0.05


def test_modal_ranking_matches_point_when_stable():
    results = _make_results({"A": 180, "B": 120, "C": 60}, n_items=200, seed=6)
    res = ranking_stability(results, n_boot=1000, seed=7)
    assert res.modal_ranking == ["A", "B", "C"]
    assert res.modal_ranking_prob > 0.7


def test_headline_string_mentions_winner_and_prob():
    results = _make_results({"A": 90, "B": 89}, n_items=150, seed=8)
    res = ranking_stability(results, n_boot=500, seed=9)
    assert res.point_ranking[0] in res.headline
    assert "%" in res.headline


def test_averages_over_replicate_seeds():
    # Two seeds per (config, item) -> should average, not error.
    results = []
    for s in (0, 1):
        results += _make_results({"A": 35, "B": 25}, n_items=50, seed=s)
    res = ranking_stability(results, n_boot=200, seed=0)
    assert set(res.configs) == {"A", "B"}


def test_missing_cells_raise():
    results = [
        EvalResult(config_id="A", item_id="i1", score=1.0),
        EvalResult(config_id="A", item_id="i2", score=0.0),
        EvalResult(config_id="B", item_id="i1", score=1.0),
        # B is missing i2 -> unpaired -> must raise.
    ]
    with pytest.raises(ValueError, match="same item set"):
        ranking_stability(results)


def test_single_config_raises():
    results = [
        EvalResult(config_id="A", item_id="i1", score=1.0),
        EvalResult(config_id="A", item_id="i2", score=0.0),
    ]
    with pytest.raises(ValueError, match="at least 2 configs"):
        ranking_stability(results)
