"""Variance decomposition validated by recovering a KNOWN variance split.

The milestone: simulate binary data on the logit scale where one factor carries
most of the variance, and confirm the decomposition attributes most of it there.
Because the GLMM is variational and stochastic, thresholds are loose but the
*ordering* of components is what we assert.
"""

import numpy as np
import pytest

from agentstat.data.schema import EvalResult
from agentstat.core.variance import decompose_variance, agent_variance


def _simulate_binary(
    n_items,
    n_seeds,
    sd_item,
    sd_seed,
    mu=0.0,
    seed=0,
):
    """Binary EvalResults with logit(p) = mu + item_effect + seed_effect."""
    rng = np.random.default_rng(seed)
    item_eff = rng.normal(0, sd_item, n_items)
    seed_eff = rng.normal(0, sd_seed, n_seeds)
    results = []
    for i in range(n_items):
        for j in range(n_seeds):
            eta = mu + item_eff[i] + seed_eff[j]
            p = 1.0 / (1.0 + np.exp(-eta))
            results.append(
                EvalResult(
                    config_id="A",
                    item_id=f"i{i}",
                    score=float(rng.binomial(1, p)),
                    seed=j,
                )
            )
    return results


def test_recovers_item_dominant_split():
    # Item variance >> seed variance -> item_id must be the top source.
    results = _simulate_binary(n_items=50, n_seeds=20, sd_item=1.2, sd_seed=0.3, seed=1)
    dec = decompose_variance(results, factors=("item_id", "seed"))
    assert dec.top_source() == "item_id"
    item = next(c for c in dec.components if c.factor == "item_id")
    seed = next(c for c in dec.components if c.factor == "seed")
    assert item.variance > seed.variance
    # Recovered SD is in the right ballpark of truth (1.2), but binary-data
    # variance-component estimation is attenuated (biased toward zero) — the
    # CI does not reliably cover the true SD. We assert the magnitude is close,
    # not exact coverage. This attenuation is documented on the module.
    assert 0.8 <= item.sd <= 1.3


def test_recovers_seed_dominant_split():
    # Flip it: seed variance >> item variance -> seed must be the top source.
    results = _simulate_binary(n_items=40, n_seeds=25, sd_item=0.3, sd_seed=1.2, seed=2)
    dec = decompose_variance(results, factors=("item_id", "seed"))
    assert dec.top_source() == "seed"


def test_percentages_sum_to_100():
    results = _simulate_binary(n_items=40, n_seeds=15, sd_item=1.0, sd_seed=0.5, seed=3)
    dec = decompose_variance(results, factors=("item_id", "seed"))
    total_pct = sum(c.pct_of_total for c in dec.components) + dec.residual_pct
    assert total_pct == pytest.approx(100.0, abs=1e-6)


def test_table_includes_residual_row():
    results = _simulate_binary(n_items=30, n_seeds=10, sd_item=1.0, sd_seed=0.5, seed=4)
    dec = decompose_variance(results, factors=("item_id", "seed"))
    table = dec.as_table()
    assert "residual" in set(table["factor"])
    assert len(table) == 3  # item_id, seed, residual


def test_rejects_continuous_scores():
    results = [
        EvalResult(config_id="A", item_id=f"i{k}", score=float(k) / 10, seed=0)
        for k in range(10)
    ]
    with pytest.raises(ValueError, match="binary"):
        decompose_variance(results, factors=("item_id",))


def test_drops_all_none_and_single_level_factors():
    # prompt_variant is all None; config-level factor 'seed' has one level.
    results = _simulate_binary(n_items=30, n_seeds=1, sd_item=1.0, sd_seed=0.0, seed=5)
    dec = decompose_variance(results, factors=("item_id", "seed", "prompt_variant"))
    factors_used = {c.factor for c in dec.components}
    assert "item_id" in factors_used
    assert "seed" not in factors_used          # single level -> dropped
    assert "prompt_variant" not in factors_used  # all None -> dropped
    assert any("seed" in n for n in dec.notes)
    assert any("prompt_variant" in n for n in dec.notes)


def test_all_factors_unusable_raises():
    results = [
        EvalResult(config_id="A", item_id="i0", score=1.0, seed=0),
        EvalResult(config_id="A", item_id="i0", score=0.0, seed=0),
    ]
    # item_id and seed both single-level here.
    with pytest.raises(ValueError, match="no usable factors"):
        decompose_variance(results, factors=("item_id", "seed"))


def test_unknown_factor_raises():
    results = _simulate_binary(n_items=10, n_seeds=5, sd_item=1.0, sd_seed=0.5, seed=6)
    with pytest.raises(ValueError, match="not a field"):
        decompose_variance(results, factors=("nonexistent",))


def test_agent_variance_partitions_trajectory_sources():
    # Agent slice: trajectory-length (n_turns) carries more variance than seed.
    # n_turns and n_tool_calls are categorical random effects here.
    rng = np.random.default_rng(7)
    n_turn_levels = [1, 2, 3, 4, 5, 6]
    turn_eff = {t: e for t, e in zip(n_turn_levels, rng.normal(0, 1.2, len(n_turn_levels)))}
    seed_eff = {s: e for s, e in zip(range(10), rng.normal(0, 0.3, 10))}
    results = []
    for t in n_turn_levels:
        for s in range(10):
            for rep in range(6):
                eta = turn_eff[t] + seed_eff[s]
                p = 1.0 / (1.0 + np.exp(-eta))
                results.append(
                    EvalResult(
                        config_id="agent",
                        item_id=f"item_{rep}",
                        score=float(rng.binomial(1, p)),
                        seed=s,
                        n_turns=t,
                    )
                )
    dec = agent_variance(results, factors=("seed", "n_turns"))
    assert dec.top_source() == "n_turns"
