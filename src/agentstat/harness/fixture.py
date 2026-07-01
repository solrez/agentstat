"""A frozen, semi-synthetic benchmark result set for offline demos.

The experiments and figures should be reproducible without API access or budget,
so the pipeline can be demonstrated end-to-end even if a live benchmark run
slips. This generates a realistic ``list[EvalResult]`` shaped like a BFCL run:
several configs over a shared item set, across seeds, with binary scores.

The ranking is deliberately constructed to be **unstable at the top** — two
strong configs are near-tied — so the headline ranking-instability experiment
has a real finding to surface. This is a stand-in for real data, and every
experiment that consumes it also runs unchanged on real EvalResults.
"""

from __future__ import annotations

import numpy as np

from agentstat.data.schema import EvalResult

# Realized per-config difficulty offsets on the logit scale. The top two ("gpt-ish"
# and "claude-ish") are close on purpose; the rest trail.
_CONFIG_SKILL = {
    "alpha-8b": 0.95,   # top two are near-tied -> unstable winner
    "beta-8b": 0.90,
    "gamma-7b": 0.25,
    "delta-7b": -0.40,
}


def make_frozen_results(
    n_items: int = 200,
    seeds: tuple[int, ...] = (0, 1, 2),
    seed: int = 20240601,
) -> list[EvalResult]:
    """Generate the frozen semi-synthetic result set.

    ``logit(p) = config_skill + item_difficulty + seed_jitter``, so item and seed
    both contribute variance (item dominates, seed is a smaller real source) —
    exactly the structure the variance decomposition is meant to recover.
    """
    rng = np.random.default_rng(seed)
    item_difficulty = rng.normal(0.0, 1.0, n_items)      # dominant variance source
    seed_jitter = {s: rng.normal(0.0, 0.35, 1)[0] for s in seeds}  # smaller source

    results: list[EvalResult] = []
    for config_id, skill in _CONFIG_SKILL.items():
        for i in range(n_items):
            for s in seeds:
                eta = skill + item_difficulty[i] + seed_jitter[s]
                p = 1.0 / (1.0 + np.exp(-eta))
                results.append(
                    EvalResult(
                        config_id=config_id,
                        item_id=f"simple_python_{i}",
                        score=float(rng.binomial(1, p)),
                        seed=s,
                        metadata={"source": "frozen_fixture"},
                    )
                )
    return results
