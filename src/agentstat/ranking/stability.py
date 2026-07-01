"""Ranking stability — the headline experiment.

Take a leaderboard of configs, bootstrap-resample the benchmark items, re-rank
the configs on each resample, and ask: **how often does the winner change?**

The one quotable number is ``top_flip_prob`` — the fraction of resamples in
which the top-ranked config differs from the point-estimate winner. If that is
high, "config A is the best" is not a claim the data supports. The supporting
metrics (modal ranking, pairwise win probabilities, expected Kendall-tau) are
computed alongside but the flip probability is the lede.

Resampling is **item-level and paired across configs**: each iteration draws a
bootstrap set of items and re-scores every config on that *same* item set, then
re-ranks. This preserves the pairing that makes config comparisons honest, and
requires every config to have been evaluated on a shared item set (an invariant
the harness guarantees).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau

from agentstat.data.schema import EvalResult


@dataclass(frozen=True)
class RankingStability:
    configs: list[str]                      # configs, sorted best->worst by point estimate
    point_scores: dict[str, float]          # mean score per config on full data
    point_ranking: list[str]                # best->worst on the observed data
    top_flip_prob: float                    # THE headline: P(top config != observed winner)
    modal_ranking: list[str]                # most frequent full ordering under resampling
    modal_ranking_prob: float               # how often that ordering appears
    pairwise_win_prob: dict[tuple[str, str], float]  # P(row ranks above col)
    expected_kendall_tau: float             # mean tau of resampled rankings vs point ranking
    n_boot: int

    @property
    def headline(self) -> str:
        winner = self.point_ranking[0]
        return (
            f"Top config '{winner}' changes under resampling "
            f"{self.top_flip_prob:.0%} of the time "
            f"(n_boot={self.n_boot})."
        )


def _score_matrix(results: list[EvalResult]) -> tuple[list[str], list[str], np.ndarray]:
    """Build a (config x item) mean-score matrix from EvalResults.

    Averages over any replicate rows (e.g. multiple seeds) for the same
    (config, item). Requires the set of items to be identical across configs;
    raises otherwise, because unpaired configs break the paired resample.
    """
    configs = sorted({r.config_id for r in results})
    items = sorted({r.item_id for r in results})
    cfg_ix = {c: i for i, c in enumerate(configs)}
    item_ix = {it: j for j, it in enumerate(items)}

    total = np.zeros((len(configs), len(items)))
    count = np.zeros((len(configs), len(items)))
    for r in results:
        i, j = cfg_ix[r.config_id], item_ix[r.item_id]
        total[i, j] += r.score
        count[i, j] += 1

    if np.any(count == 0):
        missing = [
            (configs[i], items[j])
            for i in range(len(configs))
            for j in range(len(items))
            if count[i, j] == 0
        ]
        raise ValueError(
            "ranking stability requires every config to cover the same item set "
            f"(paired resample). Missing {len(missing)} (config, item) cells, "
            f"e.g. {missing[:3]}."
        )
    return configs, items, total / count


def _rank(mean_scores: np.ndarray, configs: list[str]) -> list[str]:
    """Order configs best->worst by score. Ties broken by config name for determinism."""
    order = sorted(range(len(configs)), key=lambda i: (-mean_scores[i], configs[i]))
    return [configs[i] for i in order]


def ranking_stability(
    results: list[EvalResult],
    n_boot: int = 1_000,
    seed: int | None = None,
) -> RankingStability:
    """Resample items, re-rank configs, and quantify how stable the ranking is."""
    configs, _items, mat = _score_matrix(results)  # mat: (n_config, n_item)
    n_config, n_item = mat.shape
    if n_config < 2:
        raise ValueError("need at least 2 configs to assess ranking stability")

    point_scores = {c: float(mat[i].mean()) for i, c in enumerate(configs)}
    point_ranking = _rank(mat.mean(axis=1), configs)
    observed_winner = point_ranking[0]
    point_rank_index = {c: r for r, c in enumerate(point_ranking)}
    point_rank_vec = [point_rank_index[c] for c in configs]

    rng = np.random.default_rng(seed)

    top_changes = 0
    full_rankings: Counter[tuple[str, ...]] = Counter()
    pair_wins = {
        (configs[a], configs[b]): 0
        for a in range(n_config)
        for b in range(n_config)
        if a != b
    }
    tau_sum = 0.0

    for _ in range(n_boot):
        idx = rng.integers(0, n_item, size=n_item)          # paired item resample
        boot_means = mat[:, idx].mean(axis=1)               # re-score every config
        ranking = _rank(boot_means, configs)

        if ranking[0] != observed_winner:
            top_changes += 1
        full_rankings[tuple(ranking)] += 1

        rank_index = {c: r for r, c in enumerate(ranking)}
        for a in range(n_config):
            for b in range(n_config):
                if a != b and rank_index[configs[a]] < rank_index[configs[b]]:
                    pair_wins[(configs[a], configs[b])] += 1

        boot_rank_vec = [rank_index[c] for c in configs]
        tau, _ = kendalltau(point_rank_vec, boot_rank_vec)
        # tau is nan only if a vector is constant, impossible for >=2 distinct ranks.
        tau_sum += 0.0 if np.isnan(tau) else tau

    modal_ranking, modal_count = full_rankings.most_common(1)[0]

    return RankingStability(
        configs=point_ranking,
        point_scores=point_scores,
        point_ranking=point_ranking,
        top_flip_prob=top_changes / n_boot,
        modal_ranking=list(modal_ranking),
        modal_ranking_prob=modal_count / n_boot,
        pairwise_win_prob={k: v / n_boot for k, v in pair_wins.items()},
        expected_kendall_tau=tau_sum / n_boot,
        n_boot=n_boot,
    )
