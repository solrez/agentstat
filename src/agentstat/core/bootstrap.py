"""Bootstrap confidence intervals for eval scores.

Thin, correct wrappers over ``scipy.stats.bootstrap`` (which has BCa built in) so
that every score gets error bars rather than a bare point estimate. We do not
hand-roll BCa: the acceleration and bias-correction terms are subtle and a
subtly-wrong implementation still looks plausible on synthetic data.

The default is bootstrap rather than a CLT/normal-approximation interval: below
~200 samples the CLT interval is disputed for eval scores, and bootstrap makes
no distributional assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CIResult:
    """A point estimate with a confidence interval."""

    point: float
    low: float
    high: float
    method: str
    confidence: float  # e.g. 0.95

    def as_tuple(self) -> tuple[float, float, float]:
        """(point_estimate, lower, upper) — the plan's headline signature."""
        return (self.point, self.low, self.high)


def _rng(seed: int | None) -> np.random.Generator:
    # scipy requires a Generator (or None); we pass one through for reproducibility.
    return np.random.default_rng(seed)


def bootstrap_ci(
    scores,
    n_boot: int = 10_000,
    method: str = "bca",
    alpha: float = 0.05,
    statistic=np.mean,
    seed: int | None = None,
) -> CIResult:
    """Bootstrap CI for a statistic (default: the mean) of ``scores``.

    Parameters
    ----------
    scores : array-like of float
        The per-item scores (0/1 for pass-fail, or continuous).
    method : {"bca", "percentile", "basic"}
        Passed to scipy. "bca" (bias-corrected and accelerated) is the default
        and is generally the most accurate for skewed / bounded score data.
    alpha : float
        Two-sided; the CI has confidence ``1 - alpha``.

    Returns
    -------
    CIResult with ``.point`` = statistic on the observed data, and
    ``.low`` / ``.high`` the CI bounds.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be 1-D")
    if scores.size < 2:
        raise ValueError("need at least 2 scores to bootstrap")

    point = float(statistic(scores))

    # BCa is undefined when the statistic has zero variance across resamples
    # (e.g. every score identical). Degenerate CI = the point itself.
    if np.all(scores == scores[0]):
        return CIResult(point, point, point, method, 1 - alpha)

    res = stats.bootstrap(
        (scores,),
        statistic,
        n_resamples=n_boot,
        method=method,
        confidence_level=1 - alpha,
        random_state=_rng(seed),
        vectorized=False,
    )
    ci = res.confidence_interval
    return CIResult(point, float(ci.low), float(ci.high), method, 1 - alpha)


@dataclass(frozen=True)
class DiffResult:
    """CI on a paired difference a - b, with the probability that a > b."""

    diff: float          # mean(a) - mean(b) on observed data
    low: float
    high: float
    prob_a_gt_b: float   # bootstrap P(mean(a*) > mean(b*))
    confidence: float
    paired: bool


def paired_bootstrap_diff(
    a,
    b,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> DiffResult:
    """CI on the difference ``mean(a) - mean(b)`` plus ``P(a > b)``.

    Inputs are **paired by position**: ``a[i]`` and ``b[i]`` must be the same
    benchmark item scored under two configs. Pairing removes item-difficulty
    variance and gives a much tighter, more honest test than treating the two
    as independent samples — so the caller is responsible for guaranteeing the
    same item set in the same order (the harness enforces this as an invariant).

    We resample **item indices once per iteration** and apply the same indices
    to both arms, which is what makes it a paired bootstrap.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"paired bootstrap needs equal-length paired arrays; got {a.shape} vs {b.shape}"
        )
    if a.ndim != 1:
        raise ValueError("a and b must be 1-D")
    if a.size < 2:
        raise ValueError("need at least 2 paired observations")

    n = a.size
    rng = _rng(seed)
    observed = float(a.mean() - b.mean())

    # Resample the same item indices for both arms (paired).
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_a = a[idx].mean(axis=1)
    boot_b = b[idx].mean(axis=1)
    boot_diff = boot_a - boot_b

    low, high = np.quantile(boot_diff, [alpha / 2, 1 - alpha / 2])
    # Half-credit ties so a==b contributes 0.5, not a spurious 0 or 1.
    prob = float(np.mean(boot_diff > 0) + 0.5 * np.mean(boot_diff == 0))

    return DiffResult(
        diff=observed,
        low=float(low),
        high=float(high),
        prob_a_gt_b=prob,
        confidence=1 - alpha,
        paired=True,
    )
