"""Power analysis: how many trials do you actually need?

Answers the question eval reports usually skip: given the noise and the size of
the difference you care about, how many items must you run to detect it — and,
after the fact, what power did the run you actually did have?

Built on ``statsmodels.stats.power``. Two entry points:
  - variance-based (continuous scores): you supply effect size + variance.
  - proportion-based (binary pass/fail): you supply two pass rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.stats.power import TTestIndPower, TTestPower
from statsmodels.stats.proportion import proportion_effectsize


def _cohens_d(effect_size: float, variance: float) -> float:
    if variance <= 0:
        raise ValueError("variance must be positive")
    return effect_size / np.sqrt(variance)


def required_n(
    effect_size: float,
    variance: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Trials-per-group needed to detect ``effect_size`` (raw mean difference).

    ``effect_size`` is on the score scale (e.g. a 0.05 accuracy gap); ``variance``
    is the per-observation score variance. Returns the ceiling of the exact
    solution, so achieved power at the returned n is >= the target.
    """
    if effect_size == 0:
        raise ValueError("effect_size must be non-zero to size a study")
    d = abs(_cohens_d(effect_size, variance))
    n = TTestIndPower().solve_power(
        effect_size=d, alpha=alpha, power=power, alternative="two-sided"
    )
    return int(np.ceil(n))


def achieved_power(
    n: int,
    effect_size: float,
    variance: float,
    alpha: float = 0.05,
) -> float:
    """Power of a run with ``n`` per group at the given effect size and variance."""
    if n < 2:
        raise ValueError("n must be at least 2")
    d = abs(_cohens_d(effect_size, variance))
    return float(
        TTestIndPower().power(
            effect_size=d, nobs1=n, alpha=alpha, alternative="two-sided"
        )
    )


def required_n_proportions(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Trials-per-group to distinguish two pass rates ``p1`` vs ``p2`` (binary evals).

    Uses the arcsine-transformed (Cohen's h) effect size, which is the standard
    choice for a difference in proportions.
    """
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
        raise ValueError("p1 and p2 must be in [0, 1]")
    if p1 == p2:
        raise ValueError("p1 and p2 must differ to size a study")
    h = abs(proportion_effectsize(p1, p2))
    n = TTestIndPower().solve_power(
        effect_size=h, alpha=alpha, power=power, alternative="two-sided"
    )
    return int(np.ceil(n))


@dataclass(frozen=True)
class PairwisePower:
    """Power analysis for one paired config-vs-config comparison."""

    n_items: int          # items actually run (per config)
    observed_diff: float  # mean(a) - mean(b) on the shared items
    achieved_power: float  # power the run actually had for this effect
    required_n: int       # items needed for the target power
    target_power: float

    @property
    def underpowered(self) -> bool:
        return self.achieved_power < self.target_power


def pairwise_power(
    a,
    b,
    alpha: float = 0.05,
    power: float = 0.8,
) -> PairwisePower:
    """Achieved power and required-n for a **paired** config comparison.

    ``a`` and ``b`` are per-item scores for two configs on the SAME items (paired
    by position). This answers the practical question a ranking raises: *given the
    gap we observed and its variance, did we run enough items to trust the call,
    and if not, how many would we need?*

    We size on the paired difference ``d = a - b`` using a one-sample t-power on
    ``d`` (mean vs 0). This is the correct paired framing — it uses the variance
    of the per-item differences, which is smaller than the between-config variance
    because the shared item difficulty cancels. So the ``required_n`` here is
    generally *lower* (less conservative) than treating a and b as two independent
    proportion samples.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("pairwise_power needs equal-length paired arrays")
    if a.size < 2:
        raise ValueError("need at least 2 paired observations")

    d = a - b
    n = d.size
    sd = d.std(ddof=1)
    observed_diff = float(d.mean())

    if sd == 0:
        # No variance in the differences: the gap (if any) is detectable at n=2,
        # or there is no gap at all.
        eff_power = 1.0 if observed_diff != 0 else 0.0
        req = 2 if observed_diff != 0 else 10**9
        return PairwisePower(n, observed_diff, eff_power, req, power)

    effect = abs(observed_diff) / sd  # standardized paired effect (Cohen's dz)
    achieved = float(TTestPower().power(effect_size=effect, nobs=n, alpha=alpha,
                                        alternative="two-sided"))
    if observed_diff == 0:
        required = 10**9  # can't power a zero effect
    else:
        required = int(np.ceil(
            TTestPower().solve_power(effect_size=effect, alpha=alpha, power=power,
                                     alternative="two-sided")
        ))
    return PairwisePower(n, observed_diff, achieved, required, power)
