"""Power analysis: how many trials do you actually need?

Answers the question eval reports usually skip: given the noise and the size of
the difference you care about, how many items must you run to detect it — and,
after the fact, what power did the run you actually did have?

Built on ``statsmodels.stats.power``. Two entry points:
  - variance-based (continuous scores): you supply effect size + variance.
  - proportion-based (binary pass/fail): you supply two pass rates.
"""

from __future__ import annotations

import numpy as np
from statsmodels.stats.power import TTestIndPower
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
