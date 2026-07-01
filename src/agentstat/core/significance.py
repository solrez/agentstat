"""Significance testing for config comparisons.

Two questions this module answers:
  1. Is the difference between two configs real, or noise?  -> permutation test
     (and the paired-bootstrap path in ``bootstrap.py``).
  2. When comparing many configs at once, which differences survive multiple-
     comparison correction?  -> Benjamini-Hochberg FDR.

We lean on scipy / statsmodels rather than hand-rolling either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def permutation_test(
    a,
    b,
    n_perm: int = 10_000,
    paired: bool = True,
    seed: int | None = None,
) -> float:
    """Two-sided permutation test for a difference in means; returns a p-value.

    Paired (default): tests each item's (a-b) difference by randomly flipping
    the sign of each paired difference — the correct null when a and b are the
    same items under two configs. Unpaired: shuffles group labels across the
    pooled sample.

    Uses ``scipy.stats.permutation_test`` so the null construction and p-value
    (including the +1 finite-sample correction) are handled correctly.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)

    if paired:
        if a.shape != b.shape:
            raise ValueError("paired permutation test needs equal-length arrays")

        def stat(x, y):
            return np.mean(x - y)

        res = stats.permutation_test(
            (a, b),
            stat,
            permutation_type="samples",  # sign-flip paired differences
            n_resamples=n_perm,
            alternative="two-sided",
            random_state=rng,
            vectorized=False,
        )
    else:
        def stat(x, y):
            return np.mean(x) - np.mean(y)

        res = stats.permutation_test(
            (a, b),
            stat,
            permutation_type="independent",  # shuffle group labels
            n_resamples=n_perm,
            alternative="two-sided",
            random_state=rng,
            vectorized=False,
        )
    return float(res.pvalue)


@dataclass(frozen=True)
class FDRResult:
    """Result of Benjamini-Hochberg FDR correction over a family of p-values."""

    reject: list[bool]        # which hypotheses survive at the FDR level
    pvals_corrected: list[float]
    alpha: float

    @property
    def n_significant(self) -> int:
        return int(sum(self.reject))


def fdr_correct(pvalues, alpha: float = 0.05) -> FDRResult:
    """Benjamini-Hochberg FDR correction: which comparisons survive.

    When you compare K configs pairwise you run many tests; without correction
    ~alpha of them look significant by chance. BH controls the expected fraction
    of false discoveries among the rejections.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or pvalues.size == 0:
        raise ValueError("pvalues must be a non-empty 1-D array")
    if np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("p-values must be in [0, 1]")

    reject, pvals_corrected, _, _ = multipletests(
        pvalues, alpha=alpha, method="fdr_bh"
    )
    return FDRResult(
        reject=reject.tolist(),
        pvals_corrected=pvals_corrected.tolist(),
        alpha=alpha,
    )
