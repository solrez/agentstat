"""Variance decomposition for eval scores — where does the noise come from?

For **binary** pass/fail scores (the common eval case), a plain Gaussian ANOVA
is the wrong tool: its variance-component assumptions don't hold on 0/1 data. We
fit a **logistic mixed model** (crossed random effects) via statsmodels'
``BinomialBayesMixedGLM`` and partition variance on the logit (linear-predictor)
scale, adding the logistic residual variance (pi^2 / 3).

A spike (see repo history) confirmed this recovers a known variance split:
truth sd_item=1.0/sd_seed=0.5 -> recovered ~0.95 / ~0.58.

**Attenuation caveat.** Variance-component estimation on *binary* data is biased
toward zero: recovered SDs tend to *understate* the true SD, and for large
components the CI may not cover the truth. Binary outcomes simply carry less
information than continuous ones, and the variational fit shrinks. So read these
results as **relative** — which source dominates, and by roughly how much — not
as exact absolute variances. The ordering of components is reliable; the absolute
magnitudes are conservative.

``agent_variance`` is the differentiating slice: it partitions variance across
the *agent-specific* sources — seed, trajectory length, tool-call count — that
single-turn variance work doesn't isolate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from agentstat.data.schema import EvalResult

# Residual variance of a standard logistic distribution — the irreducible
# Bernoulli noise on the logit scale.
_LOGISTIC_RESIDUAL_VAR = np.pi**2 / 3.0


@dataclass(frozen=True)
class VarianceComponent:
    factor: str
    sd: float           # random-effect SD on the logit scale
    sd_low: float
    sd_high: float
    variance: float     # sd**2
    pct_of_total: float  # share of total variance (incl. logistic residual)


@dataclass(frozen=True)
class VarianceDecomposition:
    components: list[VarianceComponent]
    residual_variance: float
    residual_pct: float
    total_variance: float
    n_obs: int
    scale: str = "logit"
    notes: list[str] = field(default_factory=list)

    def as_table(self) -> pd.DataFrame:
        rows = [
            {
                "factor": c.factor,
                "sd_logit": c.sd,
                "variance": c.variance,
                "pct_of_total": c.pct_of_total,
            }
            for c in self.components
        ]
        rows.append(
            {
                "factor": "residual",
                "sd_logit": np.sqrt(self.residual_variance),
                "variance": self.residual_variance,
                "pct_of_total": self.residual_pct,
            }
        )
        return pd.DataFrame(rows)

    def top_source(self) -> str:
        """The factor (excluding residual) explaining the most variance."""
        if not self.components:
            return "residual"
        return max(self.components, key=lambda c: c.variance).factor


def _is_binary(scores: np.ndarray) -> bool:
    return np.all(np.isin(np.unique(scores), (0.0, 1.0)))


def decompose_variance(
    results: list[EvalResult],
    factors: tuple[str, ...] = ("prompt_variant", "seed", "item_id"),
    fit_method: str = "vb",
) -> VarianceDecomposition:
    """Partition binary-score variance across ``factors`` (crossed random effects).

    Each factor must be a field on ``EvalResult``. Factors that are entirely
    ``None`` or have only a single level carry no variance and are dropped with
    a note (a one-level factor is not identifiable as a random effect).

    Returns a variance-component table on the logit scale, including the
    logistic residual.
    """
    if len(results) < 2:
        raise ValueError("need at least 2 results to decompose variance")

    df = pd.DataFrame(r.to_dict() for r in results)
    y = df["score"].to_numpy(dtype=float)
    if not _is_binary(y):
        raise ValueError(
            "decompose_variance targets binary (0/1) scores; got continuous data. "
            "Use a Gaussian variance-components model for continuous scores."
        )

    notes: list[str] = []
    usable: list[str] = []
    for f in factors:
        if f not in df.columns:
            raise ValueError(f"factor {f!r} is not a field of EvalResult")
        levels = df[f].dropna().nunique()
        if df[f].isna().all():
            notes.append(f"dropped {f!r}: all values None")
        elif levels < 2:
            notes.append(f"dropped {f!r}: only {levels} level(s), no variance to attribute")
        else:
            usable.append(f)

    if not usable:
        raise ValueError(
            "no usable factors: every requested factor is constant or all-None"
        )

    # Cast factor columns to strings so statsmodels treats them as categorical
    # levels regardless of numeric dtype (seed is an int).
    model_df = pd.DataFrame({"y": df["score"].astype(float)})
    vc = {}
    for f in usable:
        col = f"f_{f}"
        model_df[col] = df[f].astype("string").fillna("__missing__")
        vc[f] = f"0 + C({col})"

    fitted = BinomialBayesMixedGLM.from_formula("y ~ 1", vc, model_df)
    result = fitted.fit_vb() if fit_method == "vb" else fitted.fit_map()

    # vcp_mean/vcp_sd carry ONE entry per component, in `vc` insertion order
    # (confirmed by spike), on the log-SD scale.
    sds = np.exp(result.vcp_mean)
    sd_low = np.exp(result.vcp_mean - 1.96 * result.vcp_sd)
    sd_high = np.exp(result.vcp_mean + 1.96 * result.vcp_sd)
    variances = sds**2

    total = float(variances.sum() + _LOGISTIC_RESIDUAL_VAR)
    components = [
        VarianceComponent(
            factor=f,
            sd=float(sds[k]),
            sd_low=float(sd_low[k]),
            sd_high=float(sd_high[k]),
            variance=float(variances[k]),
            pct_of_total=float(100.0 * variances[k] / total),
        )
        for k, f in enumerate(usable)
    ]

    return VarianceDecomposition(
        components=components,
        residual_variance=float(_LOGISTIC_RESIDUAL_VAR),
        residual_pct=float(100.0 * _LOGISTIC_RESIDUAL_VAR / total),
        total_variance=total,
        n_obs=len(df),
        notes=notes,
    )


def agent_variance(
    results: list[EvalResult],
    factors: tuple[str, ...] = ("seed", "n_turns", "n_tool_calls"),
) -> VarianceDecomposition:
    """THE NOVEL SLICE: partition variance across agent-specific noise sources.

    For multi-turn agents, output variance is not just seed nondeterminism: it
    also flows from trajectory length (``n_turns``) and tool-call count
    (``n_tool_calls``). This isolates those components so you can ask whether,
    e.g., trajectory-length nondeterminism dominates seed variance — which would
    mean single-turn variance estimates *understate* agent eval noise.

    Same logistic-mixed-model machinery as ``decompose_variance``; the contribution
    is treating these agent factors as the crossed random effects.
    """
    return decompose_variance(results, factors=factors)
