"""Variance decomposition on the BFCL results: where does eval noise come from?

Partitions binary-score variance across item difficulty and seed nondeterminism.
On single-turn BFCL, item variance is expected to dominate — the honest baseline
that the agent slice extends. For a true multi-turn agent run (with n_turns /
n_tool_calls populated), agent_variance() partitions those additional sources.

    uv run python experiments/02_agent_variance.py
"""

from pathlib import Path

from agentstat.harness.runner import load_results
from agentstat.harness.fixture import make_frozen_results
from agentstat.core.variance import decompose_variance

RESULTS = Path("results/bfcl_simple.jsonl")
FIG_DIR = Path("figures")


def load():
    if RESULTS.exists():
        return load_results(RESULTS), "BFCL simple (DeepInfra)"
    return make_frozen_results(), "frozen fixture"


def main():
    results, source = load()

    dec = decompose_variance(results, factors=("item_id", "seed"))
    print(f"=== Variance decomposition ({source}) ===")
    print(f"  scale: {dec.scale}  |  n_obs: {dec.n_obs}\n")
    print(dec.as_table().to_string(index=False))
    for note in dec.notes:
        print(f"  note: {note}")
    print(f"\n  Dominant source: {dec.top_source()}")
    print("\n  Reading: variance-component estimates on binary data are attenuated")
    print("  (biased toward zero) — read these as RELATIVE shares, not exact values.")

    _plot(dec, source)


def _plot(dec, source):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping figure.)")
        return

    FIG_DIR.mkdir(exist_ok=True)
    labels = [c.factor for c in dec.components] + ["residual"]
    pcts = [c.pct_of_total for c in dec.components] + [dec.residual_pct]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, pcts, color=["#2b6cb0", "#dd6b20", "#a0aec0"][:len(labels)])
    ax.set_ylabel("% of total variance (logit scale)")
    ax.set_title(f"Where BFCL eval variance comes from\n({source})", fontsize=11)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.0f}%", ha="center", fontsize=10)
    ax.set_ylim(0, max(pcts) * 1.2)
    fig.tight_layout()
    out = FIG_DIR / "02_variance_components.png"
    fig.savefig(out, dpi=140)
    print(f"\n  Figure written: {out}")


if __name__ == "__main__":
    main()
