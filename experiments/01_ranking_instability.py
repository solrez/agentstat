"""THE headline experiment: does the BFCL leaderboard ranking survive resampling?

Loads the cached BFCL results, computes the point-estimate ranking with bootstrap
CIs, then bootstrap-resamples the items and re-ranks to measure how often the
ordering changes. Prints the quotable number (top-flip probability) and pairwise
win probabilities, and writes the money-shot figure.

Falls back to the frozen fixture if no real results file exists, so the pipeline
runs offline.

    uv run python experiments/01_ranking_instability.py
"""

from pathlib import Path

from agentstat.harness.runner import load_results
from agentstat.harness.fixture import make_frozen_results
from agentstat.ranking.stability import ranking_stability
from agentstat.core.bootstrap import bootstrap_ci

RESULTS = Path("results/bfcl_simple.jsonl")
FIG_DIR = Path("figures")


def load():
    if RESULTS.exists():
        print(f"Using real results: {RESULTS}")
        return load_results(RESULTS), "BFCL simple (DeepInfra)"
    print("No real results found — using frozen fixture (offline demo).")
    return make_frozen_results(), "frozen fixture"


def main():
    results, source = load()

    # Per-config accuracy with bootstrap CIs.
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in results:
        by_config[r.config_id].append(r.score)

    print(f"\n=== Per-config accuracy ({source}) ===")
    cis = {}
    for cfg, scores in by_config.items():
        ci = bootstrap_ci(scores, n_boot=5000, method="bca", seed=0)
        cis[cfg] = ci
        print(f"  {cfg:16s} acc={ci.point:.3f}  95% CI [{ci.low:.3f}, {ci.high:.3f}]")

    # Ranking stability — the headline.
    stab = ranking_stability(results, n_boot=5000, seed=1)
    print(f"\n=== Ranking stability ===")
    print(f"  Point ranking : {' > '.join(stab.point_ranking)}")
    print(f"  Modal ranking : {' > '.join(stab.modal_ranking)} "
          f"(appears {stab.modal_ranking_prob:.0%} of resamples)")
    print(f"  Expected Kendall-tau vs point ranking: {stab.expected_kendall_tau:.3f}")
    print(f"\n  >>> HEADLINE: {stab.headline}")

    print(f"\n  Pairwise P(row ranks above col):")
    configs = stab.point_ranking
    for a in configs:
        for b in configs:
            if a != b:
                p = stab.pairwise_win_prob[(a, b)]
                if 0.05 < p < 0.95:  # highlight the contestable pairs
                    print(f"    {a} > {b}: {p:.2f}  <-- not decisive")

    _plot(cis, stab, source)


def _plot(cis, stab, source):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping figure. `uv sync --extra plot`)")
        return

    FIG_DIR.mkdir(exist_ok=True)
    configs = stab.point_ranking
    points = [cis[c].point for c in configs]
    lows = [cis[c].point - cis[c].low for c in configs]
    highs = [cis[c].high - cis[c].point for c in configs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = range(len(configs))
    ax.errorbar(points, y, xerr=[lows, highs], fmt="o", color="#2b6cb0",
                capsize=4, markersize=8, linewidth=2)
    ax.set_yticks(list(y))
    ax.set_yticklabels(configs)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (bootstrap BCa 95% CI)")
    ax.set_title(
        f"BFCL ranking is unstable: top config flips {stab.top_flip_prob:.0%} "
        f"of resamples\n({source})", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "01_ranking_instability.png"
    fig.savefig(out, dpi=140)
    print(f"\n  Figure written: {out}")


if __name__ == "__main__":
    main()
