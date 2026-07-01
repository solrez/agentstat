"""Multi-turn agent variance — the two-panel novel finding.

Panel A: proxy-SUCCESS variance decomposed across item vs seed. Directly
comparable to the single-turn result (where seed variance was ~1%). The question:
is a multi-turn agent's success MORE seed-sensitive than single-turn scoring?

Panel B: n_tool_calls seed variance at FIXED task. Agents vary in *how* they
solve a task (how many tool calls) across reruns of the identical item — a source
of nondeterminism single-turn evals have no equivalent for.

    uv run python experiments/03_agent_variance.py
"""

from pathlib import Path

import numpy as np

from agentstat.harness.runner import load_results
from agentstat.core.variance import decompose_variance
from agentstat.data.schema import EvalResult

RESULTS = Path("results/bfcl_multiturn.jsonl")
FIG_DIR = Path("figures")


def _binarize_success(results, threshold=1.0):
    """Proxy score is continuous recall; binarize to success=(recall>=threshold)
    so the tested binary GLMM decomposition applies."""
    return [
        EvalResult(
            config_id=r.config_id, item_id=r.item_id,
            score=float(r.score >= threshold), seed=r.seed,
            n_turns=r.n_turns, n_tool_calls=r.n_tool_calls, metadata=r.metadata,
        )
        for r in results
    ]


def _within_item_toolcall_variance(results):
    """Panel B: variance of n_tool_calls WITHIN item across seeds (fixed task),
    vs between items. Returns (within_frac, between_frac, mean_within_sd)."""
    from collections import defaultdict
    by_item = defaultdict(list)
    for r in results:
        by_item[r.item_id].append(r.n_tool_calls)
    item_means = {k: np.mean(v) for k, v in by_item.items()}
    grand = np.mean([n for v in by_item.values() for n in v])

    within = np.mean([np.var(v, ddof=0) for v in by_item.values() if len(v) > 1])
    between = np.var(list(item_means.values()), ddof=0)
    total = within + between
    within_sd = np.mean([np.std(v, ddof=0) for v in by_item.values() if len(v) > 1])
    return within / total, between / total, within_sd, grand


def main():
    if not RESULTS.exists():
        print(f"No multi-turn results at {RESULTS}. Run experiments/run_multiturn.py first.")
        return
    results = load_results(RESULTS)
    print(f"Loaded {len(results)} multi-turn results, "
          f"{len({r.item_id for r in results})} items, "
          f"seeds {sorted({r.seed for r in results})}")

    # ---- Panel A: success variance item vs seed ----
    binary = _binarize_success(results, threshold=1.0)
    succ = np.mean([b.score for b in binary])
    print(f"\n=== Panel A: proxy-success (recall==1.0) variance ===")
    print(f"  overall success rate: {succ:.3f}")
    decA = decompose_variance(binary, factors=("item_id", "seed"))
    print(decA.as_table().to_string(index=False))
    for note in decA.notes:
        print(f"  note: {note}")
    print(f"  dominant source: {decA.top_source()}")

    # ---- Panel B: n_tool_calls within-item (seed) variance ----
    within_f, between_f, within_sd, grand = _within_item_toolcall_variance(results)
    print(f"\n=== Panel B: n_tool_calls nondeterminism ===")
    print(f"  mean n_tool_calls: {grand:.1f}")
    print(f"  within-item (seed) variance share : {within_f:.1%}")
    print(f"  between-item (task) variance share: {between_f:.1%}")
    print(f"  mean within-item SD (calls that vary by seed alone): {within_sd:.2f}")
    print(f"\n  >>> Even at a FIXED task, the agent's tool-call count varies by "
          f"~{within_sd:.1f} calls across seeds — nondeterminism single-turn evals lack.")

    _plot(decA, within_f, between_f, within_sd)


def _plot(decA, within_f, between_f, within_sd):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping figure.)")
        return
    FIG_DIR.mkdir(exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: success variance components
    labels = [c.factor for c in decA.components] + ["residual"]
    pcts = [c.pct_of_total for c in decA.components] + [decA.residual_pct]
    axA.bar(labels, pcts, color=["#2b6cb0", "#dd6b20", "#a0aec0"][:len(labels)])
    for i, p in enumerate(pcts):
        axA.text(i, p + 1, f"{p:.0f}%", ha="center", fontsize=10)
    axA.set_ylabel("% of success variance (logit)")
    axA.set_title("A. Agent success: item vs seed", fontsize=11)
    axA.set_ylim(0, max(pcts) * 1.2)

    # Panel B: tool-call variance within vs between item
    axB.bar(["within-item\n(seed)", "between-item\n(task)"],
            [within_f * 100, between_f * 100], color=["#dd6b20", "#2b6cb0"])
    axB.text(0, within_f * 100 + 1, f"{within_f:.0%}", ha="center", fontsize=10)
    axB.text(1, between_f * 100 + 1, f"{between_f:.0%}", ha="center", fontsize=10)
    axB.set_ylabel("% of n_tool_calls variance")
    axB.set_title(f"B. Tool-call nondeterminism\n(~{within_sd:.1f} calls vary by seed alone)",
                  fontsize=11)
    axB.set_ylim(0, 100)

    fig.suptitle("Multi-turn agent variance (BFCL multi_turn, no-execution rollout)",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "03_agent_variance.png"
    fig.savefig(out, dpi=140)
    print(f"\n  Figure written: {out}")


if __name__ == "__main__":
    main()
