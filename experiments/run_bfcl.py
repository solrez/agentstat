"""Run the real BFCL 'simple' benchmark on DeepInfra and cache the results.

Three configs across capability tiers, 200 items x 3 seeds each, all cached so
reruns cost nothing. Writes a shared-item-set list[EvalResult] to disk that the
ranking-instability and variance experiments consume.

    uv run python experiments/run_bfcl.py
"""

from datetime import datetime
from pathlib import Path

from agentstat.harness.bfcl import load_bfcl_simple
from agentstat.harness.runner import Config, run_benchmark, save_results
from agentstat.logging_utils import get_logger

CONFIGS = [
    Config(id="llama-3.1-70b", provider="deepinfra",
           model="meta-llama/Meta-Llama-3.1-70B-Instruct"),
    Config(id="qwen2.5-7b", provider="deepinfra",
           model="Qwen/Qwen2.5-7B-Instruct"),
    Config(id="llama-3.1-8b", provider="deepinfra",
           model="meta-llama/Meta-Llama-3.1-8B-Instruct"),
]

N_ITEMS = 200
SEEDS = (0, 1, 2)
OUT = Path("results/bfcl_simple.jsonl")


def main():
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = get_logger("bfcl-run", run_id=run_id)

    items = load_bfcl_simple(category="simple_python", limit=N_ITEMS)
    log.info("Loaded %d BFCL simple items.", len(items))

    results = run_benchmark(CONFIGS, items, seeds=SEEDS, logger=log)
    save_results(results, OUT)

    # Quick per-config accuracy summary.
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[r.config_id].append(r.score)
    log.info("Saved %d results to %s", len(results), OUT)
    for cfg, scores in agg.items():
        log.info("  %s: acc=%.3f (n=%d)", cfg, sum(scores) / len(scores), len(scores))


if __name__ == "__main__":
    main()
