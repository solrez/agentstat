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
    Config(id="nemotron-3-120b", provider="deepinfra",
           model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"),
    Config(id="deepseek-v4-flash", provider="deepinfra",
           model="deepseek-ai/DeepSeek-V4-Flash"),
    Config(id="gemma-4-26b", provider="deepinfra",
           model="google/gemma-4-26B-A4B-it"),
]

N_ITEMS = 400
SEEDS = (0, 1, 2)
MAX_WORKERS = 8
OUT = Path("results/bfcl_simple.jsonl")


def main():
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = get_logger("bfcl-run", run_id=run_id)

    items = load_bfcl_simple(category="simple_python", limit=N_ITEMS)
    log.info("Loaded %d BFCL simple items.", len(items))

    results = run_benchmark(CONFIGS, items, seeds=SEEDS, logger=log,
                            max_workers=MAX_WORKERS)
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
