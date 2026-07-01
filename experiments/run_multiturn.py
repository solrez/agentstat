"""Run BFCL multi-turn rollouts (no-execution) and cache the trajectory metrics.

Produces a list[EvalResult] with n_turns / n_tool_calls / proxy-score populated,
across seeds, which the agent-variance experiment decomposes.

Multi-turn episodes are many calls each (each turn can take several tool-call
steps), so this is far heavier than the single-turn run. Defaults are a subset;
raise N_ITEMS / add categories for a fuller run. Everything is cached, so reruns
are free.

    uv run python experiments/run_multiturn.py
"""

from datetime import datetime
from pathlib import Path

from agentstat.harness.multiturn import load_multi_turn
from agentstat.harness.rollout import run_multi_turn
from agentstat.harness.runner import Config, save_results
from agentstat.logging_utils import get_logger

# One capable, cheap, tool-reliable model — the variance question is about the
# agent's own nondeterminism, so a single config across seeds is what we need.
CONFIG = Config(id="gemma-4-26b", provider="deepinfra",
                model="google/gemma-4-26B-A4B-it")

CATEGORIES = ["multi_turn_base"]   # add more (long_context, miss_func, miss_param) for breadth
N_ITEMS = 60          # per category; multi-turn episodes are expensive
SEEDS = (0, 1, 2)     # replicate seeds -> lets us separate seed from item variance
MAX_WORKERS = 8
OUT = Path("results/bfcl_multiturn.jsonl")


def main():
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = get_logger("mt-run", run_id=run_id)

    all_results = []
    for category in CATEGORIES:
        items = load_multi_turn(category=category, limit=N_ITEMS)
        log.info("category %s: %d items, avg n_turns=%.1f",
                 category, len(items), sum(i.n_turns for i in items) / len(items))
        results, failed = run_multi_turn(CONFIG, items, seeds=SEEDS, logger=log,
                                         max_workers=MAX_WORKERS)
        all_results.extend(results)

    save_results(all_results, OUT)
    log.info("saved %d results to %s", len(all_results), OUT)

    # Quick sanity: n_tool_calls spread at fixed item across seeds.
    from collections import defaultdict
    by_item = defaultdict(list)
    for r in all_results:
        by_item[r.item_id].append(r.n_tool_calls)
    varying = sum(1 for v in by_item.values() if len(set(v)) > 1)
    log.info("items where n_tool_calls varies across seeds: %d/%d",
             varying, len(by_item))


if __name__ == "__main__":
    main()
