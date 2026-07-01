"""Run configs against a BFCL AST category and emit EvalResults.

A "config" is a (provider, model, temperature) triple with a stable id. The
runner evaluates **every config on the same item set** — the paired-resample
invariant that ``ranking_stability`` and ``paired_bootstrap_diff`` depend on —
optionally across several seeds so seed variance can be decomposed.

Every provider call is cached, so a full rerun costs nothing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from agentstat.data.schema import EvalResult
from agentstat.harness.bfcl import BFCLItem, to_openai_tools
from agentstat.harness.providers import ChatProvider
from agentstat.harness.scoring import score_prediction, extract_call

_log = logging.getLogger("agentstat")


@dataclass(frozen=True)
class Config:
    """One evaluation config: a model at a temperature on a provider."""

    id: str
    provider: str
    model: str
    temperature: float = 0.0


def run_config(
    config: Config,
    items: list[BFCLItem],
    seeds: tuple[int, ...] = (0,),
    provider: ChatProvider | None = None,
    logger: logging.Logger | None = None,
    log_every: int = 25,
) -> list[EvalResult]:
    """Evaluate one config over ``items`` across ``seeds``; return EvalResults.

    Logs per-config progress at ``log_every``-call cadence, distinguishes cached
    from live calls, records the running accuracy, and logs any call that errors
    (the error is re-raised — a failed call should not silently score 0).
    """
    log = logger or _log
    prov = provider or ChatProvider(provider=config.provider)
    results: list[EvalResult] = []

    total = len(items) * len(seeds)
    log.info("config %s: starting %d calls (%d items x %d seeds), model=%s",
             config.id, total, len(items), len(seeds), config.model)

    done = 0
    n_cached = 0
    n_pass = 0
    t0 = time.time()
    for item in items:
        tools = to_openai_tools(item.functions)
        messages = [{"role": "user", "content": item.prompt}]
        for seed in seeds:
            try:
                response = prov.chat(
                    model=config.model,
                    messages=messages,
                    tools=tools,
                    temperature=config.temperature,
                    seed=seed,
                )
            except Exception:
                log.exception("config %s: call failed on item=%s seed=%s",
                              config.id, item.id, seed)
                raise

            # Providers that support caching expose last_call_cached; others
            # simply report False (the runner stays decoupled from the impl).
            cached = getattr(prov, "last_call_cached", False)
            name, args = extract_call(response)
            score = score_prediction(name, args, item.ground_truth)
            results.append(
                EvalResult(
                    config_id=config.id,
                    item_id=item.id,
                    score=score,
                    seed=seed,
                    prompt_variant=None,
                    metadata={
                        "provider": config.provider,
                        "model": config.model,
                        "temperature": config.temperature,
                        "predicted_name": name,
                    },
                )
            )
            done += 1
            n_cached += int(cached)
            n_pass += int(score == 1.0)
            if done % log_every == 0 or done == total:
                rate = done / max(time.time() - t0, 1e-6)
                log.info(
                    "config %s: %d/%d (%.0f%%) | acc=%.3f | %d cached | %.1f calls/s",
                    config.id, done, total, 100 * done / total,
                    n_pass / done, n_cached, rate,
                )

    log.info("config %s: done. acc=%.3f (n=%d), %d/%d from cache",
             config.id, n_pass / total, total, n_cached, total)
    return results


def run_benchmark(
    configs: list[Config],
    items: list[BFCLItem],
    seeds: tuple[int, ...] = (0,),
    logger: logging.Logger | None = None,
) -> list[EvalResult]:
    """Run every config over the SAME items (shared-item-set invariant)."""
    log = logger or _log
    log.info("benchmark: %d configs x %d items x %d seeds = %d total calls",
             len(configs), len(items), len(seeds),
             len(configs) * len(items) * len(seeds))
    all_results: list[EvalResult] = []
    for config in configs:
        all_results.extend(run_config(config, items, seeds=seeds, logger=log))
    log.info("benchmark: complete, %d results", len(all_results))
    return all_results


def save_results(results: list[EvalResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r.to_dict()) for r in results))


def load_results(path: str | Path) -> list[EvalResult]:
    text = Path(path).read_text()
    return [EvalResult.from_dict(json.loads(l)) for l in text.splitlines() if l.strip()]
