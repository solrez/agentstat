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
    skip_errors: bool = True,
) -> tuple[list[EvalResult], set[str]]:
    """Evaluate one config over ``items`` across ``seeds``.

    Returns ``(results, failed_item_ids)``. When a call errors and ``skip_errors``
    is True, the whole item (all its seeds) is logged and skipped, and its id is
    added to ``failed_item_ids`` so the caller can keep the item set paired across
    configs. With ``skip_errors=False`` the error is re-raised.

    Logs per-config progress at ``log_every``-call cadence, distinguishing cached
    from live calls and tracking running accuracy.
    """
    log = logger or _log
    prov = provider or ChatProvider(provider=config.provider)
    results: list[EvalResult] = []
    failed: set[str] = set()

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
        item_results: list[EvalResult] = []
        try:
            for seed in seeds:
                response = prov.chat(
                    model=config.model,
                    messages=messages,
                    tools=tools,
                    temperature=config.temperature,
                    seed=seed,
                )
                # Providers that support caching expose last_call_cached; others
                # report False (the runner stays decoupled from the impl).
                cached = getattr(prov, "last_call_cached", False)
                name, args = extract_call(response)
                score = score_prediction(name, args, item.ground_truth)
                item_results.append(
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
                n_cached += int(cached)
                n_pass += int(score == 1.0)
                done += 1
        except Exception:
            log.exception("config %s: item=%s failed", config.id, item.id)
            if not skip_errors:
                raise
            failed.add(item.id)
            done += len(seeds) - len(item_results)  # account for the skipped seeds
            continue  # drop this item's partial results entirely
        else:
            results.extend(item_results)
        finally:
            if done % log_every < len(seeds) or done >= total:
                rate = done / max(time.time() - t0, 1e-6)
                log.info(
                    "config %s: %d/%d (%.0f%%) | acc=%.3f | %d cached | %d failed | %.1f calls/s",
                    config.id, done, total, 100 * done / total,
                    n_pass / max(done - len(failed) * len(seeds), 1),
                    n_cached, len(failed), rate,
                )

    log.info("config %s: done. acc=%.3f, %d cached, %d items failed",
             config.id, n_pass / max(len(results), 1), n_cached, len(failed))
    return results, failed


def run_benchmark(
    configs: list[Config],
    items: list[BFCLItem],
    seeds: tuple[int, ...] = (0,),
    logger: logging.Logger | None = None,
    skip_errors: bool = True,
) -> list[EvalResult]:
    """Run every config over the SAME items, preserving the paired-item invariant.

    Any item that fails for *any* config is dropped from *every* config's results,
    so all configs share an identical item set (what ``ranking_stability`` and
    ``paired_bootstrap_diff`` require).
    """
    log = logger or _log
    log.info("benchmark: %d configs x %d items x %d seeds = %d total calls",
             len(configs), len(items), len(seeds),
             len(configs) * len(items) * len(seeds))

    per_config: list[list[EvalResult]] = []
    all_failed: set[str] = set()
    for config in configs:
        res, failed = run_config(
            config, items, seeds=seeds, logger=log, skip_errors=skip_errors
        )
        per_config.append(res)
        all_failed |= failed

    if all_failed:
        log.warning("dropping %d item(s) that failed for at least one config, "
                    "to keep the item set paired: %s",
                    len(all_failed), sorted(all_failed))

    all_results = [
        r for res in per_config for r in res if r.item_id not in all_failed
    ]
    log.info("benchmark: complete, %d results across %d items",
             len(all_results),
             len({r.item_id for r in all_results}))
    return all_results


def save_results(results: list[EvalResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r.to_dict()) for r in results))


def load_results(path: str | Path) -> list[EvalResult]:
    text = Path(path).read_text()
    return [EvalResult.from_dict(json.loads(l)) for l in text.splitlines() if l.strip()]
