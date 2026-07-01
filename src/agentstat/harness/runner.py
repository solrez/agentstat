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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

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


def _eval_item(
    config: Config,
    prov: ChatProvider,
    item: BFCLItem,
    seeds: tuple[int, ...],
) -> tuple[list[EvalResult], int]:
    """Evaluate one item across all seeds. Returns (results, n_cached).

    Raises on any provider error so the caller can decide to skip or re-raise;
    an item is atomic — either all its seeds succeed or it is dropped whole.
    """
    tools = to_openai_tools(item.functions)
    messages = [{"role": "user", "content": item.prompt}]
    item_results: list[EvalResult] = []
    n_cached = 0
    for seed in seeds:
        response = prov.chat(
            model=config.model,
            messages=messages,
            tools=tools,
            temperature=config.temperature,
            seed=seed,
        )
        n_cached += int(getattr(prov, "last_call_cached", False))
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
    return item_results, n_cached


def run_config(
    config: Config,
    items: list[BFCLItem],
    seeds: tuple[int, ...] = (0,),
    provider: ChatProvider | None = None,
    logger: logging.Logger | None = None,
    log_every: int = 25,
    skip_errors: bool = True,
    max_workers: int = 8,
) -> tuple[list[EvalResult], set[str]]:
    """Evaluate one config over ``items`` across ``seeds``, items run concurrently.

    Returns ``(results, failed_item_ids)``. Each item is dispatched to a thread
    pool of ``max_workers`` (httpx is blocking, so threads overlap the network
    latency). An item is atomic: if any of its seeds errors and ``skip_errors``
    is True, the whole item is logged and skipped and its id recorded, so the
    caller can keep the item set paired across configs. ``max_workers=1`` runs
    sequentially. With ``skip_errors=False`` the first error re-raises.

    The cache is safe under concurrency: a hit just reads a file, and duplicate
    in-flight misses at worst write the same content-addressed file twice.
    """
    log = logger or _log
    prov = provider or ChatProvider(provider=config.provider)
    results: list[EvalResult] = []
    failed: set[str] = set()

    total = len(items) * len(seeds)
    log.info("config %s: starting %d calls (%d items x %d seeds) x%d workers, model=%s",
             config.id, total, len(items), len(seeds), max_workers, config.model)

    done = 0
    n_cached = 0
    n_pass = 0
    t0 = time.time()
    lock = Lock()

    def handle(item: BFCLItem):
        nonlocal done, n_cached, n_pass
        try:
            item_results, item_cached = _eval_item(config, prov, item, seeds)
        except Exception:
            log.exception("config %s: item=%s failed", config.id, item.id)
            if not skip_errors:
                raise
            with lock:
                failed.add(item.id)
                done += len(seeds)
            return
        with lock:
            results.extend(item_results)
            n_cached += item_cached
            n_pass += sum(int(r.score == 1.0) for r in item_results)
            done += len(seeds)
            if done % log_every < len(seeds) or done >= total:
                rate = done / max(time.time() - t0, 1e-6)
                scored = max(done - len(failed) * len(seeds), 1)
                log.info(
                    "config %s: %d/%d (%.0f%%) | acc=%.3f | %d cached | %d failed | %.1f calls/s",
                    config.id, done, total, 100 * done / total,
                    n_pass / scored, n_cached, len(failed), rate,
                )

    if max_workers <= 1:
        for item in items:
            handle(item)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(handle, item) for item in items]
            for f in as_completed(futures):
                f.result()  # propagate re-raised errors when skip_errors=False

    log.info("config %s: done. acc=%.3f, %d cached, %d items failed",
             config.id, n_pass / max(len(results), 1), n_cached, len(failed))
    return results, failed


def run_benchmark(
    configs: list[Config],
    items: list[BFCLItem],
    seeds: tuple[int, ...] = (0,),
    logger: logging.Logger | None = None,
    skip_errors: bool = True,
    max_workers: int = 8,
) -> list[EvalResult]:
    """Run every config over the SAME items, preserving the paired-item invariant.

    Any item that fails for *any* config is dropped from *every* config's results,
    so all configs share an identical item set (what ``ranking_stability`` and
    ``paired_bootstrap_diff`` require). ``max_workers`` sets per-config request
    concurrency.
    """
    log = logger or _log
    log.info("benchmark: %d configs x %d items x %d seeds = %d total calls",
             len(configs), len(items), len(seeds),
             len(configs) * len(items) * len(seeds))

    per_config: list[list[EvalResult]] = []
    all_failed: set[str] = set()
    for config in configs:
        res, failed = run_config(
            config, items, seeds=seeds, logger=log, skip_errors=skip_errors,
            max_workers=max_workers,
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
