"""Run configs against a BFCL AST category and emit EvalResults.

A "config" is a (provider, model, temperature) triple with a stable id. The
runner evaluates **every config on the same item set** — the paired-resample
invariant that ``ranking_stability`` and ``paired_bootstrap_diff`` depend on —
optionally across several seeds so seed variance can be decomposed.

Every provider call is cached, so a full rerun costs nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentstat.data.schema import EvalResult
from agentstat.harness.bfcl import BFCLItem, to_openai_tools
from agentstat.harness.providers import ChatProvider
from agentstat.harness.scoring import score_prediction, extract_call


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
) -> list[EvalResult]:
    """Evaluate one config over ``items`` across ``seeds``; return EvalResults."""
    prov = provider or ChatProvider(provider=config.provider)
    results: list[EvalResult] = []
    for item in items:
        tools = to_openai_tools(item.functions)
        messages = [{"role": "user", "content": item.prompt}]
        for seed in seeds:
            response = prov.chat(
                model=config.model,
                messages=messages,
                tools=tools,
                temperature=config.temperature,
                seed=seed,
            )
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
    return results


def run_benchmark(
    configs: list[Config],
    items: list[BFCLItem],
    seeds: tuple[int, ...] = (0,),
) -> list[EvalResult]:
    """Run every config over the SAME items (shared-item-set invariant)."""
    all_results: list[EvalResult] = []
    for config in configs:
        all_results.extend(run_config(config, items, seeds=seeds))
    return all_results


def save_results(results: list[EvalResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r.to_dict()) for r in results))


def load_results(path: str | Path) -> list[EvalResult]:
    text = Path(path).read_text()
    return [EvalResult.from_dict(json.loads(l)) for l in text.splitlines() if l.strip()]
