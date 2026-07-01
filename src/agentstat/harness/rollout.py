"""No-execution multi-turn rollout: measure agent trajectory metrics.

We roll a model through a BFCL multi-turn episode turn-by-turn, but we do NOT
execute its tool calls against real stateful classes (per the proxy-first
decision). Instead, each emitted tool call is counted and answered with a stubbed
generic result so the conversation continues. This is enough to produce the
agent-variance signal:

  - ``n_turns``       : fixed per item (the number of user turns).
  - ``n_tool_calls``  : total tool calls the model made across all turns/steps —
                        this is the SEED-VARYING trajectory quantity.
  - proxy ``score``   : recall of ground-truth call names (fraction of the
                        distinct functions BFCL expected that the model invoked).

Caveats (stated honestly): without real execution, tool results are stubbed, so
later turns are slightly off the true distribution, and the proxy score is not
BFCL's official state-based metric. It is a consistent signal for *variance*
comparison, which is the question we're answering.

Per-turn the model may call tools over several steps; we stop a turn when it
emits no tool call or hits the step cap (BFCL uses 20).
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock

from agentstat.data.schema import EvalResult
from agentstat.harness.multiturn import MultiTurnItem, tools_for_item
from agentstat.harness.providers import ChatProvider
from agentstat.harness.runner import Config
from agentstat.harness.scoring import extract_all_calls

_log = logging.getLogger("agentstat")

MAX_STEPS_PER_TURN = 20  # BFCL's MAXIMUM_STEP_LIMIT
_STUB_TOOL_RESULT = json.dumps({"status": "ok", "note": "stubbed (no-execution rollout)"})

# Ground-truth calls are bare strings like "cd(folder='document')"; pull the name.
_CALL_NAME_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _gt_call_names(ground_truth: list[list[str]]) -> set[str]:
    names: set[str] = set()
    for turn in ground_truth:
        for call in turn:
            m = _CALL_NAME_RE.match(call)
            if m:
                names.add(m.group(1))
    return names


@dataclass(frozen=True)
class RolloutResult:
    item_id: str
    n_turns: int
    n_tool_calls: int
    called_names: set[str]
    proxy_score: float   # recall of ground-truth call names in [0, 1]


def rollout_episode(
    config: Config,
    item: MultiTurnItem,
    provider: ChatProvider,
    seed: int,
) -> RolloutResult:
    """Run one no-execution rollout of ``item`` and return trajectory metrics."""
    history: list[dict] = []
    n_tool_calls = 0
    called_names: set[str] = set()

    for turn_idx in range(item.n_turns):
        tools = tools_for_item(item, turn_idx=turn_idx)
        history.append({"role": "user", "content": item.user_message(turn_idx)})

        steps = 0
        while steps < MAX_STEPS_PER_TURN:
            response = provider.chat(
                model=config.model,
                messages=history,
                tools=tools,
                temperature=config.temperature,
                seed=seed,
            )
            message = response["choices"][0]["message"]
            calls = extract_all_calls(response)

            if not calls:
                # Model gave a text answer -> this turn is done.
                history.append({"role": "assistant",
                                "content": message.get("content") or ""})
                break

            # Record the assistant's tool-call message, then stub each result.
            history.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": message.get("tool_calls", []),
            })
            for name, _args in calls:
                if name:
                    called_names.add(name)
                    n_tool_calls += 1
            for tc in message.get("tool_calls", []):
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": _STUB_TOOL_RESULT,
                })
            steps += 1

    gt_names = _gt_call_names(item.ground_truth)
    proxy = len(called_names & gt_names) / len(gt_names) if gt_names else 0.0
    return RolloutResult(
        item_id=item.id,
        n_turns=item.n_turns,
        n_tool_calls=n_tool_calls,
        called_names=called_names,
        proxy_score=proxy,
    )


def run_multi_turn(
    config: Config,
    items: list[MultiTurnItem],
    seeds: tuple[int, ...] = (0,),
    provider: ChatProvider | None = None,
    logger: logging.Logger | None = None,
    max_workers: int = 8,
    log_every: int = 10,
) -> tuple[list[EvalResult], set[str]]:
    """Roll out ``config`` over multi-turn ``items`` across ``seeds``, concurrently.

    Returns ``(results, failed_item_ids)``. Each EvalResult carries the proxy
    ``score`` plus ``n_turns`` and ``n_tool_calls`` — the fields ``agent_variance``
    decomposes. An episode is atomic (all seeds) and dropped whole on error.
    """
    log = logger or _log
    prov = provider or ChatProvider(provider=config.provider)
    results: list[EvalResult] = []
    failed: set[str] = set()
    total = len(items) * len(seeds)
    log.info("multiturn %s: %d episodes (%d items x %d seeds) x%d workers, model=%s",
             config.id, total, len(items), len(seeds), max_workers, config.model)

    done = 0
    t0 = time.time()
    lock = Lock()

    def handle(item: MultiTurnItem):
        nonlocal done
        try:
            episode_results = []
            for seed in seeds:
                rr = rollout_episode(config, item, prov, seed)
                episode_results.append(
                    EvalResult(
                        config_id=config.id,
                        item_id=item.id,
                        score=rr.proxy_score,
                        seed=seed,
                        n_turns=rr.n_turns,
                        n_tool_calls=rr.n_tool_calls,
                        metadata={"provider": config.provider, "model": config.model,
                                  "called_names": sorted(rr.called_names)},
                    )
                )
        except Exception:
            log.exception("multiturn %s: item=%s failed", config.id, item.id)
            with lock:
                failed.add(item.id)
                done += len(seeds)
            return
        with lock:
            results.extend(episode_results)
            done += len(seeds)
            if done % log_every < len(seeds) or done >= total:
                rate = done / max(time.time() - t0, 1e-6)
                log.info("multiturn %s: %d/%d (%.0f%%) | %d failed | %.2f ep/s",
                         config.id, done, total, 100 * done / total,
                         len(failed), rate)

    if max_workers <= 1:
        for item in items:
            handle(item)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for f in as_completed([pool.submit(handle, it) for it in items]):
                f.result()

    log.info("multiturn %s: done. %d results, %d items failed",
             config.id, len(results), len(failed))
    return results, failed
