"""No-execution rollout tests with a scripted fake provider — no network."""

import json

from agentstat.harness.multiturn import MultiTurnItem
from agentstat.harness.runner import Config
from agentstat.harness.rollout import (
    rollout_episode,
    run_multi_turn,
    _gt_call_names,
    MAX_STEPS_PER_TURN,
)


def _tool_response(names):
    """Build an OpenAI response with the given tool-call names (one step)."""
    return {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": f"c{i}", "function": {"name": n, "arguments": "{}"}}
        for i, n in enumerate(names)
    ]}}]}


def _text_response():
    return {"choices": [{"message": {"content": "done", "tool_calls": None}}]}


class ScriptedProvider:
    """Returns queued responses in order; supports counting."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.last_call_cached = False

    def chat(self, model, messages, tools, temperature, seed):
        r = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return r


ITEM = MultiTurnItem(
    id="multi_turn_base_0",
    turns=[
        [{"role": "user", "content": "turn 1"}],
        [{"role": "user", "content": "turn 2"}],
    ],
    involved_classes=["GorillaFileSystem"],
    initial_config={},
    ground_truth=[["cd(folder='a')", "ls()"], ["grep(pattern='x')"]],
)


def test_gt_call_names_parsing():
    gt = [["cd(folder='a')", "ls()"], ["grep(pattern='x')"]]
    assert _gt_call_names(gt) == {"cd", "ls", "grep"}


def test_rollout_counts_turns_and_calls(monkeypatch):
    # turn 1: model calls cd, ls (1 step) then a text answer ends the turn.
    # turn 2: model calls grep (1 step) then text ends.
    script = [
        _tool_response(["cd", "ls"]),   # turn 1 step 1
        _text_response(),               # turn 1 ends
        _tool_response(["grep"]),       # turn 2 step 1
        _text_response(),               # turn 2 ends
    ]
    prov = ScriptedProvider(script)
    cfg = Config(id="c", provider="deepinfra", model="m")
    rr = rollout_episode(cfg, ITEM, prov, seed=0)
    assert rr.n_turns == 2
    assert rr.n_tool_calls == 3            # cd, ls, grep
    assert rr.called_names == {"cd", "ls", "grep"}
    assert rr.proxy_score == 1.0           # all 3 gt names called


def test_rollout_partial_proxy(monkeypatch):
    # Model only ever calls cd -> recall 1/3.
    script = [_tool_response(["cd"]), _text_response(),
              _tool_response(["cd"]), _text_response()]
    prov = ScriptedProvider(script)
    cfg = Config(id="c", provider="deepinfra", model="m")
    rr = rollout_episode(cfg, ITEM, prov, seed=0)
    assert rr.called_names == {"cd"}
    assert abs(rr.proxy_score - 1 / 3) < 1e-9


def test_rollout_multi_step_within_turn(monkeypatch):
    # Model calls tools over 3 steps in turn 1 before answering.
    script = [
        _tool_response(["cd"]),
        _tool_response(["ls"]),
        _tool_response(["grep"]),
        _text_response(),          # ends turn 1
        _text_response(),          # turn 2: immediate text answer, 0 calls
    ]
    prov = ScriptedProvider(script)
    cfg = Config(id="c", provider="deepinfra", model="m")
    rr = rollout_episode(cfg, ITEM, prov, seed=0)
    assert rr.n_tool_calls == 3


def test_rollout_respects_step_cap(monkeypatch):
    # Model never stops calling tools -> capped at MAX_STEPS_PER_TURN per turn.
    prov = ScriptedProvider([_tool_response(["cd"])])  # always a tool call
    cfg = Config(id="c", provider="deepinfra", model="m")
    rr = rollout_episode(cfg, ITEM, prov, seed=0)
    # 2 turns, each capped at MAX_STEPS_PER_TURN calls (1 call/step).
    assert rr.n_tool_calls == 2 * MAX_STEPS_PER_TURN


class CyclingProvider:
    """Deterministic per-episode: emits (cd,ls)->text for turn1, grep->text for turn2,
    detecting the turn from the current user message so it works across many episodes."""

    def __init__(self):
        self.last_call_cached = False

    def chat(self, model, messages, tools, temperature, seed):
        # Was the last assistant a tool call already this turn? If so, answer text.
        last = messages[-1]
        if last.get("role") == "tool":
            return _text_response()
        content = last.get("content", "")
        if "turn 1" in content:
            return _tool_response(["cd", "ls"])
        if "turn 2" in content:
            return _tool_response(["grep"])
        return _text_response()


def test_run_multi_turn_populates_agent_fields(monkeypatch):
    monkeypatch.setattr("agentstat.harness.rollout.ChatProvider",
                        lambda *a, **k: CyclingProvider())
    cfg = Config(id="c", provider="deepinfra", model="m")
    results, failed = run_multi_turn(cfg, [ITEM], seeds=(0, 1), max_workers=1)
    assert failed == set()
    assert len(results) == 2                     # 1 item x 2 seeds
    for r in results:
        assert r.n_turns == 2
        assert r.n_tool_calls == 3
        assert r.score == 1.0
        assert r.n_turns is not None and r.n_tool_calls is not None
