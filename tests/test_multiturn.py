"""Multi-turn loader tests: schema parsing + tool building, no network."""

import pytest

from agentstat.harness.multiturn import (
    MultiTurnItem,
    tools_for_item,
    CLASS_DOC_STEM,
)


def _item(**kw):
    base = dict(
        id="multi_turn_base_0",
        turns=[
            [{"role": "user", "content": "do X"}],
            [{"role": "user", "content": "then Y"}],
        ],
        involved_classes=["GorillaFileSystem"],
        initial_config={"GorillaFileSystem": {"root": {}}},
        excluded_function=[],
        missed_function={},
        ground_truth=[["cd(folder='a')"], ["ls()"]],
    )
    base.update(kw)
    return MultiTurnItem(**base)


def test_n_turns_and_user_message():
    item = _item()
    assert item.n_turns == 2
    assert item.user_message(0) == "do X"
    assert item.user_message(1) == "then Y"


def test_twitter_maps_to_posting_api():
    # The gotcha: TwitterAPI's docs live under posting_api, not twitter_api.
    assert CLASS_DOC_STEM["TwitterAPI"] == "posting_api"


def test_tools_for_item_filters_excluded_and_sanitizes(monkeypatch):
    # Stub the func-doc loader so no network is needed.
    fake_docs = {
        "GorillaFileSystem": [
            {"name": "cd", "description": "change dir",
             "parameters": {"type": "dict",
                            "properties": {"folder": {"type": "string"}},
                            "required": ["folder"]},
             "response": {"type": "dict"}},
            {"name": "cp", "description": "copy",
             "parameters": {"type": "dict", "properties": {}}},
        ]
    }
    monkeypatch.setattr(
        "agentstat.harness.multiturn.load_func_docs", lambda classes, **kw: fake_docs
    )
    item = _item(excluded_function=["cp"])
    tools = tools_for_item(item)
    names = {t["function"]["name"] for t in tools}
    assert names == {"cd"}                       # cp excluded
    params = tools[0]["function"]["parameters"]
    assert params["type"] == "object"            # dict -> object
    assert "response" not in tools[0]["function"]  # response dropped


def test_missed_function_unlocks_at_turn(monkeypatch):
    fake_docs = {
        "GorillaFileSystem": [
            {"name": "cd", "description": "", "parameters": {"type": "dict", "properties": {}}},
        ]
    }
    held = {"name": "secret", "description": "", "parameters": {"type": "dict", "properties": {}}}
    monkeypatch.setattr(
        "agentstat.harness.multiturn.load_func_docs", lambda classes, **kw: fake_docs
    )
    item = _item(missed_function={"1": [held]})

    # At turn 0, the held-out tool is not yet available.
    names0 = {t["function"]["name"] for t in tools_for_item(item, turn_idx=0)}
    assert names0 == {"cd"}
    # At turn 1, it unlocks.
    names1 = {t["function"]["name"] for t in tools_for_item(item, turn_idx=1)}
    assert names1 == {"cd", "secret"}


def test_unknown_class_raises(monkeypatch):
    item = _item(involved_classes=["NopeAPI"])
    with pytest.raises(ValueError, match="unknown involved class"):
        tools_for_item(item)
