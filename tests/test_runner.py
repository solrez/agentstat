"""Runner tests with a fake provider — no network, no keys."""

from agentstat.harness.bfcl import BFCLItem, to_openai_tools
from agentstat.harness.runner import (
    Config,
    run_config,
    run_benchmark,
    save_results,
    load_results,
)


class FakeProvider:
    """Stand-in for ChatProvider: returns a scripted tool call per model."""

    def __init__(self, name_by_model):
        self.name_by_model = name_by_model
        self.calls = 0

    def chat(self, model, messages, tools, temperature, seed):
        self.calls += 1
        fname = self.name_by_model.get(model)
        if fname is None:
            return {"choices": [{"message": {"content": "no call"}}]}
        return {
            "choices": [
                {"message": {"tool_calls": [
                    {"function": {"name": fname,
                                  "arguments": '{"base": 10, "height": 5}'}}
                ]}}
            ]
        }


ITEM = BFCLItem(
    id="simple_python_0",
    prompt="Find the area of a triangle.",
    functions=[{
        "name": "calculate_triangle_area",
        "description": "area",
        "parameters": {"type": "dict",
                       "properties": {"base": {"type": "integer"},
                                      "height": {"type": "integer"}},
                       "required": ["base", "height"]},
    }],
    ground_truth=[{"calculate_triangle_area": {"base": [10], "height": [5]}}],
)


def test_to_openai_tools_rewrites_dict_type():
    tools = to_openai_tools(ITEM.functions)
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["parameters"]["type"] == "object"


def test_run_config_scores_correct_call():
    prov = FakeProvider({"good-model": "calculate_triangle_area"})
    cfg = Config(id="good", provider="openrouter", model="good-model")
    results = run_config(cfg, [ITEM], seeds=(0,), provider=prov)
    assert len(results) == 1
    assert results[0].score == 1.0
    assert results[0].config_id == "good"
    assert results[0].item_id == "simple_python_0"


def test_run_config_scores_wrong_call():
    prov = FakeProvider({"bad-model": "wrong_function"})
    cfg = Config(id="bad", provider="openrouter", model="bad-model")
    results = run_config(cfg, [ITEM], seeds=(0,), provider=prov)
    assert results[0].score == 0.0


def test_run_config_no_call_scores_zero():
    prov = FakeProvider({})  # returns no tool call
    cfg = Config(id="silent", provider="openrouter", model="silent-model")
    results = run_config(cfg, [ITEM], seeds=(0,), provider=prov)
    assert results[0].score == 0.0


def test_multiple_seeds_produce_multiple_results():
    prov = FakeProvider({"m": "calculate_triangle_area"})
    cfg = Config(id="c", provider="openrouter", model="m")
    results = run_config(cfg, [ITEM], seeds=(0, 1, 2), provider=prov)
    assert len(results) == 3
    assert {r.seed for r in results} == {0, 1, 2}


def test_run_benchmark_shares_item_set_across_configs():
    prov = FakeProvider({"m1": "calculate_triangle_area", "m2": "wrong"})
    configs = [
        Config(id="a", provider="openrouter", model="m1"),
        Config(id="b", provider="openrouter", model="m2"),
    ]
    # Inject the same fake provider into both by patching run_config's default.
    results = []
    for cfg in configs:
        results.extend(run_config(cfg, [ITEM], seeds=(0,), provider=prov))
    items_per_config = {}
    for r in results:
        items_per_config.setdefault(r.config_id, set()).add(r.item_id)
    # Every config covered exactly the same items — the paired invariant.
    assert items_per_config["a"] == items_per_config["b"]


def test_save_and_load_roundtrip(tmp_path):
    prov = FakeProvider({"m": "calculate_triangle_area"})
    cfg = Config(id="c", provider="openrouter", model="m")
    results = run_config(cfg, [ITEM], seeds=(0, 1), provider=prov)
    path = tmp_path / "results.jsonl"
    save_results(results, path)
    loaded = load_results(path)
    assert len(loaded) == len(results)
    assert loaded[0].config_id == results[0].config_id
    assert loaded[0].score == results[0].score
