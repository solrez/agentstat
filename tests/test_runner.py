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


def test_to_openai_tools_sanitizes_nested_nonstandard_types():
    # The bug that killed the 70b run: non-standard types nested deep in the
    # schema ("any", "float", "tuple") must all be rewritten, not just top-level.
    fn = {
        "name": "f",
        "description": "d",
        "parameters": {
            "type": "dict",
            "properties": {
                "data": {"type": "any"},                      # -> string
                "ratio": {"type": "float"},                   # -> number
                "coords": {"type": "tuple", "items": {"type": "float"}},  # -> array/number
                "nested": {
                    "type": "dict",
                    "properties": {"x": {"type": "any"}},     # deep -> string
                },
            },
            "required": ["data"],
        },
    }
    params = to_openai_tools([fn])[0]["function"]["parameters"]
    props = params["properties"]
    assert params["type"] == "object"
    assert props["data"]["type"] == "string"
    assert props["ratio"]["type"] == "number"
    assert props["coords"]["type"] == "array"
    assert props["coords"]["items"]["type"] == "number"
    assert props["nested"]["type"] == "object"
    assert props["nested"]["properties"]["x"]["type"] == "string"


def test_run_config_scores_correct_call():
    prov = FakeProvider({"good-model": "calculate_triangle_area"})
    cfg = Config(id="good", provider="openrouter", model="good-model")
    results, failed = run_config(cfg, [ITEM], seeds=(0,), provider=prov)
    assert len(results) == 1
    assert results[0].score == 1.0
    assert results[0].config_id == "good"
    assert results[0].item_id == "simple_python_0"
    assert failed == set()


def test_run_config_scores_wrong_call():
    prov = FakeProvider({"bad-model": "wrong_function"})
    cfg = Config(id="bad", provider="openrouter", model="bad-model")
    results, _ = run_config(cfg, [ITEM], seeds=(0,), provider=prov)
    assert results[0].score == 0.0


def test_run_config_no_call_scores_zero():
    prov = FakeProvider({})  # returns no tool call
    cfg = Config(id="silent", provider="openrouter", model="silent-model")
    results, _ = run_config(cfg, [ITEM], seeds=(0,), provider=prov)
    assert results[0].score == 0.0


def test_multiple_seeds_produce_multiple_results():
    prov = FakeProvider({"m": "calculate_triangle_area"})
    cfg = Config(id="c", provider="openrouter", model="m")
    results, _ = run_config(cfg, [ITEM], seeds=(0, 1, 2), provider=prov)
    assert len(results) == 3
    assert {r.seed for r in results} == {0, 1, 2}


class ExplodingProvider:
    """Fails on a specific item id; succeeds otherwise."""

    def __init__(self, bad_item_id):
        self.bad_item_id = bad_item_id
        self.last_call_cached = False

    def chat(self, model, messages, tools, temperature, seed):
        # We can't see item_id directly; encode it in the prompt for the fake.
        if self.bad_item_id in messages[0]["content"]:
            raise RuntimeError("simulated 422")
        return {"choices": [{"message": {"tool_calls": [
            {"function": {"name": "calculate_triangle_area",
                          "arguments": '{"base": 10, "height": 5}'}}]}}]}


def test_run_config_skips_failed_item():
    good = ITEM
    bad = BFCLItem(id="simple_python_BAD", prompt="BAD_MARKER trigger failure",
                   functions=ITEM.functions, ground_truth=ITEM.ground_truth)
    prov = ExplodingProvider(bad_item_id="BAD_MARKER")
    cfg = Config(id="c", provider="openrouter", model="m")
    results, failed = run_config(cfg, [good, bad], seeds=(0, 1), provider=prov)
    # good item -> 2 results; bad item -> skipped entirely, no partial rows.
    assert {r.item_id for r in results} == {"simple_python_0"}
    assert len(results) == 2
    assert failed == {"simple_python_BAD"}


def test_run_config_reraises_when_not_skipping():
    bad = BFCLItem(id="b", prompt="BAD_MARKER", functions=ITEM.functions,
                   ground_truth=ITEM.ground_truth)
    prov = ExplodingProvider(bad_item_id="BAD_MARKER")
    cfg = Config(id="c", provider="openrouter", model="m")
    try:
        run_config(cfg, [bad], seeds=(0,), provider=prov, skip_errors=False)
        assert False, "should have raised"
    except RuntimeError:
        pass


def test_run_benchmark_shares_item_set_across_configs():
    prov = FakeProvider({"m1": "calculate_triangle_area", "m2": "wrong"})
    configs = [
        Config(id="a", provider="openrouter", model="m1"),
        Config(id="b", provider="openrouter", model="m2"),
    ]
    results = []
    for cfg in configs:
        res, _ = run_config(cfg, [ITEM], seeds=(0,), provider=prov)
        results.extend(res)
    items_per_config = {}
    for r in results:
        items_per_config.setdefault(r.config_id, set()).add(r.item_id)
    # Every config covered exactly the same items — the paired invariant.
    assert items_per_config["a"] == items_per_config["b"]


def test_run_benchmark_drops_item_failed_by_any_config(monkeypatch):
    # config 'a' fails on the bad item; the whole benchmark must drop that item
    # for config 'b' too, keeping the item set paired.
    from agentstat.harness import runner as runner_mod

    good = ITEM
    bad = BFCLItem(id="simple_python_BAD", prompt="BAD_MARKER",
                   functions=ITEM.functions, ground_truth=ITEM.ground_truth)

    def fake_provider_for(config):
        if config.id == "a":
            return ExplodingProvider(bad_item_id="BAD_MARKER")
        # config b succeeds on everything
        return ExplodingProvider(bad_item_id="__never__")

    # Patch run_config to inject a per-config provider.
    orig = runner_mod.run_config
    def patched(config, items, seeds=(0,), provider=None, **kw):
        return orig(config, items, seeds=seeds, provider=fake_provider_for(config), **kw)
    monkeypatch.setattr(runner_mod, "run_config", patched)

    configs = [Config(id="a", provider="openrouter", model="m"),
               Config(id="b", provider="openrouter", model="m")]
    results = run_benchmark(configs, [good, bad], seeds=(0,))
    # BAD item dropped for BOTH configs; only the good item survives, x2 configs.
    assert {r.item_id for r in results} == {"simple_python_0"}
    assert {r.config_id for r in results} == {"a", "b"}


def test_concurrent_run_all_results_present_and_correct():
    # Many items through the thread pool: no results lost, no double-count,
    # counts consistent under concurrency.
    import threading

    class CountingProvider:
        def __init__(self):
            self.last_call_cached = False
            self._n = 0
            self._lock = threading.Lock()

        def chat(self, model, messages, tools, temperature, seed):
            with self._lock:
                self._n += 1
            return {"choices": [{"message": {"tool_calls": [
                {"function": {"name": "calculate_triangle_area",
                              "arguments": '{"base": 10, "height": 5}'}}]}}]}

    items = [
        BFCLItem(id=f"simple_python_{i}", prompt=f"item {i}",
                 functions=ITEM.functions, ground_truth=ITEM.ground_truth)
        for i in range(50)
    ]
    prov = CountingProvider()
    cfg = Config(id="c", provider="openrouter", model="m")
    results, failed = run_config(cfg, items, seeds=(0, 1), provider=prov,
                                 max_workers=8)
    assert failed == set()
    assert len(results) == 100                      # 50 items x 2 seeds
    assert len({r.item_id for r in results}) == 50  # every item present once-per-seed
    assert prov._n == 100                           # exactly one call per (item, seed)
    assert all(r.score == 1.0 for r in results)


def test_concurrent_skips_failed_items_only():
    import threading

    class SometimesFails:
        def __init__(self):
            self.last_call_cached = False
            self._lock = threading.Lock()

        def chat(self, model, messages, tools, temperature, seed):
            if "FAIL" in messages[0]["content"]:
                raise RuntimeError("boom")
            return {"choices": [{"message": {"tool_calls": [
                {"function": {"name": "calculate_triangle_area",
                              "arguments": '{"base": 10, "height": 5}'}}]}}]}

    items = []
    for i in range(20):
        prompt = "FAIL" if i % 5 == 0 else f"ok {i}"  # 4 of 20 fail
        items.append(BFCLItem(id=f"simple_python_{i}", prompt=prompt,
                              functions=ITEM.functions, ground_truth=ITEM.ground_truth))
    prov = SometimesFails()
    cfg = Config(id="c", provider="openrouter", model="m")
    results, failed = run_config(cfg, items, seeds=(0,), provider=prov, max_workers=8)
    assert len(failed) == 4
    assert len(results) == 16
    # No partial rows from failed items.
    assert failed.isdisjoint({r.item_id for r in results})


def test_save_and_load_roundtrip(tmp_path):
    prov = FakeProvider({"m": "calculate_triangle_area"})
    cfg = Config(id="c", provider="openrouter", model="m")
    results, _ = run_config(cfg, [ITEM], seeds=(0, 1), provider=prov)
    path = tmp_path / "results.jsonl"
    save_results(results, path)
    loaded = load_results(path)
    assert len(loaded) == len(results)
    assert loaded[0].config_id == results[0].config_id
    assert loaded[0].score == results[0].score
