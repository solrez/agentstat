"""The frozen fixture must be deterministic and yield an unstable top ranking."""

from agentstat.harness.fixture import make_frozen_results
from agentstat.ranking.stability import ranking_stability


def test_fixture_is_deterministic():
    a = make_frozen_results(n_items=50, seeds=(0, 1))
    b = make_frozen_results(n_items=50, seeds=(0, 1))
    assert [r.to_dict() for r in a] == [r.to_dict() for r in b]


def test_fixture_shape():
    results = make_frozen_results(n_items=100, seeds=(0, 1, 2))
    assert len(results) == 4 * 100 * 3  # configs x items x seeds
    assert {r.config_id for r in results} == {
        "alpha-8b", "beta-8b", "gamma-7b", "delta-7b"
    }
    assert all(r.score in (0.0, 1.0) for r in results)


def test_fixture_top_ranking_is_unstable():
    # The headline property: the two strong configs are near-tied, so the top
    # rank flips a large fraction of the time. This is what the money-shot
    # experiment demonstrates.
    results = make_frozen_results()
    res = ranking_stability(results, n_boot=1000, seed=1)
    assert res.top_flip_prob > 0.25          # genuinely unstable
    assert res.point_ranking[0] in ("alpha-8b", "beta-8b")


def test_fixture_shares_item_set_across_configs():
    results = make_frozen_results(n_items=30, seeds=(0,))
    by_config = {}
    for r in results:
        by_config.setdefault(r.config_id, set()).add(r.item_id)
    item_sets = list(by_config.values())
    assert all(s == item_sets[0] for s in item_sets)
