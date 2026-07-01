"""The data contract everything in AgentStat consumes.

A benchmark run is a ``list[EvalResult]``. Bootstrap resamples over this list,
variance decomposition groups by the factor columns, and ranking compares
aggregates per ``config_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class EvalResult:
    """A single scored evaluation trial.

    One row = one (config, item, seed) evaluation. The factor columns
    (``seed``, ``n_turns``, ``n_tool_calls``, ``prompt_variant``) are what the
    variance decomposition partitions over; leave them ``None`` when they don't
    apply (e.g. single-turn evals have no ``n_turns``).
    """

    config_id: str          # which agent/model config produced this
    item_id: str            # which benchmark item
    score: float            # 0/1 for pass-fail, or continuous
    seed: int | None = None
    n_turns: int | None = None          # agent: trajectory length
    n_tool_calls: int | None = None
    prompt_variant: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalResult":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
