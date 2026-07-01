"""Load BFCL v4 multi-turn items, tools, and ground truth.

Multi-turn is structurally different from single-turn:
  - ``question`` is a list of *turns*, each a list of chat messages.
  - Tools are NOT inline; they come from per-class function-doc files, unioned
    over the item's ``involved_classes`` and filtered by ``excluded_function``.
  - Each item carries an ``initial_config`` = per-class scenario state that the
    stateful execution env loads before the rollout.

``n_turns = len(question)`` is fixed per item (the number of user messages) — it
is the item-level *trajectory length*, the variable we decompose variance across.
The seed-varying quantity is how many tool calls the model makes (``n_tool_calls``),
produced later by the rollout loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

_RAW_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval"
)
DEFAULT_DATA_DIR = Path(".cache") / "bfcl_mt"

# involved_classes name -> func-doc filename stem (snake_case). Note the gotcha:
# TwitterAPI's doc/source live under "posting_api", not "twitter_api".
CLASS_DOC_STEM = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
    "WebSearchAPI": "web_search",
}


@dataclass(frozen=True)
class MultiTurnItem:
    id: str
    turns: list[list[dict[str, Any]]]     # question: list of turns, each a list of messages
    involved_classes: list[str]
    initial_config: dict[str, Any]
    excluded_function: list[str] = field(default_factory=list)
    missed_function: dict[str, list[dict]] = field(default_factory=dict)  # turn_idx(str) -> held-out docs
    ground_truth: list[list[str]] = field(default_factory=list)  # per-turn list of call strings

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    def user_message(self, turn_idx: int) -> str:
        """The user content for a turn (turns are almost always a single user msg)."""
        return "\n".join(
            m["content"] for m in self.turns[turn_idx] if m.get("role") == "user"
        )


def _download_jsonl(url: str) -> list[dict]:
    resp = httpx.get(url, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def _cached_jsonl(url: str, cache_path: Path) -> list[dict]:
    if cache_path.exists():
        return [json.loads(l) for l in cache_path.read_text().splitlines() if l.strip()]
    rows = _download_jsonl(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(json.dumps(r) for r in rows))
    return rows


def load_func_docs(
    involved_classes: list[str],
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict[str, list[dict]]:
    """Load (and cache) the raw function-doc entries for each class, by class name."""
    data_dir = Path(data_dir)
    docs: dict[str, list[dict]] = {}
    for cls in involved_classes:
        stem = CLASS_DOC_STEM.get(cls)
        if stem is None:
            raise ValueError(f"unknown involved class {cls!r}; add it to CLASS_DOC_STEM")
        url = f"{_RAW_BASE}/data/multi_turn_func_doc/{stem}.json"
        docs[cls] = _cached_jsonl(url, data_dir / "func_doc" / f"{stem}.json")
    return docs


def tools_for_item(
    item: MultiTurnItem,
    turn_idx: int | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> list[dict]:
    """Build the OpenAI-format tool list offered for ``item`` (optionally at a turn).

    Union the func docs across ``involved_classes``, drop ``excluded_function``,
    and — when ``turn_idx`` is given — append any ``missed_function`` tools that
    unlock at that turn. Non-standard ``"dict"`` types are rewritten to ``"object"``
    and the informational ``response`` field is dropped.
    """
    from agentstat.harness.bfcl import _sanitize_schema  # reuse the type sanitizer

    docs_by_class = load_func_docs(item.involved_classes, data_dir=data_dir)
    excluded = set(item.excluded_function)

    raw_docs: list[dict] = []
    for cls in item.involved_classes:
        raw_docs.extend(docs_by_class[cls])

    # Unlock held-out tools up to and including this turn.
    if turn_idx is not None:
        for t_str, held in item.missed_function.items():
            if int(t_str) <= turn_idx:
                raw_docs.extend(held)

    tools = []
    for fn in raw_docs:
        if fn["name"] in excluded:
            continue
        params = _sanitize_schema(fn.get("parameters", {}))
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return tools


def load_multi_turn(
    category: str = "multi_turn_base",
    limit: int | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    version: str = "BFCL_v4",
) -> list[MultiTurnItem]:
    """Fetch (and disk-cache) a multi-turn category joined with ground truth."""
    filename = f"{version}_{category}.json"
    data_dir = Path(data_dir)

    questions = _cached_jsonl(
        f"{_RAW_BASE}/data/{filename}", data_dir / filename
    )
    answers = _cached_jsonl(
        f"{_RAW_BASE}/data/possible_answer/{filename}",
        data_dir / f"possible_answer_{filename}",
    )
    gt_by_id = {a["id"]: a["ground_truth"] for a in answers}

    items: list[MultiTurnItem] = []
    for q in questions:
        items.append(
            MultiTurnItem(
                id=q["id"],
                turns=q["question"],
                involved_classes=q["involved_classes"],
                initial_config=q.get("initial_config", {}),
                excluded_function=q.get("excluded_function", []),
                missed_function=q.get("missed_function", {}),
                ground_truth=gt_by_id.get(q["id"], []),
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items
