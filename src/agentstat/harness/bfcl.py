"""Load and score the BFCL "simple" (single-turn, AST) category.

Data lives in the gorilla repo as JSONL (despite the .json extension). We fetch
the question file and the matching possible-answer file, join them by ``id``, and
score model predictions by AST-matching against the acceptable-value lists — the
same check BFCL uses for its single-turn categories.

We target the AST categories only (simple / multiple / parallel), which are
scored by structural match, not by executing tool calls. Multi-turn categories
require a stateful execution environment and are out of scope here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

_RAW_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)
DEFAULT_DATA_DIR = Path(".cache") / "bfcl"


@dataclass(frozen=True)
class BFCLItem:
    id: str
    prompt: str                      # flattened user question
    functions: list[dict[str, Any]]  # tool definitions offered to the model
    ground_truth: list[dict[str, Any]]  # [{func_name: {arg: [acceptable, ...]}}]


def _download_jsonl(filename: str, subdir: str = "") -> list[dict]:
    url = f"{_RAW_BASE}/{subdir + '/' if subdir else ''}{filename}"
    resp = httpx.get(url, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def load_bfcl_simple(
    category: str = "simple_python",
    limit: int | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    version: str = "BFCL_v4",
) -> list[BFCLItem]:
    """Fetch (and disk-cache) a BFCL AST category, joined with ground truth.

    ``category`` is the suffix, e.g. ``"simple_python"`` -> ``BFCL_v4_simple_python.json``.
    Files are cached under ``data_dir`` so subsequent loads are offline.
    """
    filename = f"{version}_{category}.json"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    q_path = data_dir / filename
    a_path = data_dir / f"possible_answer_{filename}"

    if q_path.exists() and a_path.exists():
        questions = [json.loads(l) for l in q_path.read_text().splitlines() if l.strip()]
        answers = [json.loads(l) for l in a_path.read_text().splitlines() if l.strip()]
    else:
        questions = _download_jsonl(filename)
        answers = _download_jsonl(filename, subdir="possible_answer")
        q_path.write_text("\n".join(json.dumps(q) for q in questions))
        a_path.write_text("\n".join(json.dumps(a) for a in answers))

    gt_by_id = {a["id"]: a["ground_truth"] for a in answers}

    items: list[BFCLItem] = []
    for q in questions:
        # question is [turn][msg]; simple is single-turn single-user-message.
        turns = q["question"]
        user_msgs = [
            m["content"]
            for turn in turns
            for m in turn
            if m.get("role") == "user"
        ]
        items.append(
            BFCLItem(
                id=q["id"],
                prompt="\n".join(user_msgs),
                functions=q["function"],
                ground_truth=gt_by_id.get(q["id"], []),
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def to_openai_tools(functions: list[dict[str, Any]]) -> list[dict]:
    """Convert BFCL function defs to OpenAI tool-call format.

    BFCL uses a non-standard ``"type": "dict"`` for object params; OpenAI expects
    ``"object"``. We rewrite that at the top level of each parameter schema.
    """
    tools = []
    for fn in functions:
        params = json.loads(json.dumps(fn.get("parameters", {})))  # deep copy
        if params.get("type") == "dict":
            params["type"] = "object"
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
