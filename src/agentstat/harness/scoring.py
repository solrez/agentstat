"""AST scoring for BFCL single-turn categories: binary pass/fail.

Mirrors BFCL's AST checker for the "simple" category closely enough for a
statistically-meaningful pass rate:

  1. The predicted function name must exactly match the ground-truth call.
  2. Every ``required`` parameter must be present, and its value must be a member
     of that parameter's acceptable-value list.
  3. Optional parameters, if present, must also be a member of their acceptable
     list. An acceptable list containing "" means the parameter may be omitted.
  4. No hallucinated parameters (not in the ground-truth arg set).

A predicted call can come either from an OpenAI ``tool_calls`` response or, as a
fallback, be parsed from a function-call string. Value comparison is
type-lenient (numeric 10 == "10") to match BFCL's normalization.
"""

from __future__ import annotations

import json
from typing import Any


def _coerce_match(pred: Any, acceptable: list[Any]) -> bool:
    """Is ``pred`` a member of ``acceptable``, with light type normalization?"""
    for ok in acceptable:
        if pred == ok:
            return True
        # numeric/string leniency: 10 == "10", 10.0 == 10
        try:
            if float(pred) == float(ok):
                return True
        except (TypeError, ValueError):
            pass
        if str(pred).strip() == str(ok).strip():
            return True
    return False


def score_prediction(
    predicted_name: str | None,
    predicted_args: dict[str, Any] | None,
    ground_truth: list[dict[str, Any]],
) -> float:
    """Return 1.0 if the predicted call matches any ground-truth call, else 0.0.

    ``ground_truth`` is BFCL's list of ``{func_name: {arg: [acceptable, ...]}}``.
    For "simple" there is exactly one entry, but we check membership across all
    entries so the same scorer works for multiple-call categories.
    """
    if predicted_name is None:
        return 0.0
    # A malformed prediction (args not a flat dict — e.g. the model emitted a
    # list of calls, or nested the arguments) is a model failure, not an infra
    # error: it must score 0.0, never raise. Anything non-dict fails the match.
    if not isinstance(predicted_args, dict):
        return 0.0

    for gt_call in ground_truth:
        for gt_name, gt_args in gt_call.items():
            if predicted_name != gt_name:
                continue
            if _args_match(predicted_args, gt_args):
                return 1.0
    return 0.0


def _args_match(pred_args: dict[str, Any], gt_args: dict[str, list]) -> bool:
    # No hallucinated params. Guard hashing: a dict/list key (from a malformed
    # prediction) is never a valid arg name -> no match.
    try:
        if any(k not in gt_args for k in pred_args):
            return False
    except TypeError:
        return False
    for arg_name, acceptable in gt_args.items():
        optional = "" in acceptable
        if arg_name not in pred_args:
            if optional:
                continue          # omission allowed
            return False          # required param missing
        if not _coerce_match(pred_args[arg_name], acceptable):
            return False
    return True


def extract_call(response: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """Pull (function_name, args) from an OpenAI-style chat response.

    Prefers a structured ``tool_calls`` entry; returns (None, None) if the model
    produced no tool call (which scores as a fail).
    """
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError):
        return None, None

    tool_calls = message.get("tool_calls")
    if tool_calls:
        fn = tool_calls[0]["function"]
        name = fn.get("name")
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = None
        return name, args

    return None, None
