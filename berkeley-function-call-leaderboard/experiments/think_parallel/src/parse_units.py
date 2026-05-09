from __future__ import annotations

import json
import re
from typing import Any


BOOL_FIELDS = {
    "can_think_now",
    "needs_prior_observation_for_arguments",
    "needs_prior_execution_before_action",
}


def _extract_json_candidate(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    raise ValueError("No JSON object or list found in response")


def parse_think_units(text: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        payload = json.loads(_extract_json_candidate(text))
    except Exception as exc:
        return [], str(exc)

    if isinstance(payload, list):
        units = payload
    elif isinstance(payload, dict):
        units = payload.get("think_units", [])
    else:
        return [], f"JSON root must be object/list, got {type(payload).__name__}"

    if not isinstance(units, list):
        return [], "think_units must be a list"

    normalized: list[dict[str, Any]] = []
    for i, unit in enumerate(units, start=1):
        if not isinstance(unit, dict):
            continue
        item = {
            "unit_id": str(unit.get("unit_id") or f"u{i}"),
            "think": str(unit.get("think") or ""),
            "action_hint": unit.get("action_hint"),
            "arguments_hint": unit.get("arguments_hint") or {},
            "dependency_claims": unit.get("dependency_claims") or {},
        }
        for field in BOOL_FIELDS:
            value = unit.get(field)
            item[field] = value if isinstance(value, bool) else None
        normalized.append(item)
    return normalized, None
