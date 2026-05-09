from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_config" not in obj:
                records.append(obj)
    return records


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return read_jsonl(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    raise TypeError(f"Unsupported JSON payload in {path}")


def join_user_goal(question: list[list[dict[str, str]]]) -> str:
    turns: list[str] = []
    for i, turn in enumerate(question, start=1):
        user_parts = [
            msg.get("content", "")
            for msg in turn
            if msg.get("role") == "user" and msg.get("content")
        ]
        if user_parts:
            turns.append(f"[turn {i}] " + "\n".join(user_parts))
    return "\n".join(turns)


def _compact_file_tree(value: Any, max_file_chars: int) -> Any:
    if isinstance(value, dict):
        if value.get("type") == "file" and "content" in value:
            content = str(value["content"])
            compact = dict(value)
            if len(content) > max_file_chars:
                compact["content"] = content[:max_file_chars] + "...<truncated>"
            return compact
        return {k: _compact_file_tree(v, max_file_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [_compact_file_tree(v, max_file_chars) for v in value]
    return value


def summarize_initial_config(initial_config: dict[str, Any], max_chars: int = 5000) -> str:
    compact = _compact_file_tree(initial_config, max_file_chars=300)
    text = json.dumps(compact, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def category_to_data_name(category: str) -> str:
    return "BFCL_v4_" + category


def load_bfcl_records(
    bfcl_root: Path,
    category: str,
    ids: list[str] | None,
) -> list[dict[str, Any]]:
    source = f"bfcl_eval.utils.load_dataset_entry({category})"
    try:
        from bfcl_eval.utils import load_dataset_entry

        records = load_dataset_entry(category, include_language_specific_hint=False)
    except Exception:
        data_name = category_to_data_name(category)
        data_path = bfcl_root / "bfcl_eval" / "data" / f"{data_name}.json"
        source = str(data_path)
        records = load_json_or_jsonl(data_path)
    if ids:
        wanted = set(ids)
        records = [record for record in records if record.get("id") in wanted]
        found = {record.get("id") for record in records}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"Missing BFCL ids in {source}: {missing}")
        order = {sample_id: i for i, sample_id in enumerate(ids)}
        records.sort(key=lambda record: order[record["id"]])
    return records


def load_graphs(path: Path) -> dict[str, dict[str, Any]]:
    return {record["question_id"]: record for record in read_jsonl(path)}


def load_edges(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in read_jsonl(path):
        grouped.setdefault(record["question_id"], []).append(record)
    return grouped
