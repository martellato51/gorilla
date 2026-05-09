from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any


LABEL_PRIORITY = {"Strong": 4, "Weak-D": 3, "Weak-A": 2, "Independent": 1}


def normalize_tool_name(name: Any) -> str:
    text = str(name or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return re.sub(r"[^a-z0-9_]", "", text.lower())


def hop_tool_name(hop: dict[str, Any]) -> str:
    question = hop.get("question_raw") or hop.get("question") or ""
    return question.split("(", 1)[0].strip()


def hop_args_text(hop: dict[str, Any]) -> str:
    question = hop.get("question_raw") or hop.get("question") or ""
    match = re.search(r"\((.*)\)\s*$", question)
    return match.group(1) if match else ""


def edge_labels_by_dst(edges: list[dict[str, Any]]) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        dst = edge.get("dst")
        if dst:
            labels[dst].append(edge.get("tier2") or edge.get("tier1") or "Strong")
    return labels


def summarize_labels(labels: list[str]) -> str:
    if not labels:
        return "Independent"
    return max(labels, key=lambda label: LABEL_PRIORITY.get(label, 0))


def _args_overlap(unit_args: Any, args_text: str) -> int:
    if not unit_args:
        return 0
    try:
        unit_text = json.dumps(unit_args, ensure_ascii=False).lower()
    except TypeError:
        unit_text = str(unit_args).lower()
    tokens = {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", unit_text)
        if len(token) > 1
    }
    gold = args_text.lower()
    return sum(1 for token in tokens if token in gold)


def match_units_to_gold(
    units: list[dict[str, Any]],
    graph: dict[str, Any] | None,
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not graph:
        return [
            {
                **unit,
                "matched_hop_id": None,
                "matched_tool": None,
                "gold_edge_labels": [],
                "gold_edge_label_summary": None,
                "match_score": 0,
            }
            for unit in units
        ]

    hops = graph.get("hops", [])
    labels_by_dst = edge_labels_by_dst(edges)
    used: set[str] = set()
    matched: list[dict[str, Any]] = []

    for unit in units:
        unit_tool = normalize_tool_name(unit.get("action_hint"))
        best: tuple[int, dict[str, Any] | None] = (0, None)
        for hop in hops:
            hop_id = hop.get("id")
            if hop_id in used:
                continue
            hop_tool = normalize_tool_name(hop_tool_name(hop))
            score = 0
            if unit_tool and hop_tool and unit_tool == hop_tool:
                score += 100
            elif unit_tool and hop_tool and (unit_tool in hop_tool or hop_tool in unit_tool):
                score += 60
            score += min(20, _args_overlap(unit.get("arguments_hint"), hop_args_text(hop)))
            if score > best[0]:
                best = (score, hop)

        score, hop = best
        if hop is None or score < 60:
            matched.append(
                {
                    **unit,
                    "matched_hop_id": None,
                    "matched_tool": None,
                    "gold_edge_labels": [],
                    "gold_edge_label_summary": None,
                    "match_score": score,
                }
            )
            continue

        hop_id = hop["id"]
        used.add(hop_id)
        labels = labels_by_dst.get(hop_id, [])
        matched.append(
            {
                **unit,
                "matched_hop_id": hop_id,
                "matched_tool": hop_tool_name(hop),
                "gold_question": hop.get("question_raw") or hop.get("question"),
                "gold_edge_labels": labels,
                "gold_edge_label_summary": summarize_labels(labels),
                "match_score": score,
            }
        )
    return matched
