from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def find_unit_span(response: str, think: str) -> tuple[int | None, int | None]:
    if not think:
        return None, None
    pos = response.find(think)
    if pos == -1:
        compact = " ".join(think.split())
        response_compact = " ".join(response.split())
        pos = response_compact.find(compact)
        if pos == -1:
            return None, None
        return None, None
    return pos, pos + len(think)


def token_stats_for_span(
    trace_tokens: list[dict[str, Any]],
    start: int | None,
    end: int | None,
) -> dict[str, Any]:
    if start is None or end is None:
        return {
            "span_found": False,
            "n_tokens": 0,
            "mean_confidence": None,
            "min_confidence": None,
            "bottom_10p_mean_confidence": None,
            "commit_step_mean": None,
        }
    selected = [
        token
        for token in trace_tokens
        if token.get("confidence") is not None
        and overlap(start, end, token.get("char_start", -1), token.get("char_end", -1))
    ]
    if not selected:
        return {
            "span_found": True,
            "n_tokens": 0,
            "mean_confidence": None,
            "min_confidence": None,
            "bottom_10p_mean_confidence": None,
            "commit_step_mean": None,
        }
    confs = [float(token["confidence"]) for token in selected]
    steps = [int(token["commit_step"]) for token in selected if token.get("commit_step") is not None]
    sorted_confs = sorted(confs)
    bottom_n = max(1, int(len(sorted_confs) * 0.1))
    return {
        "span_found": True,
        "n_tokens": len(selected),
        "mean_confidence": mean(confs),
        "min_confidence": min(confs),
        "bottom_10p_mean_confidence": mean(sorted_confs[:bottom_n]),
        "commit_step_mean": mean(steps) if steps else None,
    }


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"modes": {}, "overall": {}}
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)

    for mode, items in sorted(by_mode.items()):
        matched = [item for item in items if item.get("matched_hop_id")]
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in matched:
            by_label[item.get("gold_edge_label_summary") or "Unknown"].append(item)

        label_stats = {}
        for label, label_items in sorted(by_label.items()):
            confs = [
                item.get("mean_confidence")
                for item in label_items
                if item.get("mean_confidence") is not None
            ]
            label_stats[label] = {
                "n": len(label_items),
                "n_with_confidence": len(confs),
                "mean_confidence": mean(confs) if confs else None,
            }

        out["modes"][mode] = {
            "n_units": len(items),
            "n_matched": len(matched),
            "n_unmatched": len(items) - len(matched),
            "match_rate": len(matched) / len(items) if items else 0.0,
            "labels": label_stats,
            "ranking": {
                "weak_a_above_strong": pairwise_above(
                    by_label.get("Weak-A", []), by_label.get("Strong", [])
                ),
                "independent_above_strong": pairwise_above(
                    by_label.get("Independent", []), by_label.get("Strong", [])
                ),
            },
        }
    out["overall"]["n_units"] = len(rows)
    out["overall"]["n_matched"] = sum(1 for row in rows if row.get("matched_hop_id"))
    return out


def pairwise_above(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float | None:
    left_confs = [
        item.get("mean_confidence")
        for item in left
        if item.get("mean_confidence") is not None
    ]
    right_confs = [
        item.get("mean_confidence")
        for item in right
        if item.get("mean_confidence") is not None
    ]
    if not left_confs or not right_confs:
        return None
    wins = 0.0
    total = 0
    for left_conf in left_confs:
        for right_conf in right_confs:
            total += 1
            if left_conf > right_conf:
                wins += 1.0
            elif left_conf == right_conf:
                wins += 0.5
    return wins / total if total else None


def render_summary(metrics: dict[str, Any]) -> str:
    lines = ["# BFCL Think Parallel v0 Summary", ""]
    for mode, stats in metrics.get("modes", {}).items():
        lines.append(f"## {mode}")
        lines.append(f"- units: {stats['n_units']}")
        lines.append(f"- matched: {stats['n_matched']}")
        lines.append(f"- match_rate: {stats['match_rate']:.3f}")
        for label, label_stats in stats.get("labels", {}).items():
            conf = label_stats.get("mean_confidence")
            conf_text = "null" if conf is None else f"{conf:.4f}"
            lines.append(
                f"- {label}: n={label_stats['n']}, "
                f"n_with_confidence={label_stats['n_with_confidence']}, "
                f"mean_confidence={conf_text}"
            )
        for name, value in stats.get("ranking", {}).items():
            value_text = "null" if value is None else f"{value:.3f}"
            lines.append(f"- ranking.{name}: {value_text}")
        lines.append("")
    return "\n".join(lines)
