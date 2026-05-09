from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from src.bfcl_data import load_bfcl_records, load_edges, load_graphs
    from src.llada_trace import TraceableLLaDABackend
    from src.match_gold import match_units_to_gold
    from src.metrics import (
        find_unit_span,
        render_summary,
        summarize_metrics,
        token_stats_for_span,
    )
    from src.parse_units import parse_think_units
    from src.prompt_builder import build_prompt
else:
    from .bfcl_data import load_bfcl_records, load_edges, load_graphs
    from .llada_trace import TraceableLLaDABackend
    from .match_gold import match_units_to_gold
    from .metrics import (
        find_unit_span,
        render_summary,
        summarize_metrics,
        token_stats_for_span,
    )
    from .parse_units import parse_think_units
    from .prompt_builder import build_prompt


def split_ids(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BFCL LLaDA think-unit discovery")
    parser.add_argument("--ids", default=None, help="Comma-separated BFCL ids")
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument(
        "--mode",
        default="both",
        choices=["goal_tools", "goal_init_tools", "both"],
    )
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bfcl-root", default=os.getcwd())
    parser.add_argument("--dtype", default=os.getenv("LLADA_DTYPE", "bfloat16"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bfcl_root = Path(args.bfcl_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw_generations.jsonl"
    units_path = output_dir / "think_units.jsonl"
    conf_path = output_dir / "token_confidence.jsonl"
    matched_path = output_dir / "matched_units.jsonl"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    for path in [raw_path, units_path, conf_path, matched_path]:
        path.unlink(missing_ok=True)

    ids = split_ids(args.ids)
    modes = ["goal_tools", "goal_init_tools"] if args.mode == "both" else [args.mode]
    records = load_bfcl_records(bfcl_root, args.category, ids)
    graphs = load_graphs(Path(args.graphs))
    edges = load_edges(Path(args.edges))

    print(f"Loading LLaDA backend for {len(records)} samples, modes={modes}")
    backend = TraceableLLaDABackend(dtype=args.dtype)
    all_matched_rows: list[dict[str, Any]] = []

    for sample in records:
        sample_id = sample["id"]
        graph = graphs.get(sample_id)
        sample_edges = edges.get(sample_id, [])
        for mode in modes:
            print(f"Running {sample_id} mode={mode}")
            messages, functions, goal = build_prompt(sample, mode, args.category)
            result = backend.generate(messages)
            raw_record = {
                "question_id": sample_id,
                "mode": mode,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "goal": goal,
                "prompt": messages,
                "n_functions": len(functions),
                "raw_response": result.text,
                "latency": result.latency,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "generation_config": result.generation_config,
            }
            write_jsonl(raw_path, raw_record)

            units, parse_error = parse_think_units(result.text)
            write_jsonl(
                units_path,
                {
                    "question_id": sample_id,
                    "mode": mode,
                    "parse_error": parse_error,
                    "units": units,
                    "raw_response": result.text,
                },
            )

            for token in result.trace_tokens:
                write_jsonl(
                    conf_path,
                    {
                        "question_id": sample_id,
                        "mode": mode,
                        **token,
                    },
                )

            matched = match_units_to_gold(units, graph, sample_edges)
            enriched_rows = []
            for unit in matched:
                start, end = find_unit_span(result.text, unit.get("think", ""))
                stats = token_stats_for_span(result.trace_tokens, start, end)
                row = {
                    "question_id": sample_id,
                    "mode": mode,
                    **unit,
                    "think_char_start": start,
                    "think_char_end": end,
                    **stats,
                }
                enriched_rows.append(row)
                all_matched_rows.append(row)
            write_jsonl(
                matched_path,
                {
                    "question_id": sample_id,
                    "mode": mode,
                    "rows": enriched_rows,
                },
            )

    metrics = summarize_metrics(all_matched_rows)
    metrics["run"] = {
        "category": args.category,
        "ids": [record["id"] for record in records],
        "modes": modes,
        "graphs": str(Path(args.graphs).resolve()),
        "edges": str(Path(args.edges).resolve()),
        "offline_full_trajectory_planning": True,
        "note": "Graph hops and edges were used only for post-hoc matching/evaluation, not prompt input.",
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(render_summary(metrics), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
