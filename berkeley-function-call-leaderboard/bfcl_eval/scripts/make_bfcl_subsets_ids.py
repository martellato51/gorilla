#!/usr/bin/env python
"""Create the BFCL subset ID file used by the backbone jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional


DEFAULT_BFCL_ROOT = Path(
    os.getenv(
        "BFCL_PROJECT_ROOT",
        "/home/ilju/research/bfcl_v3_ea13468/"
        "berkeley-function-call-leaderboard",
    )
)

def env_limit(name: str, default: Optional[int]) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    if value.lower() in {"all", "none", "full"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, 'all', 'none', or 'full'")
    return parsed


def env_ids(category: str) -> Optional[list[str]]:
    env_name = f"BFCL_{category.upper()}_IDS"
    value = os.getenv(env_name)
    if value is None or value.strip() == "":
        return None
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids:
        raise ValueError(f"{env_name} must contain at least one comma-separated ID")
    return ids


def load_whitelist(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


SUBSET_LIMITS = {
    "simple": 50,
    "multiple": 50,
    "parallel": 50,
    "parallel_multiple": 50,
    "java": 50,
    "javascript": 50,
    "live_simple": 50,
    "live_multiple": 50,
    "live_parallel": None,
    "live_parallel_multiple": None,
    "multi_turn_base": env_limit("BFCL_MULTI_TURN_BASE_LIMIT", 50),
    "multi_turn_miss_func": env_limit("BFCL_MULTI_TURN_MISS_FUNC_LIMIT", 50),
    "multi_turn_miss_param": env_limit("BFCL_MULTI_TURN_MISS_PARAM_LIMIT", 50),
    "multi_turn_long_context": env_limit("BFCL_MULTI_TURN_LONG_CONTEXT_LIMIT", 10),
}


def parse_categories(raw_categories: str) -> list[str]:
    if not raw_categories or raw_categories == "all":
        return list(SUBSET_LIMITS.keys())
    categories = [item.strip() for item in raw_categories.split(",") if item.strip()]
    invalid = [category for category in categories if category not in SUBSET_LIMITS]
    if invalid:
        raise ValueError(f"Unknown BFCL subset categories: {', '.join(invalid)}")
    return categories


def read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            ids.append(json.loads(line)["id"])
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfcl-root", default=str(DEFAULT_BFCL_ROOT))
    parser.add_argument("--output", default="test_case_ids_to_generate.json")
    parser.add_argument(
        "--categories",
        default="all",
        help="Comma-separated BFCL categories to include in the subset ID file.",
    )
    parser.add_argument(
        "--whitelist",
        default="test_case_ids_to_generate_v3.json",
        help="Whitelist JSON file to use for ID selection (relative to bfcl-root).",
    )
    args = parser.parse_args()

    bfcl_root = Path(args.bfcl_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = bfcl_root / output

    whitelist_path = Path(args.whitelist)
    if not whitelist_path.is_absolute():
        whitelist_path = bfcl_root / whitelist_path
    whitelist = load_whitelist(whitelist_path)
    if whitelist:
        print(f"Loaded whitelist: {whitelist_path} ({len(whitelist)} categories)")
    else:
        print(f"No whitelist found at {whitelist_path}, falling back to ids[:limit]")

    selected_categories = parse_categories(args.categories)
    subset: dict[str, list[str]] = {}
    total = 0
    print("BFCL subset:")
    for category in selected_categories:
        limit = SUBSET_LIMITS[category]
        data_file = bfcl_root / "bfcl_eval" / "data" / f"BFCL_v3_{category}.json"
        all_ids = read_ids(data_file)
        all_ids_set = set(all_ids)

        # Priority: env var IDs > whitelist > ids[:limit]
        requested_ids = env_ids(category)
        if requested_ids is not None:
            missing_ids = sorted(set(requested_ids) - all_ids_set)
            if missing_ids:
                raise ValueError(
                    f"Unknown IDs for {category}: {', '.join(missing_ids)}"
                )
            selected = requested_ids
            source = "env"
        else:
            whitelist_ids = whitelist.get(category)
            if whitelist_ids is None:
                raise ValueError(
                    f"Category '{category}' not found in whitelist {whitelist_path}"
                )
            missing_ids = sorted(set(whitelist_ids) - all_ids_set)
            if missing_ids:
                raise ValueError(
                    f"Whitelist IDs not found in data for {category}: {', '.join(missing_ids)}"
                )
            selected = whitelist_ids
            source = "whitelist"

        subset[category] = selected
        total += len(selected)
        print(f"  {category:24s} selected={len(selected):3d} source={source}")

    output.write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")
    print(f"Total selected: {total}")
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
