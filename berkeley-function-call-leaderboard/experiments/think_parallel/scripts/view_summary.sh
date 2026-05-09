#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: $0 OUTPUT_DIR}"

echo "== summary.md =="
sed -n '1,200p' "$RUN_DIR/summary.md"

echo
echo "== parsed units =="
jq -r '
  .question_id as $qid
  | .mode as $mode
  | if .parse_error then
      "## \($qid) \($mode)\nPARSE_ERROR: \(.parse_error)\n"
    else
      "## \($qid) \($mode)\n" +
      ([.units[]? | "- \(.unit_id) \(.action_hint // "<no-action>") :: \(.think)"] | join("\n"))
    end
' "$RUN_DIR/think_units.jsonl"

echo
echo "== matched =="
jq -r '
  .question_id as $qid
  | .mode as $mode
  | "## \($qid) \($mode)\n" +
    ([.rows[]? |
      "- \(.unit_id) -> \(.matched_hop_id // "unmatched") " +
      "[\(.gold_edge_label_summary // "no-label")] " +
      "conf=\(.mean_confidence // "null") action=\(.action_hint // "null")"
    ] | join("\n"))
' "$RUN_DIR/matched_units.jsonl"
