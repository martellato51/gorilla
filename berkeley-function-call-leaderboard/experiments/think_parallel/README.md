# BFCL Think Parallel v0

This experiment runs an offline full-trajectory planning diagnostic with BFCL
multi-turn samples and LLaDA. It reuses the existing BFCL tool-schema prompt
construction and the local LLaDA environment, but it does not run BFCL official
action evaluation.

Two prompt modes are supported:

- `goal_tools`: full user goal plus BFCL tool schemas.
- `goal_init_tools`: full user goal plus a compact `initial_config` summary plus
  BFCL tool schemas.

`bfcl_graphs.jsonl` and `bfcl_edges.jsonl` are used only after generation for
matching and confidence analysis. Gold hops/edges are not placed in the prompt.

## Run

Smoke test:

```bash
RESEARCH_ROOT=/path/to/research \
IDS=multi_turn_base_46 MODE=both sbatch experiments/think_parallel/scripts/run_llada_discovery.job
```

Mixed-label discovery set:

```bash
IDS=multi_turn_base_66,multi_turn_base_187,multi_turn_base_193,multi_turn_base_121 \
MODE=both \
sbatch experiments/think_parallel/scripts/run_llada_discovery.job
```

Portable path knobs:

- `RESEARCH_ROOT`: parent directory containing `DiffuAgent/`, `Fast-dLLM/`,
  `model/`, and optionally `ParallelAgent/`.
- `BFCL_ROOT`: override the BFCL checkout path directly.
- `LLADA_MODEL_PATH`: override the LLaDA-8B-Instruct snapshot path.
- `FAST_DLLM_LLADA_PATH`: override the Fast-dLLM v1 LLaDA code path.
- `GRAPHS` / `EDGES`: override BFCL graph annotation files.

OOM-free candidate set from the previous BFCL LLaDA run:

```bash
IDS=multi_turn_base_1,multi_turn_base_3,multi_turn_base_6,multi_turn_base_9,multi_turn_base_10,multi_turn_base_12,multi_turn_base_16,multi_turn_base_25,multi_turn_base_26,multi_turn_base_29,multi_turn_base_37,multi_turn_base_38,multi_turn_base_39,multi_turn_base_41,multi_turn_base_46 \
MODE=both \
sbatch experiments/think_parallel/scripts/run_llada_discovery.job
```

## Outputs

Each run writes:

- `raw_generations.jsonl`: prompt, raw response, latency, token counts.
- `think_units.jsonl`: parsed JSON think units.
- `token_confidence.jsonl`: token text, chosen probability, commit step, position.
- `matched_units.jsonl`: generated unit to gold hop matching and edge labels.
- `metrics.json`: mode-level match and confidence summary.
- `summary.md`: compact human-readable summary.

View a run:

```bash
experiments/think_parallel/scripts/view_summary.sh experiments/think_parallel/outputs/<run_name>
```

## Interpretation

This is not online BFCL multi-turn inference. It is an offline full-trajectory
planning setting designed to test whether LLaDA can emit useful action-level
think units from the goal, and whether denoising confidence separates
Independent, Weak-A, Weak-D, and Strong dependency labels.
