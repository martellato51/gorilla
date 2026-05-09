# Installation Guide for DiffuAgent BFCL Backbone Evaluation

This guide documents the reproducible BFCL v4 setup used for DiffuAgent
backbone evaluation. The maintained code lives in a Gorilla fork branch rather
than in the top-level DiffuAgent repository.

## Prerequisites

- Git (version 2.25+ for sparse checkout support)
- Python 3.10+
- No pip installation required (code-only setup)

## Repository Layout

```bash
git clone --branch my-change --recurse-submodules git@github.com:martellato51/DiffuAgent.git
cd DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard
```

If the server does not have a GitHub SSH key, configure SSH first or replace the
clone/submodule URLs with URLs that the server can authenticate against.

Expected remotes inside `unified_envs/gorilla`:

```bash
origin   git@github.com:martellato51/gorilla.git
upstream https://github.com/ShishirPatil/gorilla
```

Submodules are pinned to a commit. If you need to edit Gorilla files, checkout
the working branch inside the submodule:

```bash
cd /path/to/DiffuAgent/unified_envs/gorilla
git checkout diffuagent-bfcl
```

This branch already contains the DiffuAgent BFCL additions. Do not copy files
from the older `DiffuAgent/BFCL` tree on top of it.

## External Checkouts and Models

Prepare these sibling paths:

```text
research/
├── DiffuAgent/
│   └── unified_envs/gorilla/berkeley-function-call-leaderboard/
├── Fast-dLLM/
│   └── v1/llada/
└── model/
    ├── Qwen3-8B/
    └── LLaDA-8B-Instruct/
```

Important environment variables:

```bash
export BFCL_ROOT=/path/to/DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard
export RESEARCH_ROOT=/path/to/research
export QWEN_MODEL_PATH=/path/to/model/Qwen3-8B
export LLADA_MODEL_PATH=/path/to/model/LLaDA-8B-Instruct
export FAST_DLLM_LLADA_PATH=/path/to/Fast-dLLM/v1/llada
```

## Environment Setup

If this BFCL checkout was obtained through the wrapper DiffuAgent repository,
the conda yml files are available at:

```text
../../../envs/qwen3.yml
../../../envs/llada8b.yml
../../../envs/llada2.1.yml
```

Create the BFCL backbone environments first:

```bash
conda env create -f ../../../envs/qwen3.yml
conda env create -f ../../../envs/llada8b.yml
```

Portable Slurm template. This creates or updates two conda environments, matching
the setup that has been used successfully on the current server:

- `qwen3`: Qwen3-8B + vLLM/OpenAI-compatible serving.
- `llada8b`: LLaDA-8B-Instruct + Fast-dLLM v1 local inference.

```bash
cd "$BFCL_ROOT"
sbatch jobs/setup_bfcl_env.sbatch
```

Override env names if needed:

```bash
QWEN_ENV_NAME=my_qwen_env LLADA_ENV_NAME=my_llada_env sbatch jobs/setup_bfcl_env.sbatch
```

The setup job installs BFCL editable in both environments, installs vLLM extras
for `qwen3`, installs Fast-dLLM v1 requirements for `llada8b`, and runs
`register_backbone.py` in both.

## Model Registration

For the backbone experiments, `register_backbone.py` registers:

```text
backbone/qwen3-8b -> LLMHandler
backbone/llada    -> LocalDLLMHandler
```

`register_diffuagent.py` is only needed for the full selector/editor model
matrix. It is not required for pure backbone comparisons and may import extra
optional dependencies.

## Running Backbone Evaluation

Portable tracked template:

```bash
cd "$BFCL_ROOT"
TEST_CATEGORIES=simple_python sbatch jobs/run_backbone_eval.sbatch
```

This job starts Qwen3-8B through vLLM in the `qwen3` env, evaluates
`backbone/qwen3-8b`, then switches to the `llada8b` env and evaluates
`backbone/llada` against the same BFCL subset.

```bash
QWEN_ENV_NAME=my_qwen_env \
LLADA_ENV_NAME=my_llada_env \
TEST_CATEGORIES=simple_python,multiple \
sbatch jobs/run_backbone_eval.sbatch
```

Detailed notes are in `README_BFCL_BACKBONE.md`.

On the current server, older hand-tuned Slurm jobs are kept outside this repo in
`/data/home/martellato41/research/jobs_bfcl/`. They are local records; use
`jobs/*.sbatch` in this repo as the portable starting point.

## Output Hygiene

Generated outputs are intentionally ignored by git:

```text
logs/
logger/
result/
score/
result_runs/
subset/
subset_runs/
experiments/**/outputs/
```
