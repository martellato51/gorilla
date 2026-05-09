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
git clone git@github.com:martellato51/DiffuAgent.git
cd DiffuAgent
mkdir -p unified_envs
cd unified_envs
git clone --branch diffuagent-bfcl git@github.com:martellato51/gorilla.git
cd gorilla/berkeley-function-call-leaderboard
```

Expected remotes inside `unified_envs/gorilla`:

```bash
origin   git@github.com:martellato51/gorilla.git
upstream https://github.com/ShishirPatil/gorilla
```

The branch to use is:

```bash
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
export QWEN_MODEL_PATH=/path/to/model/Qwen3-8B
export LLADA_MODEL_PATH=/path/to/model/LLaDA-8B-Instruct
export FAST_DLLM_LLADA_PATH=/path/to/Fast-dLLM/v1/llada
```

## Environment Setup

Portable Slurm template:

```bash
cd "$BFCL_ROOT"
sbatch jobs/setup_bfcl_env.sbatch
```

On the current server, the latest successful local setup instead uses the
shared `qwen3` and `llada8b` conda environments plus:

```bash
cd /data/home/martellato41/research
sbatch setup_bfcl.job
```

That local job installs BFCL editable, installs optional vLLM dependencies when
requested, installs Fast-dLLM v1 requirements, and runs `register_backbone.py`.

## Model Registration

For the backbone experiments, the required registration script is:

```bash
python register_backbone.py
```

It registers:

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
TEST_CATEGORY=simple_python sbatch jobs/run_backbone_eval.sbatch
```

Current server jobs:

```bash
cd /data/home/martellato41/research

sbatch run_bfcl_backbone.job    # Qwen3-8B then LLaDA-8B
sbatch run_bfcl_llada.job       # LLaDA-only
sbatch run_bfcl_qwen3.job       # Qwen-only, expects vLLM endpoint
```

Detailed notes are in `README_BFCL_BACKBONE.md`.

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
