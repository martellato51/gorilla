#!/usr/bin/env bash
# Server-specific paths and environment settings.
# On a new server, only this file needs to be edited.
#
# Checklist:
#   1. BFCL_ROOT        — where this repo is cloned
#   2. CONDA_SH         — path to conda.sh (miniconda or anaconda)
#   3. QWEN_PYTHON      — python binary inside the qwen3 conda env
#   4. LLADA_PYTHON     — python binary inside the llada8b conda env
#   5. MAIN_AGENT_MODEL_PATH — Qwen3-8B model weight directory
#   6. LLADA_MODEL_PATH      — LLaDA-8B-Instruct model weight directory
#   7. LLADA21_MODEL_PATH    — LLaDA2.1-mini model weight directory
#   8. FAST_DLLM_LLADA_PATH  — Fast-dLLM v1/llada code directory
#   9. CUDA_VISIBLE_DEVICES  — GPUs to use (e.g. "0,1" or "1,2,3")

# --- Paths ---
export BFCL_ROOT="${BFCL_ROOT:-/home/ilju/research/DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard}"
export RESEARCH_ROOT="${RESEARCH_ROOT:-/home/ilju/research}"
export FAST_DLLM_LLADA_PATH="${FAST_DLLM_LLADA_PATH:-${RESEARCH_ROOT}/Fast-dLLM/v1/llada}"

# --- Conda envs ---
export QWEN_ENV_NAME="${QWEN_ENV_NAME:-qwen3}"
export LLADA_ENV_NAME="${LLADA_ENV_NAME:-llada8b}"
export LLADA21_ENV_NAME="${LLADA21_ENV_NAME:-llada2.1}"
export QWEN_PYTHON="${QWEN_PYTHON:-/home/ilju/miniconda3/envs/${QWEN_ENV_NAME}/bin/python}"
export LLADA_PYTHON="${LLADA_PYTHON:-/home/ilju/miniconda3/envs/${LLADA_ENV_NAME}/bin/python}"
export LLADA21_PYTHON="${LLADA21_PYTHON:-/home/ilju/miniconda3/envs/${LLADA21_ENV_NAME}/bin/python}"
export CONDA_SH="${CONDA_SH:-/home/ilju/miniconda3/etc/profile.d/conda.sh}"

# --- Model paths ---
export MAIN_AGENT_MODEL_PATH="${MAIN_AGENT_MODEL_PATH:-/data/ilju/Qwen3-8B}"
export LLADA_MODEL_PATH="${LLADA_MODEL_PATH:-/data/ilju/LLaDA-8B-Instruct}"
export LLADA21_MODEL_PATH="${LLADA21_MODEL_PATH:-/data/ilju/LLaDA2.1-mini}"

# Match the original DiffuAgent REQUEST_DLLM LLaDA defaults.
export LLADA_BLOCK_LENGTH="${LLADA_BLOCK_LENGTH:-32}"
export LLADA_THRESHOLD="${LLADA_THRESHOLD:-0.9}"

# --- GPU ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
