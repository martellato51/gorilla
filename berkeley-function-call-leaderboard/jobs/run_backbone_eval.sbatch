#!/usr/bin/env bash
#SBATCH --job-name=bfcl-backbone
#SBATCH --output=/data/home/martellato41/research/DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard/logs/%x-%j.out
#SBATCH --error=/data/home/martellato41/research/DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard/logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1

set -euo pipefail

BFCL_ROOT="${BFCL_ROOT:-/data/home/martellato41/research/DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-bfcl2}"

export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/data/home/martellato41/research/model/Qwen3-8B}"
export LLADA_MODEL_PATH="${LLADA_MODEL_PATH:-/data/home/martellato41/research/model/LLaDA-8B-Instruct}"
export FAST_DLLM_LLADA_PATH="${FAST_DLLM_LLADA_PATH:-/data/home/martellato41/research/Fast-dLLM/v1/llada}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export TEST_CATEGORY="${TEST_CATEGORY:-simple_python}"
export NUM_THREADS="${NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${BFCL_ROOT}/logs"

init_conda() {
    if [[ -n "${CONDA_EXE:-}" ]]; then
        eval "$("${CONDA_EXE}" shell.bash hook)"
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    else
        echo "ERROR: Could not find conda. Set CONDA_EXE or source conda before sbatch." >&2
        exit 1
    fi
}

init_conda
conda activate "${CONDA_ENV_NAME}"

cd "${BFCL_ROOT}"

echo "Running BFCL backbone eval"
echo "BFCL root     : ${BFCL_ROOT}"
echo "Conda env     : ${CONDA_ENV_NAME}"
echo "Qwen path     : ${QWEN_MODEL_PATH}"
echo "LLaDA path    : ${LLADA_MODEL_PATH}"
echo "Fast-dLLM path: ${FAST_DLLM_LLADA_PATH}"
echo "Test category : ${TEST_CATEGORY}"
echo "Num threads   : ${NUM_THREADS}"
echo "CUDA devices  : ${CUDA_VISIBLE_DEVICES}"

bash run_backbone_eval.sh \
    --test-category "${TEST_CATEGORY}" \
    --num-threads "${NUM_THREADS}"
