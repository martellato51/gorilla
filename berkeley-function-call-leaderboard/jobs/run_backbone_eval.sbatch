#!/usr/bin/env bash
#SBATCH --job-name=bfcl-backbone
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BFCL_ROOT="${BFCL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RESEARCH_ROOT="${RESEARCH_ROOT:-$(cd "${BFCL_ROOT}/../../../.." && pwd)}"
SUBSET_ID_SCRIPT="${SUBSET_ID_SCRIPT:-${BFCL_ROOT}/bfcl_eval/scripts/make_bfcl_subsets_ids.py}"

QWEN_ENV_NAME="${QWEN_ENV_NAME:-qwen3}"
LLADA_ENV_NAME="${LLADA_ENV_NAME:-llada8b}"

export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${RESEARCH_ROOT}/model/Qwen3-8B}"
export MAIN_AGENT_MODEL_PATH="${MAIN_AGENT_MODEL_PATH:-${QWEN_MODEL_PATH}}"
export LLADA_MODEL_PATH="${LLADA_MODEL_PATH:-${RESEARCH_ROOT}/model/LLaDA-8B-Instruct}"
export FAST_DLLM_LLADA_PATH="${FAST_DLLM_LLADA_PATH:-${RESEARCH_ROOT}/Fast-dLLM/v1/llada}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-4096}"
export QWEN_TENSOR_PARALLEL_SIZE="${QWEN_TENSOR_PARALLEL_SIZE:-1}"
export QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.82}"
export QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-32}"
export MAIN_AGENT_BASE_URL="${MAIN_AGENT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}}"
export MAIN_AGENT_API_KEY="${MAIN_AGENT_API_KEY:-dummy}"
export BFCL_PROJECT_ROOT="${BFCL_ROOT}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TEST_CATEGORIES="${TEST_CATEGORIES:-simple_python}"
QWEN_NUM_THREADS="${QWEN_NUM_THREADS:-1}"
LLADA_NUM_THREADS="${LLADA_NUM_THREADS:-1}"
RUN_NAME="${RUN_NAME:-backbone_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
RESULT_DIR="${RESULT_DIR:-result_runs/${RUN_NAME}}"
SCORE_DIR="${SCORE_DIR:-subset_runs/${RUN_NAME}}"

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

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "Stopping vLLM server PID ${VLLM_PID} ..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}

echo "Start: $(date)"
echo "BFCL_ROOT: ${BFCL_ROOT}"
echo "RESEARCH_ROOT: ${RESEARCH_ROOT}"
echo "QWEN_ENV_NAME: ${QWEN_ENV_NAME}"
echo "LLADA_ENV_NAME: ${LLADA_ENV_NAME}"
echo "QWEN_MODEL_PATH: ${QWEN_MODEL_PATH}"
echo "LLADA_MODEL_PATH: ${LLADA_MODEL_PATH}"
echo "FAST_DLLM_LLADA_PATH: ${FAST_DLLM_LLADA_PATH}"
echo "TEST_CATEGORIES: ${TEST_CATEGORIES}"
echo "RESULT_DIR: ${RESULT_DIR}"
echo "SCORE_DIR: ${SCORE_DIR}"

cd "${BFCL_ROOT}"
init_conda

conda activate "${QWEN_ENV_NAME}"
hash -r
python register_backbone.py
python "${SUBSET_ID_SCRIPT}" --bfcl-root "${BFCL_ROOT}" --categories "${TEST_CATEGORIES}"

echo "[1/6] Starting vLLM server for Qwen3-8B ..."
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${VLLM_PORT}$"; then
    echo "Port ${VLLM_PORT} is already in use." >&2
    exit 1
fi

vllm serve "${MAIN_AGENT_MODEL_PATH}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${QWEN_MAX_MODEL_LEN}" \
    --max-num-seqs "${QWEN_MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION}" \
    --dtype bfloat16 \
    --tensor-parallel-size "${QWEN_TENSOR_PARALLEL_SIZE}" \
    --disable-custom-all-reduce &
VLLM_PID=$!
trap cleanup EXIT

VLLM_READY=0
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        VLLM_READY=1
        echo "vLLM ready after ${i} checks."
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vLLM process exited before becoming ready." >&2
        exit 1
    fi
    sleep 5
done
if [[ "${VLLM_READY}" != "1" ]]; then
    echo "vLLM did not become ready on port ${VLLM_PORT}." >&2
    exit 1
fi

echo "[2/6] Generating Qwen3-8B results ..."
python -m bfcl_eval generate \
    --model backbone/qwen3-8b \
    --test-category "${TEST_CATEGORIES}" \
    --result-dir "${RESULT_DIR}" \
    --num-threads "${QWEN_NUM_THREADS}" \
    --run-ids

echo "[3/6] Evaluating Qwen3-8B results ..."
python -m bfcl_eval evaluate \
    --model backbone/qwen3-8b \
    --test-category "${TEST_CATEGORIES}" \
    --result-dir "${RESULT_DIR}" \
    --score-dir "${SCORE_DIR}" \
    --partial-eval

cleanup
trap - EXIT
unset VLLM_PID

echo "[4/6] Switching to LLaDA env ..."
conda activate "${LLADA_ENV_NAME}"
hash -r
python register_backbone.py

echo "[5/6] Generating LLaDA results ..."
python -m bfcl_eval generate \
    --model backbone/llada \
    --test-category "${TEST_CATEGORIES}" \
    --result-dir "${RESULT_DIR}" \
    --num-threads "${LLADA_NUM_THREADS}" \
    --run-ids

echo "[6/6] Evaluating LLaDA results ..."
python -m bfcl_eval evaluate \
    --model backbone/llada \
    --test-category "${TEST_CATEGORIES}" \
    --result-dir "${RESULT_DIR}" \
    --score-dir "${SCORE_DIR}" \
    --partial-eval

echo "Done: $(date)"
echo "Result directory: ${BFCL_ROOT}/${RESULT_DIR}"
echo "Score directory: ${BFCL_ROOT}/${SCORE_DIR}"
