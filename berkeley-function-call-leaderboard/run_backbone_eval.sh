#!/usr/bin/env bash
# Backbone evaluation: pure AR (Qwen3-8B via vLLM) vs pure DLM (LLaDA-8B local).
# No Selector/Editor (pre/post modules) applied.
#
# Usage:
#   cd DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard
#   bash run_backbone_eval.sh [--test-categories <cats>] [--num-threads <n>]
#
# Defaults: test-categories=BFCL v4 single-turn categories, num-threads=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurable paths ─────────────────────────────────────────────────────────
RESEARCH_ROOT="${RESEARCH_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${RESEARCH_ROOT}/model/Qwen3-8B}"
LLADA_MODEL_PATH="${LLADA_MODEL_PATH:-${RESEARCH_ROOT}/model/LLaDA-8B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
TEST_CATEGORIES="${TEST_CATEGORIES:-simple_python,multiple,parallel,parallel_multiple,irrelevance,simple_java,simple_javascript,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_relevance,live_irrelevance}"
NUM_THREADS="${NUM_THREADS:-1}"

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-category|--test-categories) TEST_CATEGORIES="$2"; shift 2 ;;
        --num-threads)   NUM_THREADS="$2";   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Environment ────────────────────────────────────────────────────────────────
export MAIN_AGENT_MODEL_PATH="$QWEN_MODEL_PATH"
export MAIN_AGENT_BASE_URL="http://127.0.0.1:${VLLM_PORT}"
export MAIN_AGENT_API_KEY="dummy"
export LLADA_MODEL_PATH="$LLADA_MODEL_PATH"
export FAST_DLLM_LLADA_PATH="${FAST_DLLM_LLADA_PATH:-${RESEARCH_ROOT}/Fast-dLLM/v1/llada}"
export BFCL_PROJECT_ROOT="$SCRIPT_DIR"

echo "============================================================"
echo " Backbone Eval: AR (Qwen3-8B) vs DLM (LLaDA-8B)"
echo "============================================================"
echo " Test categories: $TEST_CATEGORIES"
echo " Num threads   : $NUM_THREADS"
echo " Research root : $RESEARCH_ROOT"
echo " Qwen3 path    : $QWEN_MODEL_PATH"
echo " LLaDA path    : $LLADA_MODEL_PATH"
echo "============================================================"
echo

echo "[0/4] Registering backbone models ..."
python register_backbone.py

# ── AR: Qwen3-8B via vLLM ─────────────────────────────────────────────────────
echo "[1/4] Starting vLLM server for Qwen3-8B ..."
python -m vllm.entrypoints.openai.api_server \
    --model "$QWEN_MODEL_PATH" \
    --port "$VLLM_PORT" \
    --max-model-len 4096 \
    --dtype bfloat16 \
    &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

echo "  vLLM PID: $VLLM_PID — waiting for server to be ready ..."
VLLM_READY=0
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "  vLLM ready."
        VLLM_READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "  vLLM process exited before readiness." >&2
        exit 1
    fi
    sleep 5
done

if [[ "$VLLM_READY" != "1" ]]; then
    echo "  vLLM did not become ready on port ${VLLM_PORT}." >&2
    exit 1
fi

echo "[2/4] Generating AR (backbone/qwen3-8b) ..."
python -m bfcl_eval generate \
    --model backbone/qwen3-8b \
    --test-category "$TEST_CATEGORIES" \
    --num-threads "$NUM_THREADS"

echo "[2/4] Evaluating AR (backbone/qwen3-8b) ..."
python -m bfcl_eval evaluate \
    --model backbone/qwen3-8b \
    --test-category "$TEST_CATEGORIES"

echo "  Stopping vLLM server ..."
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
trap - EXIT

# ── DLM: LLaDA-8B local ───────────────────────────────────────────────────────
echo "[3/4] Generating DLM (backbone/llada) ..."
python -m bfcl_eval generate \
    --model backbone/llada \
    --test-category "$TEST_CATEGORIES" \
    --num-threads "$NUM_THREADS"

echo "[4/4] Evaluating DLM (backbone/llada) ..."
python -m bfcl_eval evaluate \
    --model backbone/llada \
    --test-category "$TEST_CATEGORIES"

echo
echo "============================================================"
echo " Done. Results are in: ${SCRIPT_DIR}/result and ${SCRIPT_DIR}/score"
echo "============================================================"
