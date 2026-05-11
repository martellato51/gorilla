#!/usr/bin/env bash
#SBATCH --job-name=bfcl-setup
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

FAST_DLLM_ROOT="${FAST_DLLM_ROOT:-$(cd "${BFCL_ROOT}/../../../../Fast-dLLM" && pwd)}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

init_conda() {
    if [[ -f "${CONDA_SH}" ]]; then
        source "${CONDA_SH}"
    elif [[ -n "${CONDA_EXE:-}" ]]; then
        eval "$("${CONDA_EXE}" shell.bash hook)"
    else
        echo "ERROR: Could not find conda. Set CONDA_SH in jobs/env.sh." >&2
        exit 1
    fi
}

ensure_env() {
    local env_name="$1"
    if ! conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
        echo "Creating conda env: ${env_name}"
        conda create -y -n "${env_name}" "python=${PYTHON_VERSION}"
    fi
}

init_conda
cd "${BFCL_ROOT}"

ensure_env "${QWEN_ENV_NAME}"
conda activate "${QWEN_ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[oss_eval_vllm]"
python -m pip install --force-reinstall \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    "torchaudio==2.6.0" \
    "vllm==0.8.5" \
    "transformers>=4.51.1,<5.0.0" \
    "tokenizers>=0.21.1,<0.22.0" \
    "numpy<2.3" \
    soundfile sentencepiece protobuf safetensors
python register_backbone.py

ensure_env "${LLADA_ENV_NAME}"
conda activate "${LLADA_ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
if [[ -f "${FAST_DLLM_ROOT}/v1/requirements.txt" ]]; then
    python -m pip install -r "${FAST_DLLM_ROOT}/v1/requirements.txt"
else
    echo "WARNING: Fast-dLLM requirements not found at ${FAST_DLLM_ROOT}/v1/requirements.txt"
fi
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip install "torch==${LLADA_TORCH_VERSION:-2.7.0+cu118}" \
    --index-url "${LLADA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
python -m pip install \
    "transformers==4.49.0" \
    "accelerate==0.34.2" \
    "numpy==1.26.4" \
    soundfile sentencepiece protobuf safetensors pyyaml docstring-parser loguru datasets rank-bm25
python register_backbone.py

echo "BFCL environments are ready."
echo "Qwen env : ${QWEN_ENV_NAME}"
echo "LLaDA env: ${LLADA_ENV_NAME}"
echo "BFCL root: ${BFCL_ROOT}"
echo "Next     : sbatch jobs/run_backbone_eval.sbatch"
