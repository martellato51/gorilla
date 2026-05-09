#!/usr/bin/env bash
#SBATCH --job-name=bfcl-setup
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BFCL_ROOT="${BFCL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RESEARCH_ROOT="${RESEARCH_ROOT:-$(cd "${BFCL_ROOT}/../../../.." && pwd)}"
FAST_DLLM_ROOT="${FAST_DLLM_ROOT:-${RESEARCH_ROOT}/Fast-dLLM}"
QWEN_ENV_NAME="${QWEN_ENV_NAME:-qwen3}"
LLADA_ENV_NAME="${LLADA_ENV_NAME:-llada8b}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

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
python -m pip install "transformers>=4.51.0" soundfile
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
python -m pip install soundfile
python register_backbone.py

echo "BFCL environments are ready."
echo "Qwen env : ${QWEN_ENV_NAME}"
echo "LLaDA env: ${LLADA_ENV_NAME}"
echo "BFCL root: ${BFCL_ROOT}"
echo "Next     : sbatch jobs/run_backbone_eval.sbatch"
