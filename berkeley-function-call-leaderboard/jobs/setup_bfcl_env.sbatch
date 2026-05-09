#!/usr/bin/env bash
#SBATCH --job-name=bfcl-setup
#SBATCH --output=/data/home/martellato41/research/DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard/logs/%x-%j.out
#SBATCH --error=/data/home/martellato41/research/DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

BFCL_ROOT="${BFCL_ROOT:-/data/home/martellato41/research/DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard}"
FAST_DLLM_ROOT="${FAST_DLLM_ROOT:-/data/home/martellato41/research/Fast-dLLM}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-bfcl2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

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

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    echo "Creating conda env: ${CONDA_ENV_NAME}"
    conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${CONDA_ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel

cd "${BFCL_ROOT}"
python -m pip install -e ".[oss_eval_vllm]"

if [[ -f "${FAST_DLLM_ROOT}/v1/requirements.txt" ]]; then
    python -m pip install -r "${FAST_DLLM_ROOT}/v1/requirements.txt"
fi

python register_backbone.py

echo "BFCL environment is ready."
echo "Env      : ${CONDA_ENV_NAME}"
echo "BFCL root: ${BFCL_ROOT}"
echo "Next     : sbatch jobs/run_backbone_eval.sbatch"
