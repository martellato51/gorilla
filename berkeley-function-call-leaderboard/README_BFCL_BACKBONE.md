# DiffuAgent BFCL Backbone 평가 가이드

이 문서는 BFCL v4에서 DiffuAgent backbone 모델(Qwen3-8B, LLaDA-8B)만 평가하기 위한 실행 가이드입니다.

## 빠른 시작

```bash
cd <bfcl-root>   # berkeley-function-call-leaderboard 디렉터리

# 1. 서버별 경로 설정 (최초 1회)
vi jobs/env.sh

# 2. 환경 설치
bash jobs/setup_bfcl_env.sbatch

# 3. Qwen3-8B 싱글턴 평가
export TEST_CATEGORIES="simple_python,multiple,parallel,parallel_multiple,simple_java,simple_javascript,live_simple,live_multiple,live_parallel,live_parallel_multiple"
runlog bash jobs/run_bfcl_qwen3.job

# 4. LLaDA-8B 멀티턴 평가
export TEST_CATEGORIES="multi_turn_base"
runlog bash jobs/run_bfcl_llada.job
```

## 서버별 경로 설정

**모든 서버별 경로는 `jobs/env.sh` 하나에 모여 있습니다.** 새 서버에서는 이 파일만 수정하면 됩니다. 모든 job 파일이 실행 시 자동으로 이 파일을 source합니다.

```bash
vi jobs/env.sh
```

수정할 항목:

| 변수 | 설명 |
|------|------|
| `BFCL_ROOT` | 이 repo의 `berkeley-function-call-leaderboard` 경로 |
| `CONDA_SH` | miniconda/anaconda의 `conda.sh` 경로 |
| `QWEN_PYTHON` | `qwen3` conda env의 python 바이너리 |
| `LLADA_PYTHON` | `llada8b` conda env의 python 바이너리 |
| `MAIN_AGENT_MODEL_PATH` | Qwen3-8B 모델 weight 디렉터리 |
| `LLADA_MODEL_PATH` | LLaDA-8B-Instruct 모델 weight 디렉터리 |
| `FAST_DLLM_LLADA_PATH` | Fast-dLLM `v1/llada` 코드 디렉터리 |
| `CUDA_VISIBLE_DEVICES` | 사용할 GPU 번호 (예: `"1,2,3"`) |

## 환경 설치

**진입점은 항상 `jobs/setup_bfcl_env.sbatch`입니다.**

```bash
bash jobs/setup_bfcl_env.sbatch
```

`env.sh`에서 경로를 읽어 두 conda 환경을 순서대로 세팅합니다.

**qwen3 환경** (`qwen3` conda env):
- `pip install -e ".[oss_eval_vllm]"` — BFCL + vLLM extra
- torch 2.6.0, vllm 0.8.5, transformers >=4.51.1,<5.0.0
- `register_backbone.py` 실행

**llada8b 환경** (`llada8b` conda env):
- `pip install -e .` — BFCL base
- Fast-dLLM v1 requirements 설치
- torch 2.7.0+cu118
- transformers 4.49.0, accelerate 0.34.2
- `register_backbone.py` 실행

## Job 파일 목록

| 파일 | 용도 |
|------|------|
| `jobs/setup_bfcl_env.sbatch` | 환경 설치 (항상 여기서 시작) |
| `jobs/run_bfcl_qwen3.job` | Qwen3-8B generate + evaluate |
| `jobs/run_bfcl_llada.job` | LLaDA-8B generate + evaluate |
| `jobs/run_bfcl_singleturn_eval.job` | evaluate만 (generate 결과가 이미 있을 때) |

## 실행 방법

환경변수를 `export`로 설정한 뒤 `runlog`로 실행합니다.

```bash
export TEST_CATEGORIES="simple_python,multiple"
runlog bash jobs/run_bfcl_qwen3.job
```

`runlog`는 `~/.bashrc`에 등록된 함수로, `/home/ilju/research/logs/`에 타임스탬프 로그를 저장합니다.

Slurm 없이 직접 실행할 때는 `bash`로 실행합니다 (`sbatch` 대신).

## 모델별 실행 방식

### Qwen3-8B

vLLM OpenAI-compatible 서버를 내부에서 띄운 뒤 BFCL generate/evaluate를 실행합니다.

주요 환경변수 (기본값은 `jobs/env.sh`에서 설정):

```bash
MAIN_AGENT_MODEL_PATH=<jobs/env.sh에서 설정>
VLLM_PORT=8000
QWEN_MAX_MODEL_LEN=40960
QWEN_TENSOR_PARALLEL_SIZE=2
QWEN_ENABLE_THINKING=0          # thinking 모드 비활성화
CUDA_VISIBLE_DEVICES=<jobs/env.sh에서 설정>
```

`MAIN_AGENT_BASE_URL`에는 `/v1`을 붙이지 않습니다. DiffuAgent `REQUEST_LLM`이 내부에서 경로를 추가합니다.

```bash
MAIN_AGENT_BASE_URL=http://127.0.0.1:8000   # 올바름
MAIN_AGENT_BASE_URL=http://127.0.0.1:8000/v1  # 잘못됨
```

### LLaDA-8B

Fast-dLLM v1 코드를 Python 프로세스 안에서 직접 로드합니다. vLLM 서버 불필요.

주요 환경변수 (기본값은 `jobs/env.sh`에서 설정):

```bash
LLADA_MODEL_PATH=<jobs/env.sh에서 설정>
FAST_DLLM_LLADA_PATH=<jobs/env.sh에서 설정>
LLADA_GEN_LENGTH=128
LLADA_STEPS=128
LLADA_BLOCK_LENGTH=128          # gen_length와 동일 = vanilla fully-parallel diffusion
LLADA_USE_CACHE=0
LLADA_DUAL_CACHE=0
CUDA_VISIBLE_DEVICES=1,2,3
```

`LLADA_BLOCK_LENGTH=LLADA_GEN_LENGTH`이면 block diffusion 없이 완전 병렬 diffusion입니다.

## 태스크 샘플링

각 카테고리에서 평가할 sample ID는 `test_case_ids_to_generate_v3.json` whitelist를 기준으로 선택합니다. `make_bfcl_subsets_ids.py`가 이 whitelist를 읽어 `test_case_ids_to_generate.json`을 생성하고, generate 시 `--run-ids` 플래그로 해당 ID만 실행합니다.

whitelist가 없거나 해당 category가 없으면 에러가 발생합니다 (fallback 없음).

카테고리별 기본 sample 수:

| 카테고리 | 수 |
|----------|-----|
| simple_python, multiple, parallel, parallel_multiple | 50 |
| simple_java, simple_javascript | 50 |
| live_simple, live_multiple | 50 |
| live_parallel, live_parallel_multiple | 전체 |
| multi_turn_base/miss_func/miss_param | 50 |
| multi_turn_long_context | 10 |

## Git 구조

```text
DiffuAgent/unified_envs/gorilla
├── origin   git@github.com:martellato51/gorilla.git
├── upstream https://github.com/ShishirPatil/gorilla
└── branch   diffuagent-bfcl
```

이 nested repo는 상위 `DiffuAgent` repo와 독립적인 git repository입니다. VS Code에서는 두 repo의 상태를 따로 확인해야 합니다.

## 결과 경로

```text
<BFCL_ROOT>/result_runs/<RUN_NAME>/   # generate 결과 (--result-dir)
<BFCL_ROOT>/subset_runs/<RUN_NAME>/   # evaluate 점수 (--score-dir)
<BFCL_ROOT>/logger/                   # raw prompt/response 로그
~/research/logs/                      # runlog 실행 로그
```

`RUN_NAME`은 기본값이 `qwen3_mt_<timestamp>` 또는 `llada_mt_<timestamp>`입니다.

## 자주 만난 문제

### `No module named 'torch'`

conda env가 비어있거나 `setup_bfcl_env.sbatch`가 실행되지 않은 상태입니다.

```bash
bash jobs/setup_bfcl_env.sbatch
```

### `unbound variable: CONDA_ENV`

job 파일 내 `echo "$CONDA_ENV"` 등 존재하지 않는 변수를 참조할 때 `set -u`가 걸립니다. job 파일에서 해당 echo 라인을 제거하면 됩니다.

### `TEST_CATEGORIES` 가 전달 안 됨

`VAR=value runlog bash ...` 형태는 bash 함수에 환경변수가 전달되지 않습니다. `export`를 먼저 사용합니다.

```bash
export TEST_CATEGORIES="simple_python,multiple"
runlog bash jobs/run_bfcl_qwen3.job
```

### `Invalid test category name: simple`

BFCL v4에서는 `simple` 대신 `simple_python`을 사용합니다.

### `Got unexpected extra arguments`

category를 공백으로 넘기면 발생합니다. comma-separated로 넘겨야 합니다.

```bash
export TEST_CATEGORIES="simple_python,multiple"   # 올바름
```

### LLaDA가 `generate.py`를 못 찾는 문제

`jobs/env.sh`의 `FAST_DLLM_LLADA_PATH`가 올바른지 확인합니다. 해당 경로에 `generate.py`와 `model/modeling_llada.py`가 있어야 합니다.
