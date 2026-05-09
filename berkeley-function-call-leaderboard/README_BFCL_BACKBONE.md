# DiffuAgent BFCL Backbone 평가 가이드

이 문서는 기존 BFCL README와 독립적인 로컬 실행 기록입니다. 현재 세팅은 BFCL v4에서 DiffuAgent의 Selector/Editor 같은 부가 모듈 없이 순수 backbone만 평가하기 위한 구성입니다.

평가 대상 backbone은 두 개입니다.

- `backbone/qwen3-8b`: Qwen3-8B를 vLLM OpenAI-compatible 서버로 띄워서 평가
- `backbone/llada`: LLaDA-8B-Instruct를 Fast-dLLM 코드로 Python 프로세스 안에서 직접 로드해서 평가

## 로컬 디렉터리 구조

중요한 경로는 다음과 같습니다.

```text
/data/home/martellato41/research
├── DiffuAgent/
│   └── unified_envs/gorilla/berkeley-function-call-leaderboard/
├── Fast-dLLM/
│   └── v1/llada/
├── model/
│   ├── Qwen3-8B/
│   └── LLaDA-8B-Instruct/
├── setup_bfcl.job
├── run_bfcl_backbone.job
├── run_bfcl_llada.job
└── run_bfcl_qwen3.job
```

BFCL 실행 루트는 항상 아래 경로입니다.

```bash
/data/home/martellato41/research/DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard
```

## 환경 세팅

환경 세팅 job은 다음처럼 실행합니다.

```bash
cd /data/home/martellato41/research
sbatch setup_bfcl.job
```

기본 conda 환경은 `llada8b`입니다.

```bash
CONDA_ENV=llada8b
PYTHON=/data/home/martellato41/.conda/envs/llada8b/bin/python
```

`setup_bfcl.job`이 하는 일은 다음과 같습니다.

1. `llada8b` conda 환경을 활성화합니다.
2. BFCL checkout을 editable mode로 설치합니다: `pip install -e .`
3. `INSTALL_VLLM=1`이면 vLLM extra를 설치합니다: `pip install -e ".[oss_eval_vllm]"`
4. Fast-dLLM v1 의존성을 설치합니다.
5. BFCL import 과정에서 필요한 optional dependency인 `soundfile`을 설치합니다.
6. `register_backbone.py`로 `backbone/qwen3-8b`, `backbone/llada`를 BFCL 모델 목록에 등록합니다.
7. 두 backbone model id가 `MODEL_CONFIG_MAPPING`에 들어갔는지 확인합니다.

`register_diffuagent.py`는 기본적으로 실행하지 않습니다.

```bash
REGISTER_DIFFUAGENT=0
```

이 backbone 실험에는 full DiffuAgent model matrix가 필요 없고, `register_diffuagent.py`는 BFCL의 여러 optional handler를 import하기 때문에 불필요한 dependency 문제를 만들 수 있습니다. full DiffuAgent variant까지 등록해야 할 때만 아래처럼 켭니다.

```bash
REGISTER_DIFFUAGENT=1 sbatch setup_bfcl.job
```

## Backbone 등록 구조

backbone 등록에 관여하는 파일은 다음입니다.

```text
bfcl_eval/build_handlers_backbone.py
register_backbone.py
bfcl_eval/constants/model_config.py
```

`bfcl_eval/build_handlers_backbone.py`는 다음 모델 id를 추가합니다.

```text
backbone/qwen3-8b -> LLMHandler
backbone/llada    -> LocalDLLMHandler
```

`bfcl_eval/constants/model_config.py`에는 다음 코드가 들어가 있습니다.

```python
from bfcl_eval.build_handlers_backbone import add_backbone_model_configs
backbone_model_map = add_backbone_model_configs()

MODEL_CONFIG_MAPPING = {
    **backbone_model_map,
    **api_inference_model_map,
    **local_inference_model_map,
    **third_party_inference_model_map,
}
```

따라서 BFCL에서 아래 모델명을 바로 사용할 수 있습니다.

```bash
python -m bfcl_eval generate --model backbone/qwen3-8b ...
python -m bfcl_eval generate --model backbone/llada ...
```

job들은 실행 전에 `register_backbone.py`를 한 번 더 호출합니다. 이미 등록되어 있으면 아래 메시지를 출력하고 넘어갑니다.

```text
Backbone models already registered - skipping.
```

## 현재 LLaDA Backbone Job

현재 사용한 job은 다음입니다.

```bash
/data/home/martellato41/research/run_bfcl_llada.job
```

이 job은 LLaDA만 돌립니다. Qwen/vLLM은 사용하지 않습니다.

기본 실행 명령은 다음과 같습니다.

```bash
cd /data/home/martellato41/research
sbatch run_bfcl_llada.job
```

실제로 job 내부에서는 아래 명령을 실행합니다.

```bash
python -m bfcl_eval generate \
  --model backbone/llada \
  --test-category "$TEST_CATEGORIES" \
  --num-threads "$NUM_THREADS" \
  --allow-overwrite

python -m bfcl_eval evaluate \
  --model backbone/llada \
  --test-category "$TEST_CATEGORIES"
```

기본 category는 다음입니다.

```bash
TEST_CATEGORIES=simple_python,multiple,parallel,parallel_multiple,irrelevance,simple_java,simple_javascript,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_relevance,live_irrelevance
```

즉 현재 기본 설정은 BFCL v4의 single-turn 계열만 돌립니다.

- `simple_python`
- `simple_java`
- `simple_javascript`
- `multiple`
- `parallel`
- `parallel_multiple`
- `irrelevance`
- `live_simple`
- `live_multiple`
- `live_parallel`
- `live_parallel_multiple`
- `live_relevance`
- `live_irrelevance`

포함되는 종류는 다음과 같습니다.

```text
SINGLE_TURN_CATEGORY = NON_LIVE_CATEGORY + LIVE_CATEGORY
```

`NON_LIVE_CATEGORY`:

```text
simple_python
simple_java
simple_javascript
multiple
parallel
parallel_multiple
irrelevance
```

`LIVE_CATEGORY`:

```text
live_simple
live_multiple
live_parallel
live_parallel_multiple
live_irrelevance
live_relevance
```

현재 기본 job에는 아래 계열은 포함되어 있지 않습니다.

- `multi_turn_base`
- `multi_turn_miss_func`
- `multi_turn_miss_param`
- `multi_turn_long_context`
- `memory_kv`, `memory_vector`, `memory_rec_sum`
- `web_search_base`, `web_search_no_snippet`
- `format_sensitivity`

## BFCL 버전과 Category 이름

현재 checkout은 BFCL v4입니다.

```python
VERSION_PREFIX = "BFCL_v4"
```

따라서 결과 파일 이름도 `BFCL_v4_...` 형태로 저장됩니다.

주의할 점은 BFCL v3식 이름인 `simple`, `java`, `javascript`가 이 checkout에서는 유효하지 않다는 것입니다. v4에서는 다음 이름을 써야 합니다.

```text
simple_python
simple_java
simple_javascript
```

category는 job 변수로 넘길 때 comma-separated 문자열로 넘겨야 합니다.

```bash
TEST_CATEGORIES=simple_python,multiple sbatch run_bfcl_llada.job
```

공백으로 넘기면 Typer가 나머지 단어를 extra argument로 처리합니다.

## 결과 저장 경로

BFCL의 기본 결과 경로는 `bfcl_eval/constants/eval_config.py`에서 정해집니다.

```python
PROJECT_ROOT = Path(os.getenv("BFCL_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RESULT_PATH = PROJECT_ROOT / "result"
SCORE_PATH = PROJECT_ROOT / "score"
```

`run_bfcl_llada.job`는 다음을 export합니다.

```bash
export BFCL_PROJECT_ROOT="$BFCL_ROOT"
```

따라서 현재 LLaDA backbone 결과 경로는 다음입니다.

```text
/data/home/martellato41/research/DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard/result/backbone_llada/
```

카테고리별 결과 파일은 아래 패턴으로 저장됩니다.

```text
result/backbone_llada/<category_group>/BFCL_v4_<category>_result.json
```

예를 들어 현재 생성된 파일은 다음입니다.

```text
result/backbone_llada/non_live/BFCL_v4_irrelevance_result.json
```

평가 점수는 generation이 끝난 뒤 `evaluate` 단계에서 아래 경로에 저장됩니다.

```text
/data/home/martellato41/research/DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard/score/
```

점수 파일 패턴은 다음입니다.

```text
score/backbone_llada/<category_group>/BFCL_v4_<category>_score.json
```

추가로 DiffuAgent handler의 raw prompt/response 로그는 아래에 저장됩니다.

```text
logger/backbone_llada.jsonl
```

현재 확인된 로그 파일은 다음입니다.

```text
/data/home/martellato41/research/DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard/logger/backbone_llada.jsonl
```

## Qwen3-8B vLLM 실행 방식

Qwen3-8B는 LLaDA와 달리 모델을 handler 안에서 직접 로드하지 않습니다. `run_backbone_eval.sh`가 먼저 vLLM 서버를 띄웁니다.

```bash
vllm serve "$QWEN_MODEL_PATH" \
  --port "$VLLM_PORT" \
  --max-model-len 4096 \
  --dtype bfloat16
```

그 다음 아래 환경변수를 설정합니다.

```bash
MAIN_AGENT_MODEL_PATH=/data/home/martellato41/research/model/Qwen3-8B
MAIN_AGENT_BASE_URL=http://127.0.0.1:8000
MAIN_AGENT_API_KEY=dummy
```

중요한 점은 `MAIN_AGENT_BASE_URL`에 `/v1`을 붙이지 않는 것입니다. DiffuAgent의 `REQUEST_LLM` wrapper가 내부에서 `/v1/chat/completions`, `/models`, `/tokenize`를 붙입니다.

올바른 값:

```bash
MAIN_AGENT_BASE_URL=http://127.0.0.1:8000
```

잘못된 값:

```bash
MAIN_AGENT_BASE_URL=http://127.0.0.1:8000/v1
```

`backbone/qwen3-8b` 흐름은 다음과 같습니다.

```text
backbone/qwen3-8b
└── LLMHandler
    └── DiffuagentBaseHandler(backend="llm")
        └── REQUEST_LLM
            ├── POST /tokenize
            └── POST /v1/chat/completions
```

## LLaDA-8B Local DLLM 실행 방식

`backbone/llada`는 HTTP DLLM 서버를 사용하지 않습니다. 대신 `LocalDLLMHandler`가 Fast-dLLM 코드를 같은 Python 프로세스 안에서 직접 import하고 모델을 로드합니다.

관련 파일:

```text
bfcl_eval/model_handler/api_inference/diffuagent/handlers_backbone.py
```

필요한 환경변수:

```bash
LLADA_MODEL_PATH=/data/home/martellato41/research/model/LLaDA-8B-Instruct
FAST_DLLM_LLADA_PATH=/data/home/martellato41/research/Fast-dLLM/v1/llada
```

실행 흐름:

```text
backbone/llada
└── LocalDLLMHandler
    └── _initialize_backend
        ├── FAST_DLLM_LLADA_PATH를 sys.path에 추가
        ├── model.modeling_llada.LLaDAModelLM import
        ├── AutoConfig / AutoTokenizer 로드
        ├── LLaDAModelLM.from_pretrained(..., device_map="auto")
        └── LocalDLLMBackend으로 감싸기
```

추론 흐름:

```text
DiffuagentBaseHandler._query_prompting
└── _query_dllm
    └── LocalDLLMBackend.chat_completion
        ├── tokenizer.apply_chat_template(...)
        ├── generate_with_dual_cache(...)
        └── tokenizer.batch_decode(...)
```

현재 기본 generation 설정:

```text
steps=128
gen_length=128
block_length=32
remasking=low_confidence
mask_id=126336
context_length=4000
```

LLaDA는 프로세스 안에서 모델을 직접 로드하므로 GPU가 필요합니다. 현재 job은 다음 자원을 요청합니다.

```text
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
```

## BFCL 평가 프레임워크 흐름

generation 경로는 다음입니다.

```text
python -m bfcl_eval generate
└── bfcl_eval.__main__.generate
    └── bfcl_eval._llm_response_generation.main
        ├── parse_test_category_argument(...)
        ├── build_handler(model_name, temperature)
        │   └── MODEL_CONFIG_MAPPING[model_name].model_handler(...)
        └── multi_threaded_inference(handler, test_case, ...)
            └── handler.inference(...)
```

현재 backbone 모델들은 `is_fc_model=False`로 등록되어 있으므로 BFCL의 prompting path를 탑니다.

```text
BaseHandler.inference
└── inference_single_turn_prompting 또는 inference_multi_turn_prompting
    └── DiffuagentBaseHandler._query_prompting
```

현재 `run_bfcl_llada.job` 기본 category는 모두 single-turn이므로 실제로는 `inference_single_turn_prompting` 중심으로 흐릅니다.

evaluate 경로는 다음입니다.

```text
python -m bfcl_eval evaluate
└── bfcl_eval.__main__.evaluate
    └── bfcl_eval.eval_checker.eval_runner.main
        ├── result 파일 로드
        ├── 같은 model handler 생성
        └── BFCL ground truth와 비교하여 score 파일 작성
```

## 전체 Backbone 실행

Qwen3-8B와 LLaDA를 한 job에서 순서대로 돌리려면 다음을 사용합니다.

```bash
cd /data/home/martellato41/research
TEST_CATEGORY=simple_python sbatch run_bfcl_backbone.job
```

이 job은 내부에서 `run_backbone_eval.sh`를 호출합니다.

순서는 다음입니다.

1. backbone 모델 등록 확인
2. Qwen3-8B vLLM 서버 시작
3. `backbone/qwen3-8b` generate/evaluate
4. vLLM 서버 종료
5. `backbone/llada` generate/evaluate

## LLaDA만 실행

```bash
cd /data/home/martellato41/research
TEST_CATEGORIES=simple_python sbatch run_bfcl_llada.job
```

기본 전체 single-turn 세트를 돌리려면 변수 없이 제출합니다.

```bash
sbatch run_bfcl_llada.job
```

## Qwen만 실행

Qwen-only job은 vLLM 서버가 이미 떠 있다고 가정합니다.

```bash
cd /data/home/martellato41/research
MAIN_AGENT_BASE_URL=http://127.0.0.1:8000 \
TEST_CATEGORIES=simple_python \
sbatch run_bfcl_qwen3.job
```

## 자주 만난 문제

### `No module named 'soundfile'`

`register_diffuagent.py`가 BFCL 전체 handler를 import하면서 생긴 문제입니다. backbone 평가에는 `register_diffuagent.py`가 필요 없어서 기본적으로 스킵합니다. `setup_bfcl.job`에는 그래도 `soundfile` 설치를 넣어두었습니다.

### `Got unexpected extra arguments (...)`

category를 공백으로 넘겼을 때 발생합니다. comma-separated로 넘겨야 합니다.

```bash
TEST_CATEGORIES=simple_python,multiple sbatch run_bfcl_llada.job
```

### `Invalid test category name provided: simple`

현재 checkout은 BFCL v4입니다. `simple` 대신 `simple_python`을 써야 합니다.

### vLLM endpoint가 `/v1/v1/...`가 되는 문제

`MAIN_AGENT_BASE_URL`에는 `/v1`을 붙이지 않습니다.

```bash
MAIN_AGENT_BASE_URL=http://127.0.0.1:8000
```

### LLaDA가 `generate.py` 또는 `model.modeling_llada`를 못 찾는 문제

아래 경로가 맞는지 확인합니다.

```bash
FAST_DLLM_LLADA_PATH=/data/home/martellato41/research/Fast-dLLM/v1/llada
```

해당 디렉터리에는 다음 파일이 있어야 합니다.

```text
generate.py
model/modeling_llada.py
```

## 빠른 명령 모음

환경 세팅:

```bash
cd /data/home/martellato41/research
sbatch setup_bfcl.job
```

LLaDA smoke test:

```bash
TEST_CATEGORIES=simple_python sbatch run_bfcl_llada.job
```

현재 기본 LLaDA single-turn 세트:

```bash
sbatch run_bfcl_llada.job
```

Qwen + LLaDA backbone smoke test:

```bash
TEST_CATEGORY=simple_python sbatch run_bfcl_backbone.job
```

Qwen only:

```bash
MAIN_AGENT_BASE_URL=http://127.0.0.1:8000 \
TEST_CATEGORIES=simple_python \
sbatch run_bfcl_qwen3.job
```
