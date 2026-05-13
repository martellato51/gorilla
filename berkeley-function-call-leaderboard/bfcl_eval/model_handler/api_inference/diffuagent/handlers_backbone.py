"""
Backbone-only handlers for pure DLM/AR comparison (no Selector/Editor).

LocalDLLMBackend  – wraps Fast-dLLM LLaDA in-process (no HTTP server).
LocalDLLMHandler  – DLLMHandler subclass that uses LocalDLLMBackend.
"""

import os
import sys
import time
import torch
from typing import Any
from types import SimpleNamespace
from overrides import override

from bfcl_eval.model_handler.api_inference.diffuagent.handlers import DLLMHandler

# Fast-dLLM path (generate.py + model/ live here). Keep this configurable because
# the BFCL overlay is often copied into a nested checkout.
_FAST_DLLM_LLADA_PATH = os.path.abspath(
    os.environ.get(
        "FAST_DLLM_LLADA_PATH",
        "/data/home/martellato41/research/Fast-dLLM/v1/llada",
    )
)

MASK_ID = 126336

_LLADA_THINK_DIAGNOSTIC_INSTRUCTION = (
    "Diagnostic instruction for analysis only: Before the function call, write a "
    "brief explanation of your reasoning inside <think>...</think>. After </think>, "
    "return only the BFCL-required function call content in the exact requested "
    "format, with no extra prose."
)


def _ensure_fast_dllm_on_path():
    if _FAST_DLLM_LLADA_PATH not in sys.path:
        sys.path.insert(0, _FAST_DLLM_LLADA_PATH)


class LocalDLLMBackend:
    """
    Wraps Fast-dLLM's LLaDA model to expose the same interface as REQUEST_DLLM:
      - chat_completion(messages, temperature, max_tokens, quiet) -> SimpleNamespace
      - num_tokens_from_messages(messages, quiet) -> int
      - check_server_availability() -> None (no-op)
    """

    def __init__(
        self,
        model,
        tokenizer,
        steps: int | None = None,
        gen_length: int | None = None,
        block_length: int | None = None,
        remasking: str = "low_confidence",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.steps = steps or int(os.getenv("LLADA_STEPS", "128"))
        self.gen_length = gen_length or int(os.getenv("LLADA_GEN_LENGTH", "128"))
        self.block_length = block_length or int(os.getenv("LLADA_BLOCK_LENGTH", "32"))
        self.oom_steps = int(os.getenv("LLADA_OOM_STEPS", "64"))
        self.oom_gen_length = int(os.getenv("LLADA_OOM_GEN_LENGTH", "64"))
        self.oom_block_length = int(os.getenv("LLADA_OOM_BLOCK_LENGTH", "32"))
        self.remasking = remasking
        self.temperature = float(os.getenv("LLADA_TEMPERATURE", "0.0"))
        self.threshold = self._optional_float(os.getenv("LLADA_THRESHOLD", "0.9"))
        self.use_cache = os.getenv("LLADA_USE_CACHE", "0") == "1"
        self.dual_cache = os.getenv("LLADA_DUAL_CACHE", "0") == "1"
        self.context_length = int(os.getenv("LLADA_CONTEXT_LENGTH", "4000"))
        self.enforce_context = os.getenv("LLADA_ENFORCE_CONTEXT", "0") == "1"
        self.truncate_input = os.getenv("LLADA_TRUNCATE_INPUT", "0") == "1"

    @staticmethod
    def _optional_float(value: str | None) -> float | None:
        if value is None or value.lower() in {"", "none", "null"}:
            return None
        return float(value)

    def check_server_availability(self):
        pass

    def num_tokens_from_messages(self, messages: list[dict], quiet: bool = False) -> int:
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return len(self.tokenizer.encode(prompt))

    def _encode_prompt_with_budget(
        self,
        messages: list[dict],
        input_budget: int,
        quiet: bool = False,
    ):
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        if not self.enforce_context:
            return input_ids
        if input_ids.shape[1] <= input_budget:
            return input_ids
        if not self.truncate_input:
            raise ValueError(
                "LocalDLLM input exceeds context budget: "
                f"{input_ids.shape[1]} input tokens > {input_budget} token budget "
                f"(context={self.context_length}). Set LLADA_TRUNCATE_INPUT=1 "
                "to run an explicit truncated-input experiment."
            )

        truncated_messages = self._truncate_messages_for_budget(messages, input_budget)
        truncated_prompt = self.tokenizer.apply_chat_template(
            truncated_messages, add_generation_prompt=True, tokenize=False
        )
        truncated = self.tokenizer(truncated_prompt, return_tensors="pt")["input_ids"]
        if truncated.shape[1] > input_budget:
            tail_budget = min(1024, max(1, input_budget // 3))
            head_budget = max(1, input_budget - tail_budget)
            truncated = torch.cat(
                [truncated[:, :head_budget], truncated[:, -tail_budget:]],
                dim=1,
            )

        if not quiet:
            print(
                "  ! LocalDLLM input truncated: "
                f"{input_ids.shape[1]} -> {truncated.shape[1]} tokens "
                f"(budget={input_budget})"
            )
        return truncated

    def _truncate_messages_for_budget(self, messages: list[dict], input_budget: int) -> list[dict]:
        if len(messages) <= 2:
            return messages

        keep_head = messages[:1] if messages and messages[0].get("role") == "system" else []
        tail = messages[len(keep_head):]
        kept_tail = []

        for message in reversed(tail):
            candidate = keep_head + [message] + kept_tail
            prompt = self.tokenizer.apply_chat_template(
                candidate, add_generation_prompt=True, tokenize=False
            )
            if len(self.tokenizer.encode(prompt)) <= input_budget:
                kept_tail.insert(0, message)
            elif not kept_tail:
                kept_tail.insert(0, message)
                break

        return keep_head + kept_tail

    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 256,
        quiet: bool = False,
        **kwargs,
    ) -> SimpleNamespace:
        _ensure_fast_dllm_on_path()
        from generate import generate, generate_with_prefix_cache, generate_with_dual_cache

        def normalize_generation_args(gen_length, steps, block_length):
            gen_length = min(max_tokens, gen_length, 256)
            block_length = min(block_length, gen_length)
            if gen_length % block_length != 0:
                block_length = max(
                    divisor
                    for divisor in range(block_length, 0, -1)
                    if gen_length % divisor == 0
                )
            num_blocks = gen_length // block_length
            if num_blocks > 0 and steps % num_blocks != 0:
                steps = (steps // num_blocks) * num_blocks or num_blocks
            return gen_length, steps, block_length

        gen_length, steps, block_length = normalize_generation_args(
            self.gen_length, self.steps, self.block_length
        )
        # block_length must divide gen_length

        if not quiet:
            print(
                "  ▶ LocalDLLM: "
                f"gen_length={gen_length}, steps={steps}, "
                f"block_length={block_length}, temperature={self.temperature}, "
                f"threshold={self.threshold}, use_cache={self.use_cache}, "
                f"dual_cache={self.dual_cache}, enforce_context={self.enforce_context}"
            )

        input_budget = max(1, self.context_length - gen_length)
        input_ids = self._encode_prompt_with_budget(
            messages, input_budget=input_budget, quiet=quiet
        ).to(next(self.model.parameters()).device)

        start = time.time()
        generate_fn = generate
        if self.dual_cache:
            generate_fn = generate_with_dual_cache
        elif self.use_cache:
            generate_fn = generate_with_prefix_cache

        try:
            with torch.inference_mode():
                out, _ = generate_fn(
                    self.model,
                    input_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=self.temperature,
                    remasking=self.remasking,
                    mask_id=MASK_ID,
                    threshold=self.threshold,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gen_length, steps, block_length = normalize_generation_args(
                self.oom_gen_length, self.oom_steps, self.oom_block_length
            )
            if not quiet:
                print(
                    "  ! LocalDLLM CUDA OOM; retrying with "
                    f"gen_length={gen_length}, steps={steps}, block_length={block_length}"
            )
            with torch.inference_mode():
                out, _ = generate_fn(
                    self.model,
                    input_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=self.temperature,
                    remasking=self.remasking,
                    mask_id=MASK_ID,
                    threshold=self.threshold,
                )
        latency = time.time() - start

        text = self.tokenizer.batch_decode(
            out[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]

        if not quiet:
            print(f"  ✓ LocalDLLM response in {latency:.2f}s: {text[:80]}")

        return SimpleNamespace(
            text=text,
            latency=latency,
            num_token=out.shape[1] - input_ids.shape[1],
            json={"response": text},
            object=None,
        )


class LocalDLLMHandler(DLLMHandler):
    """
    Backbone-only DLM handler: loads LLaDA locally via Fast-dLLM (no HTTP API).
    No Selector or Editor mixins applied.

    Env vars:
      LLADA_MODEL_PATH  – path to LLaDA-8B-Instruct weights
                          (default: /data/home/martellato41/research/model/LLaDA-8B-Instruct)
    """

    _DEFAULT_MODEL_PATH = (
        "/data/home/martellato41/research/model/LLaDA-8B-Instruct"
    )

    def __init__(self, model_name: str, temperature: float, registry_name: str,
                 is_fc_model: bool, dtype: str = "bfloat16") -> None:
        self._llada_model_path = os.environ.get(
            "LLADA_MODEL_PATH", self._DEFAULT_MODEL_PATH
        )
        super().__init__(model_name, temperature, registry_name, is_fc_model, dtype=dtype)

    @staticmethod
    def _think_diagnostic_enabled() -> bool:
        return os.getenv("LLADA_THINK_DIAGNOSTIC", "0") == "1"

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        inference_data = super()._pre_query_processing_prompting(test_entry)
        if not self._think_diagnostic_enabled():
            return inference_data

        first_turn = test_entry["question"][0]
        if first_turn and first_turn[0].get("role") == "system":
            system_message = first_turn[0]
            if _LLADA_THINK_DIAGNOSTIC_INSTRUCTION not in system_message["content"]:
                system_message["content"] += (
                    "\n\n" + _LLADA_THINK_DIAGNOSTIC_INSTRUCTION
                )
        return inference_data

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        if not self._think_diagnostic_enabled():
            return super()._parse_query_response_prompting(api_response)

        raw_response = api_response.text if hasattr(api_response, "text") else api_response.choices[0].text
        model_responses = raw_response
        reasoning_content = ""
        if "</think>" in raw_response:
            think_part, model_responses = raw_response.split("</think>", 1)
            reasoning_content = think_part.split("<think>")[-1].strip()
            model_responses = model_responses.strip()

        return {
            "model_responses": model_responses,
            "reasoning_content": reasoning_content,
            "raw_response": raw_response,
            "input_token": 0,
            "output_token": getattr(api_response, "num_token", 0),
        }

    @override
    def _initialize_backend(self):
        print(f"\n  Loading LLaDA locally from: {self._llada_model_path}")
        _ensure_fast_dllm_on_path()
        from model.modeling_llada import LLaDAModelLM
        from transformers import AutoConfig, AutoTokenizer

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                     "float32": torch.float32}
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        config = AutoConfig.from_pretrained(
            self._llada_model_path, trust_remote_code=True
        )
        config.flash_attention = True
        if not hasattr(config, "train_max_sequence_length"):
            config.train_max_sequence_length = config.max_sequence_length

        model = LLaDAModelLM.from_pretrained(
            self._llada_model_path,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch_dtype,
            device_map="auto",
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            self._llada_model_path, trust_remote_code=True
        )

        self.dllm = LocalDLLMBackend(model=model, tokenizer=tokenizer)
        print("  LLaDA loaded successfully.\n")

    @override
    def _initialize_features(self):
        pass
