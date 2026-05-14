"""
Backbone-only handlers for pure DLM/AR comparison (no Selector/Editor).

LocalDLLMBackend  – wraps Fast-dLLM LLaDA in-process (no HTTP server).
LocalDLLMHandler  – backbone handler that uses LocalDLLMBackend.
LocalLLaDA21Backend – wraps LLaDA2.1-mini's native HF generate().
LocalLLaDA21Handler – backbone handler that uses LocalLLaDA21Backend.
"""

import json
import os
import re
import sys
import time
import torch
from typing import Any
from types import SimpleNamespace
from overrides import override

from bfcl_eval.model_handler.api_inference.utils.request_llm import REQUEST_LLM
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.model_style import ModelStyle
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
)

# Fast-dLLM path (generate.py + model/ live here). Keep this configurable because
# the BFCL overlay is often copied into a nested checkout.
_FAST_DLLM_LLADA_PATH = os.path.abspath(
    os.environ.get(
        "FAST_DLLM_LLADA_PATH",
        "/data/home/martellato41/research/Fast-dLLM/v1/llada",
    )
)

MASK_ID = 126336
LLADA21_MASK_ID = 156895
LLADA21_EOS_ID = 156892

_LLADA_THINK_DIAGNOSTIC_INSTRUCTION = (
    "Diagnostic instruction for analysis only: Before the function call, write a "
    "brief explanation of your reasoning inside <think>...</think>. After </think>, "
    "return only the BFCL-required function call content in the exact requested "
    "format, with no extra prose."
)


class BackbonePromptingHandler(BaseHandler):
    """BFCL v3-compatible prompting handler for DiffuAgent backbone evals."""

    def __init__(self, model_name: str, temperature: float) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI_Completions
        self.backend = None

    def decode_ast(self, result, language="Python"):
        return default_decode_ast_prompting(result, language)

    def decode_execute(self, result):
        return default_decode_execute_prompting(result)

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        return {"message": [], "function": functions}

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {"role": "assistant", "content": model_response_data["model_responses"]}
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )
        return inference_data

    def _parse_response_text(self, api_response: Any) -> tuple[str, int, int]:
        if hasattr(api_response, "text"):
            return (
                api_response.text,
                getattr(api_response, "input_token", 0),
                getattr(api_response, "num_token", 0),
            )

        return (
            api_response.choices[0].message.content,
            api_response.usage.prompt_tokens,
            api_response.usage.completion_tokens,
        )

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        raw_response, input_token, output_token = self._parse_response_text(api_response)
        model_responses = raw_response
        reasoning_content = ""

        if "</think>" in raw_response:
            think_part, model_responses = raw_response.split("</think>", 1)
            reasoning_content = think_part.split("<think>")[-1].strip()
            model_responses = model_responses.strip()
        elif "<|think|>" in raw_response:
            parts = raw_response.split("<|think|>")
            reasoning_content = parts[0].rstrip("\n") if len(parts) > 1 else ""
            model_responses = parts[-1].lstrip("\n")

        return {
            "model_responses": model_responses,
            "reasoning_content": reasoning_content,
            "input_token": input_token,
            "output_token": output_token,
        }


class LLMHandler(BackbonePromptingHandler):
    """Qwen backbone handler using an existing OpenAI-compatible vLLM server."""

    def __init__(self, model_name: str, temperature: float) -> None:
        super().__init__(model_name, temperature)
        self.api_key = os.getenv("MAIN_AGENT_API_KEY") or os.getenv("VLLM_API_KEY", "dummy")
        self.base_url = os.getenv("MAIN_AGENT_BASE_URL") or os.getenv("VLLM_BASE_URL")
        self.model_path = os.getenv("MAIN_AGENT_MODEL_PATH") or os.getenv(
            "VLLM_MODEL_PATH", self.model_name.split("/")[-1]
        )
        self.context_length = int(
            os.getenv("MAIN_AGENT_CONTEXT_LENGTH", os.getenv("QWEN_MAX_MODEL_LEN", "4096"))
        )
        self.llm = REQUEST_LLM(
            model_path=self.model_path,
            base_url=self.base_url,
            api_key=self.api_key,
            context_length=self.context_length,
        )
        self.llm.check_server_availability()

    @override
    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}
        response = self.llm.chat_completion(
            messages=inference_data["message"],
            temperature=self.temperature,
            max_tokens=int(os.getenv("BFCL_LLM_MAX_TOKENS", "1024")),
        )
        return response, response.latency


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


class LocalLLaDA21Backend:
    """
    Wraps LLaDA2.1-mini's native Hugging Face generation method.

    Unlike Fast-dLLM LLaDA, LLaDA2.1 generate() returns completion tokens only,
    so callers must not slice off the prompt before decoding.
    """

    def __init__(
        self,
        model,
        tokenizer,
        steps: int | None = None,
        gen_length: int | None = None,
        block_length: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.steps = steps or int(os.getenv("LLADA21_STEPS", "128"))
        self.gen_length = gen_length or int(os.getenv("LLADA21_GEN_LENGTH", "128"))
        self.block_length = block_length or int(os.getenv("LLADA21_BLOCK_LENGTH", "32"))
        self.oom_steps = int(os.getenv("LLADA21_OOM_STEPS", "64"))
        self.oom_gen_length = int(os.getenv("LLADA21_OOM_GEN_LENGTH", "64"))
        self.oom_block_length = int(os.getenv("LLADA21_OOM_BLOCK_LENGTH", "32"))
        self.temperature = float(os.getenv("LLADA21_TEMPERATURE", "0.0"))
        self.threshold = float(os.getenv("LLADA21_THRESHOLD", "0.9"))
        self.editing_threshold = float(os.getenv("LLADA21_EDITING_THRESHOLD", "0.9"))
        self.max_post_steps = int(os.getenv("LLADA21_MAX_POST_STEPS", "16"))
        self.context_length = int(os.getenv("LLADA21_CONTEXT_LENGTH", "32768"))
        self.enforce_context = os.getenv("LLADA21_ENFORCE_CONTEXT", "0") == "1"
        self.truncate_input = os.getenv("LLADA21_TRUNCATE_INPUT", "0") == "1"

    def check_server_availability(self):
        pass

    def _format_prompt(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> str:
        if tools:
            return self.tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False
            )
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def num_tokens_from_messages(
        self,
        messages: list[dict],
        quiet: bool = False,
        tools: list[dict] | None = None,
    ) -> int:
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False
        )
        return len(self.tokenizer.encode(prompt))

    def _encode_prompt_with_budget(
        self,
        messages: list[dict],
        input_budget: int,
        quiet: bool = False,
        tools: list[dict] | None = None,
    ):
        prompt = self._format_prompt(messages, tools=tools)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        if not self.enforce_context:
            return input_ids
        if input_ids.shape[1] <= input_budget:
            return input_ids
        if not self.truncate_input:
            raise ValueError(
                "LocalLLaDA2.1 input exceeds context budget: "
                f"{input_ids.shape[1]} input tokens > {input_budget} token budget "
                f"(context={self.context_length}). Set LLADA21_TRUNCATE_INPUT=1 "
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
                "  ! LocalLLaDA2.1 input truncated: "
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

    def _normalize_generation_args(self, gen_length: int, steps: int, block_length: int, max_tokens: int):
        gen_length = min(max_tokens, gen_length, 256)
        block_length = min(block_length, gen_length)
        if gen_length % block_length != 0:
            block_length = max(
                divisor
                for divisor in range(block_length, 0, -1)
                if gen_length % divisor == 0
            )
        steps = min(steps, gen_length)
        return gen_length, steps, block_length

    def _generate(self, input_ids, gen_length: int, steps: int, block_length: int):
        return self.model.generate(
            inputs=input_ids,
            eos_early_stop=True,
            gen_length=gen_length,
            block_length=block_length,
            steps=steps,
            threshold=self.threshold,
            editing_threshold=self.editing_threshold,
            max_post_steps=self.max_post_steps,
            temperature=self.temperature,
            mask_id=LLADA21_MASK_ID,
            eos_id=LLADA21_EOS_ID,
        )

    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 256,
        quiet: bool = False,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> SimpleNamespace:
        gen_length, steps, block_length = self._normalize_generation_args(
            self.gen_length, self.steps, self.block_length, max_tokens
        )
        if not quiet:
            print(
                "  ▶ LocalLLaDA2.1: "
                f"gen_length={gen_length}, steps={steps}, "
                f"block_length={block_length}, temperature={self.temperature}, "
                f"threshold={self.threshold}, editing_threshold={self.editing_threshold}, "
                f"max_post_steps={self.max_post_steps}, enforce_context={self.enforce_context}"
            )

        input_budget = max(1, self.context_length - gen_length)
        input_ids = self._encode_prompt_with_budget(
            messages, input_budget=input_budget, quiet=quiet, tools=tools
        ).to(next(self.model.parameters()).device)

        start = time.time()
        try:
            with torch.inference_mode():
                generated_tokens = self._generate(input_ids, gen_length, steps, block_length)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gen_length, steps, block_length = self._normalize_generation_args(
                self.oom_gen_length, self.oom_steps, self.oom_block_length, max_tokens
            )
            if not quiet:
                print(
                    "  ! LocalLLaDA2.1 CUDA OOM; retrying with "
                    f"gen_length={gen_length}, steps={steps}, block_length={block_length}"
                )
            with torch.inference_mode():
                generated_tokens = self._generate(input_ids, gen_length, steps, block_length)
        latency = time.time() - start

        text = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]

        if not quiet:
            print(f"  ✓ LocalLLaDA2.1 response in {latency:.2f}s: {text[:80]}")

        return SimpleNamespace(
            text=text,
            latency=latency,
            input_token=input_ids.shape[1],
            num_token=generated_tokens.shape[1],
            json={"response": text},
            object=None,
        )


class LocalDLLMHandler(BackbonePromptingHandler):
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

    def __init__(self, model_name: str, temperature: float, dtype: str = "bfloat16") -> None:
        super().__init__(model_name, temperature)
        self.dtype = dtype
        self._llada_model_path = os.environ.get(
            "LLADA_MODEL_PATH", self._DEFAULT_MODEL_PATH
        )
        self.dllm = None

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

    def _query_prompting(self, inference_data: dict):
        if self.dllm is None:
            self._initialize_backend()

        messages = inference_data["message"]
        functions = inference_data.get("function", [])
        formatted_prompt = json.dumps(
            {"messages": messages, "functions": functions}, ensure_ascii=False
        )
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        input_token_count = self.dllm.num_tokens_from_messages(messages=messages, quiet=True)
        context_length = int(os.getenv("LLADA_CONTEXT_LENGTH", self.dllm.context_length))
        default_output_tokens = min(256, int(os.getenv("LLADA_GEN_LENGTH", "128")))
        if input_token_count >= context_length:
            max_tokens = default_output_tokens
        else:
            max_tokens = min(
                default_output_tokens,
                max(1, context_length - input_token_count - 2),
            )

        response = self.dllm.chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            quiet=False,
        )
        keep_dllm_thinking = self._think_diagnostic_enabled()
        if not keep_dllm_thinking and "</think>" in response.text:
            response.text = response.text.split("</think>", 1)[-1].strip()
            response.json["response"] = response.text

        return response, response.latency

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

        device_map = os.getenv("LLADA_DEVICE_MAP", "auto")
        max_memory = None
        max_memory_env = os.getenv("LLADA_MAX_MEMORY", "").strip()
        if max_memory_env:
            max_memory = {}
            for item in max_memory_env.split(","):
                key, value = item.split(":", 1)
                key = key.strip()
                max_memory[int(key) if key.isdigit() else key] = value.strip()
            print(f"  LLaDA device_map={device_map}, max_memory={max_memory}")

        model = LLaDAModelLM.from_pretrained(
            self._llada_model_path,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
        ).eval()
        hf_device_map = getattr(model, "hf_device_map", None)
        if hf_device_map:
            print(f"  LLaDA hf_device_map={hf_device_map}")
        param_bytes_by_device = {}
        for parameter in model.parameters():
            device = str(parameter.device)
            param_bytes_by_device[device] = (
                param_bytes_by_device.get(device, 0)
                + parameter.numel() * parameter.element_size()
            )
        print(
            "  LLaDA parameter bytes by device="
            + ", ".join(
                f"{device}:{num_bytes / (1024 ** 3):.2f}GiB"
                for device, num_bytes in sorted(param_bytes_by_device.items())
            )
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self._llada_model_path, trust_remote_code=True
        )

        self.dllm = LocalDLLMBackend(model=model, tokenizer=tokenizer)
        print("  LLaDA loaded successfully.\n")

    def _initialize_features(self):
        pass


class LocalLLaDA21Handler(LocalDLLMHandler):
    """Backbone-only handler for LLaDA2.1-mini using its native HF generate()."""

    _DEFAULT_MODEL_PATH = "/data/ilju/LLaDA2.1-mini"

    def __init__(self, model_name: str, temperature: float, dtype: str = "bfloat16") -> None:
        super().__init__(model_name, temperature, dtype=dtype)
        self._llada_model_path = os.environ.get(
            "LLADA21_MODEL_PATH", self._DEFAULT_MODEL_PATH
        )

    def _query_prompting(self, inference_data: dict):
        if self.dllm is None:
            self._initialize_backend()

        messages = inference_data["message"]
        functions = inference_data.get("function", [])
        formatted_prompt = json.dumps(
            {"messages": messages, "functions": functions}, ensure_ascii=False
        )
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        input_token_count = self.dllm.num_tokens_from_messages(messages=messages, quiet=True)
        context_length = int(os.getenv("LLADA21_CONTEXT_LENGTH", self.dllm.context_length))
        default_output_tokens = min(256, int(os.getenv("LLADA21_GEN_LENGTH", "128")))
        if input_token_count >= context_length:
            max_tokens = default_output_tokens
        else:
            max_tokens = min(
                default_output_tokens,
                max(1, context_length - input_token_count - 2),
            )

        response = self.dllm.chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            quiet=False,
        )
        keep_dllm_thinking = self._think_diagnostic_enabled()
        if not keep_dllm_thinking and "</think>" in response.text:
            response.text = response.text.split("</think>", 1)[-1].strip()
            response.json["response"] = response.text

        return response, response.latency

    def _initialize_backend(self):
        print(f"\n  Loading LLaDA2.1 locally from: {self._llada_model_path}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                     "float32": torch.float32}
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        device_map = os.getenv("LLADA21_DEVICE_MAP", "auto")
        max_memory = None
        max_memory_env = os.getenv("LLADA21_MAX_MEMORY", "").strip()
        if max_memory_env:
            max_memory = {}
            for item in max_memory_env.split(","):
                key, value = item.split(":", 1)
                key = key.strip()
                max_memory[int(key) if key.isdigit() else key] = value.strip()
            print(f"  LLaDA2.1 device_map={device_map}, max_memory={max_memory}")

        model = AutoModelForCausalLM.from_pretrained(
            self._llada_model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
        ).eval()
        hf_device_map = getattr(model, "hf_device_map", None)
        if hf_device_map:
            print(f"  LLaDA2.1 hf_device_map={hf_device_map}")

        param_bytes_by_device = {}
        for parameter in model.parameters():
            device = str(parameter.device)
            param_bytes_by_device[device] = (
                param_bytes_by_device.get(device, 0)
                + parameter.numel() * parameter.element_size()
            )
        print(
            "  LLaDA2.1 parameter bytes by device="
            + ", ".join(
                f"{device}:{num_bytes / (1024 ** 3):.2f}GiB"
                for device, num_bytes in sorted(param_bytes_by_device.items())
            )
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self._llada_model_path, trust_remote_code=True
        )

        self.dllm = LocalLLaDA21Backend(model=model, tokenizer=tokenizer)
        print("  LLaDA2.1 loaded successfully.\n")


class LocalLLaDA21ToolsHandler(LocalLLaDA21Handler):
    """LLaDA2.1 backbone handler using the tokenizer's native tool-call template."""

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        # Native tool mode lets LLaDA2.1's chat template inject <tools> and
        # <tool_call> instructions. Avoid also adding BFCL's legacy
        # [func(...)] system prompt, since the two output contracts conflict.
        return {"message": [], "function": functions}

    @override
    def _query_prompting(self, inference_data: dict):
        if self.dllm is None:
            self._initialize_backend()

        messages = inference_data["message"]
        functions = inference_data.get("function", [])
        formatted_prompt = json.dumps(
            {"messages": messages, "tools": functions}, ensure_ascii=False
        )
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        input_token_count = self.dllm.num_tokens_from_messages(
            messages=messages, tools=functions, quiet=True
        )
        context_length = int(os.getenv("LLADA21_CONTEXT_LENGTH", self.dllm.context_length))
        default_output_tokens = min(256, int(os.getenv("LLADA21_GEN_LENGTH", "128")))
        if input_token_count >= context_length:
            max_tokens = default_output_tokens
        else:
            max_tokens = min(
                default_output_tokens,
                max(1, context_length - input_token_count - 2),
            )

        response = self.dllm.chat_completion(
            messages=messages,
            tools=functions,
            temperature=self.temperature,
            max_tokens=max_tokens,
            quiet=False,
        )
        keep_dllm_thinking = self._think_diagnostic_enabled()
        if not keep_dllm_thinking and "</think>" in response.text:
            response.text = response.text.split("</think>", 1)[-1].strip()
            response.json["response"] = response.text

        return response, response.latency

    @override
    def decode_ast(self, result, language="Python"):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            return []
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
            if "name" in call and "arguments" in call and isinstance(call["arguments"], dict)
        ]

    @override
    def decode_execute(self, result):
        return convert_to_function_call(self.decode_ast(result))

    @staticmethod
    def _extract_tool_calls(input_string):
        if not isinstance(input_string, str):
            return []

        matches = re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>", input_string, re.DOTALL
        )
        if not matches:
            matches = LocalLLaDA21ToolsHandler._extract_bare_json_objects(input_string)

        tool_calls = []
        for match in matches:
            try:
                parsed = json.loads(match) if isinstance(match, str) else match
            except Exception:
                continue
            if isinstance(parsed, dict):
                tool_calls.append(parsed)
        return tool_calls

    @staticmethod
    def _extract_bare_json_objects(input_string):
        decoder = json.JSONDecoder()
        objects = []
        index = 0
        length = len(input_string)

        while index < length:
            object_start = input_string.find("{", index)
            if object_start == -1:
                break

            try:
                parsed, end = decoder.raw_decode(input_string, object_start)
            except Exception:
                index = object_start + 1
                continue

            objects.append(parsed)
            index = end

        return objects
