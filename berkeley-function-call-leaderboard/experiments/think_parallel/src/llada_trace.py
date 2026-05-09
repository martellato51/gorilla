from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer


MASK_ID = 126336
DEFAULT_RESEARCH_ROOT = os.environ.get(
    "RESEARCH_ROOT", str(Path(__file__).resolve().parents[7])
)


def ensure_fast_dllm_on_path() -> None:
    fast_path = os.path.abspath(
        os.environ.get(
            "FAST_DLLM_LLADA_PATH",
            os.path.join(DEFAULT_RESEARCH_ROOT, "Fast-dLLM/v1/llada"),
        )
    )
    if fast_path not in sys.path:
        sys.path.insert(0, fast_path)


def optional_float(value: str | None) -> float | None:
    if value is None or value.lower() in {"", "none", "null"}:
        return None
    return float(value)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    rem = total - base * steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(torch.long)
    cols = torch.arange(steps, device=block_mask_index.device).unsqueeze(0)
    return num_transfer_tokens + (cols < rem.unsqueeze(1)).to(torch.long)


def get_transfer_index_with_confidence(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: torch.Tensor | None,
    threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = (transfer_index | force_mask) & mask_index
        return x0, transfer_index, confidence

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be set when threshold is None")
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = torch.clamp(
        num_transfer_tokens.to(dtype=torch.long, device=confidence.device), min=0
    )

    _values, idx = torch.sort(confidence, dim=1, descending=True)
    batch, length = confidence.shape
    cols = torch.arange(length, device=confidence.device).unsqueeze(0).expand(batch, length)
    select_sorted = cols < num_transfer_tokens.unsqueeze(1).expand(batch, length)
    transfer_int = torch.zeros(batch, length, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index
    return x0, transfer_index, confidence


@torch.no_grad()
def generate_with_trace(
    model: Any,
    prompt: torch.Tensor,
    *,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    remasking: str,
    mask_id: int,
    threshold: float | None,
) -> tuple[torch.Tensor, int, list[dict[str, Any]]]:
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    trace: list[dict[str, Any]] = []
    prompt_len = int(prompt.shape[1])
    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = block_start + block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        step_idx = 0
        while True:
            nfe += 1
            mask_index = x == mask_id
            mask_index[:, block_end:] = 0
            logits = model(x).logits
            quota = None if threshold is not None else num_transfer_tokens[:, step_idx]
            x0, transfer_index, confidence = get_transfer_index_with_confidence(
                logits,
                temperature,
                remasking,
                mask_index,
                x,
                quota,
                threshold,
            )
            selected = transfer_index[0].nonzero(as_tuple=False).flatten().tolist()
            for abs_pos in selected:
                if abs_pos < prompt_len:
                    continue
                token_id = int(x0[0, abs_pos].item())
                trace.append(
                    {
                        "block": block_idx,
                        "commit_step": nfe,
                        "step_in_block": step_idx,
                        "abs_position": abs_pos,
                        "position": abs_pos - prompt_len,
                        "token_id": token_id,
                        "confidence": float(confidence[0, abs_pos].item()),
                    }
                )
            x[transfer_index] = x0[transfer_index]
            step_idx += 1
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
    return x, nfe, trace


class TraceableLLaDABackend:
    def __init__(self, dtype: str = "bfloat16") -> None:
        ensure_fast_dllm_on_path()
        from model.modeling_llada import LLaDAModelLM

        self.model_path = os.environ.get(
            "LLADA_MODEL_PATH",
            os.path.join(DEFAULT_RESEARCH_ROOT, "model/LLaDA-8B-Instruct"),
        )
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        config.flash_attention = True
        if not hasattr(config, "train_max_sequence_length"):
            config.train_max_sequence_length = config.max_sequence_length
        self.model = LLaDAModelLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch_dtype,
            device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.steps = int(os.getenv("LLADA_STEPS", "128"))
        self.gen_length = int(os.getenv("LLADA_GEN_LENGTH", "256"))
        self.block_length = int(os.getenv("LLADA_BLOCK_LENGTH", "32"))
        self.oom_steps = int(os.getenv("LLADA_OOM_STEPS", "64"))
        self.oom_gen_length = int(os.getenv("LLADA_OOM_GEN_LENGTH", "128"))
        self.oom_block_length = int(os.getenv("LLADA_OOM_BLOCK_LENGTH", "32"))
        self.temperature = float(os.getenv("LLADA_TEMPERATURE", "0.0"))
        self.threshold = optional_float(os.getenv("LLADA_THRESHOLD", "null"))
        self.remasking = os.getenv("LLADA_REMASKING", "low_confidence")
        self.context_length = int(os.getenv("LLADA_CONTEXT_LENGTH", "4000"))
        self.enforce_context = os.getenv("LLADA_ENFORCE_CONTEXT", "0") == "1"
        self.truncate_input = os.getenv("LLADA_TRUNCATE_INPUT", "0") == "1"

    @staticmethod
    def _normalize_generation_args(
        gen_length: int, steps: int, block_length: int
    ) -> tuple[int, int, int]:
        block_length = min(block_length, gen_length)
        if gen_length % block_length != 0:
            block_length = max(
                divisor
                for divisor in range(block_length, 0, -1)
                if gen_length % divisor == 0
            )
        num_blocks = gen_length // block_length
        if num_blocks and steps % num_blocks != 0:
            steps = (steps // num_blocks) * num_blocks or num_blocks
        return gen_length, steps, block_length

    def _encode(self, messages: list[dict[str, str]], gen_length: int) -> torch.Tensor:
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        if not self.enforce_context:
            return input_ids
        budget = max(1, self.context_length - gen_length)
        if input_ids.shape[1] <= budget:
            return input_ids
        if not self.truncate_input:
            raise ValueError(
                f"LLaDA input exceeds context budget: {input_ids.shape[1]} > {budget}. "
                "Set LLADA_TRUNCATE_INPUT=1 for an explicit truncation run."
            )
        head_budget = max(1, budget - min(1024, budget // 3))
        tail_budget = max(1, budget - head_budget)
        return torch.cat([input_ids[:, :head_budget], input_ids[:, -tail_budget:]], dim=1)

    def _add_token_text_and_spans(
        self, output_ids: torch.Tensor, trace: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        token_ids = output_ids[0].tolist()
        response = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        trace_by_pos = {item["position"]: item for item in trace}
        cursor = 0
        enriched: list[dict[str, Any]] = []
        for pos, token_id in enumerate(token_ids):
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            item = trace_by_pos.get(pos)
            if item:
                item = dict(item)
                item["token_text"] = token_text
                item["char_start"] = cursor
                item["char_end"] = cursor + len(token_text)
                enriched.append(item)
            cursor += len(token_text)
        return response, enriched

    def generate(self, messages: list[dict[str, str]]) -> SimpleNamespace:
        gen_length, steps, block_length = self._normalize_generation_args(
            self.gen_length, self.steps, self.block_length
        )
        input_ids = self._encode(messages, gen_length).to(next(self.model.parameters()).device)
        start = time.time()
        try:
            out, nfe, trace = generate_with_trace(
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
            gen_length, steps, block_length = self._normalize_generation_args(
                self.oom_gen_length, self.oom_steps, self.oom_block_length
            )
            out, nfe, trace = generate_with_trace(
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
        output_ids = out[:, input_ids.shape[1] :]
        response, trace_tokens = self._add_token_text_and_spans(output_ids, trace)
        return SimpleNamespace(
            text=response,
            latency=latency,
            nfe=nfe,
            input_tokens=int(input_ids.shape[1]),
            output_tokens=int(output_ids.shape[1]),
            trace_tokens=trace_tokens,
            generation_config={
                "gen_length": gen_length,
                "steps": steps,
                "block_length": block_length,
                "temperature": self.temperature,
                "threshold": self.threshold,
            },
        )
