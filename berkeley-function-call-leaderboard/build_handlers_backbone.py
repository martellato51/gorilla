"""
Backbone model configuration builder.

Registers pure backbone handlers (no Selector/Editor):
  backbone/qwen3-8b  → LLMHandler        (AR via vLLM)
  backbone/llada     → LocalDLLMHandler  (DLM in-process)
"""

from bfcl_eval.constants.model_config import ModelConfig
from bfcl_eval.model_handler.api_inference.diffuagent.handlers import LLMHandler
from bfcl_eval.model_handler.api_inference.diffuagent.handlers_backbone import LocalDLLMHandler


def _build_config(config_name: str, url: str, model_handler) -> ModelConfig:
    return ModelConfig(
        model_name=config_name,
        display_name=config_name,
        url=url,
        org="Backbone Eval",
        license="Apache 2.0",
        model_handler=model_handler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    )


def add_backbone_model_configs() -> dict:
    configs = {
        "backbone/qwen3-8b": _build_config(
            config_name="backbone/qwen3-8b",
            url="https://huggingface.co/Qwen/Qwen3-8B",
            model_handler=LLMHandler,
        ),
        "backbone/llada": _build_config(
            config_name="backbone/llada",
            url="https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct",
            model_handler=LocalDLLMHandler,
        ),
    }
    print(f"Generated {len(configs)} backbone configurations")
    return configs
