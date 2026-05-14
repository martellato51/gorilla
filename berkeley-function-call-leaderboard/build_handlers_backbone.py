"""
Backbone model configuration builder.

Registers pure backbone handlers (no Selector/Editor):
  backbone/qwen3-8b  -> LLMHandler        (AR via vLLM)
  backbone/llada     -> LocalDLLMHandler  (DLM in-process)
  backbone/llada2.1-mini -> LocalLLaDA21Handler (DLM in-process)
  backbone/llada2.1-mini-native-tools -> LocalLLaDA21ToolsHandler (native tools)
"""

from bfcl_eval.model_handler.api_inference.diffuagent.handlers_backbone import (
    LLMHandler,
    LocalDLLMHandler,
    LocalLLaDA21Handler,
    LocalLLaDA21ToolsHandler,
)


def _build_config(config_name: str, url: str, model_handler):
    from bfcl_eval.constants.model_config import ModelConfig

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
        "backbone/llada2.1-mini": _build_config(
            config_name="backbone/llada2.1-mini",
            url="https://huggingface.co/inclusionAI/LLaDA2.1-mini",
            model_handler=LocalLLaDA21Handler,
        ),
        "backbone/llada2.1-mini-native-tools": _build_config(
            config_name="backbone/llada2.1-mini-native-tools",
            url="https://huggingface.co/inclusionAI/LLaDA2.1-mini",
            model_handler=LocalLLaDA21ToolsHandler,
        ),
    }
    print(f"Generated {len(configs)} backbone configurations")
    return configs
