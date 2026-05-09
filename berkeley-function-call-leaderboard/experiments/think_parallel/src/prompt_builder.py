from __future__ import annotations

import copy
from typing import Any

from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model
from bfcl_eval.utils import (
    _func_doc_language_specific_pre_processing as func_doc_language_specific_pre_processing,
)

from .bfcl_data import join_user_goal, summarize_initial_config


SCHEMA_EXAMPLE = """{
  "think_units": [
    {
      "unit_id": "u1",
      "think": "...",
      "action_hint": "lockDoors",
      "arguments_hint": {"unlock": false},
      "can_think_now": true,
      "needs_prior_observation_for_arguments": false,
      "needs_prior_execution_before_action": true
    }
  ]
}"""


def build_prompt(
    sample: dict[str, Any],
    mode: str,
    category: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], str]:
    if mode not in {"goal_tools", "goal_init_tools"}:
        raise ValueError(f"Unsupported mode: {mode}")

    functions = func_doc_language_specific_pre_processing(
        copy.deepcopy(sample["function"]), category
    )
    goal = join_user_goal(sample["question"])
    user_content = [
        "You are doing an offline full-trajectory planning diagnostic.",
        "Use the full user goal and available tool schemas to propose independent think units.",
        "Do not execute tools and do not output BFCL function calls.",
        "Return only valid JSON matching this schema:",
        SCHEMA_EXAMPLE,
        "",
        "Full user goal:",
        goal,
    ]
    if mode == "goal_init_tools":
        user_content.extend(
            [
                "",
                "Initial environment configuration summary:",
                summarize_initial_config(sample.get("initial_config", {})),
            ]
        )
    user_content.extend(
        [
            "",
            "For each unit, set:",
            "- can_think_now=true when the unit can be planned from the goal/config now.",
            "- needs_prior_observation_for_arguments=true only when arguments require a previous tool output.",
            "- needs_prior_execution_before_action=true when action execution must wait for another side effect.",
            "Use action_hint names that match available tool names when possible.",
        ]
    )

    messages = [{"role": "user", "content": "\n".join(user_content)}]
    messages = system_prompt_pre_processing_chat_model(messages, functions, category)
    messages[0]["content"] += (
        "\n\nDiagnostic override: this run is not BFCL action execution. "
        "Use the function documentation only as the available tool schema, and "
        "return the requested JSON think_units object instead of function calls."
    )
    return messages, functions, goal
