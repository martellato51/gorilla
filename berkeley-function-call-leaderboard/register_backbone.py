#!/usr/bin/env python3
"""
Register backbone models in BFCL MODEL_CONFIG_MAPPING.

Run from the BFCL root directory:
  cd DiffuAgent/BFCL && python register_backbone.py
"""

import os
import sys
import shutil
from datetime import datetime


def backup_file(filepath: str) -> str:
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    return backup_path


def register_backbone() -> bool:
    print("=" * 60)
    print("Registering Backbone Models")
    print("=" * 60)
    print()

    model_config_path = "bfcl_eval/constants/model_config.py"
    if not os.path.exists(model_config_path):
        print(f"ERROR: {model_config_path} not found")
        print("Run this script from the BFCL root directory.")
        return False

    print(f"Found {model_config_path}")

    with open(model_config_path, "r") as f:
        content = f.read()

    if "backbone_model_map" in content:
        print("Backbone models already registered – skipping.")
        return True

    print("[Step 1] Backing up original file...")
    backup_path = backup_file(model_config_path)
    print(f"Backed up to: {backup_path}")

    registration_code = (
        "\n# Backbone model configurations (pure AR / DLM, no Selector/Editor)\n"
        "from bfcl_eval.build_handlers_backbone import add_backbone_model_configs\n"
        "backbone_model_map = add_backbone_model_configs()\n"
    )

    lines = content.split("\n")
    model_config_idx = None
    for i, line in enumerate(lines):
        if line.startswith("MODEL_CONFIG_MAPPING = {"):
            model_config_idx = i
            break

    if model_config_idx is None:
        print("ERROR: Cannot find MODEL_CONFIG_MAPPING")
        shutil.copy2(backup_path, model_config_path)
        return False

    lines.insert(model_config_idx, registration_code.strip())

    # Re-find insertion point (line moved)
    for i, line in enumerate(lines):
        if line.startswith("MODEL_CONFIG_MAPPING = {"):
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if "backbone_model_map" not in next_line:
                lines[i + 1] = next_line.replace(
                    "**api_inference_model_map,",
                    "**backbone_model_map,\n    **api_inference_model_map,",
                )
            break

    with open(model_config_path, "w") as f:
        f.write("\n".join(lines))

    print("[Step 2] Verifying registration...")
    try:
        if "bfcl_eval.constants.model_config" in sys.modules:
            del sys.modules["bfcl_eval.constants.model_config"]

        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

        backbone_models = [k for k in MODEL_CONFIG_MAPPING if k.startswith("backbone/")]
        if backbone_models:
            print(f"Registered {len(backbone_models)} backbone model(s):")
            for m in sorted(backbone_models):
                handler = MODEL_CONFIG_MAPPING[m].model_handler
                print(f"  {m} -> {handler.__name__}")
            print()
            print("=" * 60)
            print("Registration successful!")
            print("=" * 60)
            print()
            print("Run evaluation with:")
            print("  bfcl generate --model backbone/qwen3-8b ...")
            print("  bfcl generate --model backbone/llada ...")
            print()
            print(f"To undo: mv {backup_path} {model_config_path}")
            return True
        else:
            print("ERROR: No backbone models found after registration")
            shutil.copy2(backup_path, model_config_path)
            return False

    except Exception as exc:
        import traceback
        print(f"ERROR: {exc}")
        traceback.print_exc()
        print(f"Restoring backup: {backup_path}")
        shutil.copy2(backup_path, model_config_path)
        return False


if __name__ == "__main__":
    success = register_backbone()
    sys.exit(0 if success else 1)
