#!/usr/bin/env bash
# Compatibility shim. Shared server settings live at DiffuAgent/env.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BFCL_ROOT="${BFCL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export BFCL_PROJECT_ROOT="$BFCL_ROOT"

DIFFUAGENT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${DIFFUAGENT_ROOT}/env.sh"

export BFCL_ROOT="${BFCL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export BFCL_PROJECT_ROOT="$BFCL_ROOT"
