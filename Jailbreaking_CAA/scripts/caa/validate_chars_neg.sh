#!/usr/bin/env bash
set -euo pipefail

# CHaRS CAA validation restricted to the negative system-prompt condition.
# Usage: scripts/caa/validate_chars_neg.sh [prepare|run|select|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CAA_PROMPT_CONDITION=neg
export CAMPAIGN="${CAMPAIGN:-${SCRIPT_DIR}/../../validation_runs/chars_caa_neg}"

exec "${SCRIPT_DIR}/validate_chars.sh" "${1:-all}"
