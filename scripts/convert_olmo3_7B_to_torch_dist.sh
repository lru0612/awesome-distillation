#!/bin/bash
# Convert OLMo3-7B from HuggingFace format to Megatron torch_dist format.
# Required before the first training run.
#
# Usage:
#   bash scripts/convert_olmo3_7B_to_torch_dist.sh
#
# Adjust HF_CKPT and SAVE_DIR to your actual paths.

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

source "${SCRIPT_DIR}/models/olmo3-7B.sh"

HF_CKPT="/root/models/Olmo-3-7B-Instruct"
SAVE_DIR="/root/models/Olmo-3-7B-Instruct_torch_dist"

PYTHONPATH=/root/Megatron-LM python "${REPO_ROOT}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CKPT}" \
    --save "${SAVE_DIR}"
