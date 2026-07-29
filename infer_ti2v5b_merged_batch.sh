#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MERGED_MODEL_ROOT=${MERGED_MODEL_ROOT:-/path/to/merged/Wan2.2-TI2V-5B-cats-lora}
MODEL_ROOT=$MERGED_MODEL_ROOT
OUTPUT_DIR=${OUTPUT_DIR:-./results/merged/Wan2.2-TI2V-5B_cats_merged_batch_pred}
INFERENCE_MODE=merged

export MODEL_ROOT
export OUTPUT_DIR
export INFERENCE_MODE

exec bash "${SCRIPT_DIR}/infer_ti2v5b_lora_batch.sh"
