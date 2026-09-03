#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ====== Edit this block on the H100 machine ======
BASE_MODEL_ROOT=${BASE_MODEL_ROOT:-${MODEL_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B}}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-$SCRIPT_DIR/results/lora_sft/Wan2.2-TI2V-5B_cats_LoRA_rank64_600clips_5e-5_ga4steps}
LORA_PATH=${LORA_PATH:-$EXPERIMENT_ROOT/epoch-52.safetensors}
DATA_ROOT=${DATA_ROOT:-$SCRIPT_DIR/testsets}
METADATA_PATH=${METADATA_PATH:-$DATA_ROOT/metadata_6cases_480x832.csv}
# Optional override: PYTHON_BIN=/absolute/path/to/your/wan/environment/bin/python
COMPARE_ROOT=${COMPARE_ROOT:-$EXPERIMENT_ROOT/pred_videos/epoch52_alpha1.0_compare}
OUTPUT_DIR=${OUTPUT_DIR:-$COMPARE_ROOT/runtime_lora}
# This experiment uses rank64. Set EXPECTED_LORA_RANK='' to accept another rank.
EXPECTED_LORA_RANK=${EXPECTED_LORA_RANK-64}
# ==================================================

MODEL_ROOT=$BASE_MODEL_ROOT
INFERENCE_MODE=lora

export MODEL_ROOT
export LORA_PATH
export OUTPUT_DIR
export INFERENCE_MODE
export EXPECTED_LORA_RANK
export DATA_ROOT METADATA_PATH

exec bash "${SCRIPT_DIR}/infer_batch.sh"
