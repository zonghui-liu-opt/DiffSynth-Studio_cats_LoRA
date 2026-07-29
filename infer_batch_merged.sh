#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ====== Edit this block on the H100 machine ======
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/lora_sft/Wan2.2-TI2V-5B_cats_LoRA_rank64_600clips_5e-5_ga4steps}
MERGED_MODEL_ROOT=${MERGED_MODEL_ROOT:-$EXPERIMENT_ROOT/merged_models/epoch52_alpha1.0}
COMPARE_ROOT=${COMPARE_ROOT:-$EXPERIMENT_ROOT/pred_videos/epoch52_alpha1.0_compare}
OUTPUT_DIR=${OUTPUT_DIR:-$COMPARE_ROOT/merged_model}
EXPECTED_LORA_ALPHA=${EXPECTED_LORA_ALPHA:-1.0}
# ==================================================

MODEL_ROOT=$MERGED_MODEL_ROOT
INFERENCE_MODE=merged

export MODEL_ROOT
export OUTPUT_DIR
export INFERENCE_MODE
export EXPECTED_LORA_ALPHA

exec bash "${SCRIPT_DIR}/infer_batch.sh"
