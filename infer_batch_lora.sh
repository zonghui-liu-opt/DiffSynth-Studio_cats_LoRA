#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ====== Edit this block on the H100 machine ======
BASE_MODEL_ROOT=${BASE_MODEL_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/lora_sft/Wan2.2-TI2V-5B_cats_LoRA_rank64_600clips_5e-5_ga4steps}
LORA_PATH=${LORA_PATH:-$EXPERIMENT_ROOT/epoch-52.safetensors}
COMPARE_ROOT=${COMPARE_ROOT:-$EXPERIMENT_ROOT/pred_videos/epoch52_alpha1.0_compare}
OUTPUT_DIR=${OUTPUT_DIR:-$COMPARE_ROOT/runtime_lora}
# ==================================================

MODEL_ROOT=$BASE_MODEL_ROOT
INFERENCE_MODE=lora

export MODEL_ROOT
export LORA_PATH
export OUTPUT_DIR
export INFERENCE_MODE

exec bash "${SCRIPT_DIR}/infer_batch.sh"
