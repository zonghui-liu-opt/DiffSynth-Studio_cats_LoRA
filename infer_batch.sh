#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Shared launcher. Prefer infer_batch_lora.sh or infer_batch_merged.sh.
# These defaults preserve the original inner-network LoRA invocation.
# ====== Edit this block on the H100 machine ======
MODEL_ROOT=${MODEL_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B}
DATA_ROOT=${DATA_ROOT:-$SCRIPT_DIR/testsets}
METADATA_PATH=${METADATA_PATH:-$DATA_ROOT/metadata_6cases_480x832.csv}
LORA_PATH=${LORA_PATH:-$SCRIPT_DIR/results/lora_sft/Wan2.2-TI2V-5B_cats_LoRA_rank64_600clips_5e-5_ga4steps/epoch-52.safetensors}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRIPT_DIR/results/lora_sft/Wan2.2-TI2V-5B_cats_LoRA_rank64_600clips_5e-5_ga4steps/pred_videos/epoch52_alpha1.0}
INFERENCE_MODE=${INFERENCE_MODE:-lora}

HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
NUM_FRAMES=${NUM_FRAMES:-97}
SEED=${SEED:-1}
LORA_ALPHA=${LORA_ALPHA:-1.0}
FPS=${FPS:-24}
VIDEO_QUALITY=${VIDEO_QUALITY:-5}
DETERMINISTIC=${DETERMINISTIC:-strict}
TILED=${TILED:-1}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-5.0}
CFG_MERGE=${CFG_MERGE:-0}
SIGMA_SHIFT=${SIGMA_SHIFT:-5.0}
NEGATIVE_PROMPT=${NEGATIVE_PROMPT:-}
TIMING_ENABLED=${TIMING_ENABLED:-1}
# Full warmup calls per resolution, excluded from timing reports and video output.
WARMUP_RUNS=${WARMUP_RUNS:-0}
REPEATS=${REPEATS:-1}
# Empty means inspect and report the checkpoint rank without enforcing a value.
EXPECTED_LORA_RANK=${EXPECTED_LORA_RANK-}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DEFAULT_PYTHON=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/wan2.2/bin/python
if [ ! -x "$DEFAULT_PYTHON" ]; then
  DEFAULT_PYTHON=python3
fi
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON}
LOG_PATH=${LOG_PATH:-$OUTPUT_DIR/inference.log}
# ==================================================

if [ "$INFERENCE_MODE" != "lora" ] && [ "$INFERENCE_MODE" != "merged" ]; then
  echo "INFERENCE_MODE must be 'lora' or 'merged', got: ${INFERENCE_MODE}" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}. Set PYTHON_BIN to the existing Wan environment's python." >&2
  exit 1
fi

DIT_PATHS=("${MODEL_ROOT}"/diffusion_pytorch_model*.safetensors)
if [ ! -e "${DIT_PATHS[0]}" ]; then
  echo "No DiT safetensors found at ${MODEL_ROOT}/diffusion_pytorch_model*.safetensors" >&2
  exit 1
fi
if [ ! -f "${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth" ]; then
  echo "Missing ${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth" >&2
  exit 1
fi
if [ ! -f "${MODEL_ROOT}/Wan2.2_VAE.pth" ]; then
  echo "Missing ${MODEL_ROOT}/Wan2.2_VAE.pth" >&2
  exit 1
fi
if [ ! -d "${MODEL_ROOT}/google/umt5-xxl" ]; then
  echo "Missing tokenizer directory: ${MODEL_ROOT}/google/umt5-xxl" >&2
  exit 1
fi
if [ "$INFERENCE_MODE" = "lora" ] && [ ! -f "$LORA_PATH" ]; then
  echo "Missing LoRA checkpoint: ${LORA_PATH}" >&2
  exit 1
fi
if [ ! -d "$DATA_ROOT" ]; then
  echo "Missing data root: ${DATA_ROOT}" >&2
  exit 1
fi
if [ ! -f "$METADATA_PATH" ]; then
  echo "Missing metadata file: ${METADATA_PATH}" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export MODEL_ROOT
export INFERENCE_MODE
export DATA_ROOT
export METADATA_PATH
export LORA_PATH
export OUTPUT_DIR
export HEIGHT
export WIDTH
export NUM_FRAMES
export SEED
export LORA_ALPHA
export FPS
export VIDEO_QUALITY
export DETERMINISTIC
export TILED
export NUM_INFERENCE_STEPS CFG_SCALE CFG_MERGE SIGMA_SHIFT NEGATIVE_PROMPT
export TIMING_ENABLED WARMUP_RUNS REPEATS EXPECTED_LORA_RANK
export CUDA_VISIBLE_DEVICES
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1


echo "Starting Wan2.2-TI2V-5B ${INFERENCE_MODE} batch inference"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "METADATA_PATH=${METADATA_PATH}"
if [ "$INFERENCE_MODE" = "lora" ]; then
  echo "LORA_PATH=${LORA_PATH}, LORA_ALPHA=${LORA_ALPHA}"
fi
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "HEIGHT=${HEIGHT}, WIDTH=${WIDTH}, NUM_FRAMES=${NUM_FRAMES}, FPS=${FPS}"
echo "SEED=${SEED}, TILED=${TILED}, DETERMINISTIC=${DETERMINISTIC}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}, CFG_SCALE=${CFG_SCALE}, CFG_MERGE=${CFG_MERGE}, SIGMA_SHIFT=${SIGMA_SHIFT}"
echo "TIMING_ENABLED=${TIMING_ENABLED}, WARMUP_RUNS=${WARMUP_RUNS} per resolution, REPEATS=${REPEATS}"
echo "PYTHON_BIN=${PYTHON_BIN}, LOG_PATH=${LOG_PATH}"

mkdir -p -- "$(dirname -- "$LOG_PATH")"
: >> "$LOG_PATH"
"$PYTHON_BIN" -u "${SCRIPT_DIR}/infer_batch.py" 2>&1 | tee "$LOG_PATH"
