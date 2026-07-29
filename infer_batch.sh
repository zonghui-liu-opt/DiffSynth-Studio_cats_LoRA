#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Shared launcher. Prefer infer_batch_lora.sh or infer_batch_merged.sh.
# These defaults preserve the original inner-network LoRA invocation.
# ====== Edit this block on the H100 machine ======
MODEL_ROOT=${MODEL_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B}
DATA_ROOT=${DATA_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/testsets}
METADATA_PATH=${METADATA_PATH:-$DATA_ROOT/metadata_6cases_480x832.csv}
LORA_PATH=${LORA_PATH:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/lora_sft/Wan2.2-TI2V-5B_cats_LoRA_rank64_600clips_5e-5_ga4steps/epoch-52.safetensors}
OUTPUT_DIR=${OUTPUT_DIR:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/lora_sft/Wan2.2-TI2V-5B_cats_LoRA_rank64_600clips_5e-5_ga4steps/pred_videos/epoch52_alpha1.0}
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
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/wan2.2/bin/python}
# ==================================================

if [ "$INFERENCE_MODE" != "lora" ] && [ "$INFERENCE_MODE" != "merged" ]; then
  echo "INFERENCE_MODE must be 'lora' or 'merged', got: ${INFERENCE_MODE}" >&2
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
export CUDA_VISIBLE_DEVICES
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export DIFFSYNTH_SKIP_DOWNLOAD=True


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

"$PYTHON_BIN" "${SCRIPT_DIR}/infer_batch.py"
