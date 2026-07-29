#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ====== Edit this block on the H100 machine ======
MODEL_ROOT=${MODEL_ROOT:-/path/to/local/Wan2.2-TI2V-5B}
LORA_PATH=${LORA_PATH:-/path/to/epoch-26.safetensors}
MERGED_MODEL_ROOT=${MERGED_MODEL_ROOT:-./models/merged/Wan2.2-TI2V-5B-cats-lora}
LORA_ALPHA=${LORA_ALPHA:-1}
MAX_SHARD_SIZE_GB=${MAX_SHARD_SIZE_GB:-4}
AUX_FILES_MODE=${AUX_FILES_MODE:-symlink}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}
VERIFY_SAVE=${VERIFY_SAVE:-1}
VERIFY_RELOAD=${VERIFY_RELOAD:-1}
DETERMINISTIC=${DETERMINISTIC:-strict}
# ==================================================

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
if [ ! -f "$LORA_PATH" ]; then
  echo "Missing LoRA checkpoint: ${LORA_PATH}" >&2
  exit 1
fi
if [ -e "$MERGED_MODEL_ROOT" ]; then
  echo "MERGED_MODEL_ROOT already exists; choose a new path to avoid overwriting: ${MERGED_MODEL_ROOT}" >&2
  exit 1
fi

VERIFY_ARGS=()
if [ "$VERIFY_SAVE" = "0" ]; then
  VERIFY_ARGS+=(--skip-save-verification)
fi
if [ "$VERIFY_RELOAD" = "0" ]; then
  VERIFY_ARGS+=(--skip-reload-verification)
fi

export CUDA_VISIBLE_DEVICES
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "Merging Wan2.2-TI2V-5B LoRA into the bf16 baseline"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "LORA_PATH=${LORA_PATH}"
echo "MERGED_MODEL_ROOT=${MERGED_MODEL_ROOT}"
echo "LORA_ALPHA=${LORA_ALPHA}, AUX_FILES_MODE=${AUX_FILES_MODE}, DETERMINISTIC=${DETERMINISTIC}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

"$PYTHON_BIN" "${SCRIPT_DIR}/merge_ti2v5b_lora.py" \
  --model-root "$MODEL_ROOT" \
  --lora-path "$LORA_PATH" \
  --output-root "$MERGED_MODEL_ROOT" \
  --alpha "$LORA_ALPHA" \
  --max-shard-size-gb "$MAX_SHARD_SIZE_GB" \
  --aux-files-mode "$AUX_FILES_MODE" \
  --deterministic "$DETERMINISTIC" \
  "${VERIFY_ARGS[@]}"
