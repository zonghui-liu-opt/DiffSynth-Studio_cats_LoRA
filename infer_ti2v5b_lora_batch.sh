#!/usr/bin/env bash
set -euo pipefail

# ====== Edit this block on the H100 machine ======
MODEL_ROOT=${MODEL_ROOT:-/path/to/local/Wan2.2-TI2V-5B}
DATA_ROOT=${DATA_ROOT:-/path/to/testsets}
METADATA_PATH=${METADATA_PATH:-$DATA_ROOT/metadata_6cases_480x832.csv}
LORA_PATH=${LORA_PATH:-/path/to/epoch-26.safetensors}
OUTPUT_DIR=${OUTPUT_DIR:-./results/lora_sft/Wan2.2-TI2V-5B_cats_LoRA_batch_pred}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}

NUM_FRAMES=${NUM_FRAMES:-97}
SEED=${SEED:-1}
FPS=${FPS:-24}
VIDEO_QUALITY=${VIDEO_QUALITY:-5}
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
export DATA_ROOT
export METADATA_PATH
export LORA_PATH
export OUTPUT_DIR
export NUM_FRAMES
export SEED
export FPS
export VIDEO_QUALITY
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

"$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

from PIL import Image

data_root = Path(os.environ["DATA_ROOT"])
metadata_path = Path(os.environ["METADATA_PATH"])

with metadata_path.open(newline="", encoding="utf-8") as handle:
    sample = handle.read(4096)
    handle.seek(0)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        delimiter = "\t" if "\t" in first_line else ","
    reader = csv.DictReader(handle, delimiter=delimiter)
    rows = list(reader)
    fieldnames = reader.fieldnames or []

required = ("input_image", "prompt", "height", "width")
missing = [name for name in required if name not in fieldnames]
if missing:
    raise SystemExit(f"metadata missing required columns: {', '.join(missing)}")
if not rows:
    raise SystemExit("metadata has no rows")

for row_id, row in enumerate(rows):
    image_path = Path(row["input_image"])
    image_path = image_path if image_path.is_absolute() else data_root / image_path
    if not image_path.is_file():
        raise SystemExit(f"missing input_image at row {row_id}: {image_path}")
    with Image.open(image_path) as image:
        image.verify()
    height = int(float(row["height"]))
    width = int(float(row["width"]))
    if height <= 0 or width <= 0 or height % 32 != 0 or width % 32 != 0:
        raise SystemExit(f"invalid metadata resolution at row {row_id}: HxW={height}x{width}")
    expected_bucket = f"{height}x{width}"
    if row.get("bucket") and row["bucket"] != expected_bucket:
        raise SystemExit(
            f"metadata bucket mismatch at row {row_id}: "
            f"metadata bucket={row['bucket']}, expected={expected_bucket}"
        )

print(f"metadata ok: rows={len(rows)}, path={metadata_path}")
PY

echo "Starting Wan2.2-TI2V-5B LoRA batch inference"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "METADATA_PATH=${METADATA_PATH}"
echo "LORA_PATH=${LORA_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_FRAMES=${NUM_FRAMES}, FPS=${FPS}, SEED=${SEED}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

"$PYTHON_BIN" infer_cats_ti2v5b_lora_batch.py
