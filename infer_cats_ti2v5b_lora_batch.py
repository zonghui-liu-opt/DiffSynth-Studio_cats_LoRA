import csv
import os
from pathlib import Path

import torch
from PIL import Image

from diffsynth.utils.data import save_video
from infer_cats_ti2v5b_lora import (
    build_model_configs,
    require_path,
)


MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/path/to/local/Wan2.2-TI2V-5B"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats"))
METADATA_PATH = Path(os.environ.get("METADATA_PATH", DATA_ROOT / "metadata.csv"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./results/ti2v5b_batch_pred"))
INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "lora").strip().lower()
LORA_PATH = os.environ.get("LORA_PATH", "").strip()
LORA_ALPHA = float(os.environ.get("LORA_ALPHA", "1"))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "97"))
SEED = int(os.environ.get("SEED", "1"))
FPS = int(os.environ.get("FPS", 24))
VIDEO_QUALITY = int(os.environ.get("VIDEO_QUALITY", 5))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "strict").strip().lower()


def detect_delimiter(metadata_path):
    sample = Path(metadata_path).read_text(encoding="utf-8")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        return "\t" if "\t" in first_line else ","


def read_metadata(metadata_path=METADATA_PATH):
    metadata_path = require_path(metadata_path, "metadata file")
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=detect_delimiter(metadata_path))
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    missing = [name for name in ("prompt", "input_image") if name not in fieldnames]
    if missing:
        raise ValueError(f"metadata missing required columns: {', '.join(missing)}")
    return rows


def input_image_path(data_root, row):
    path = Path(row["input_image"])
    return path if path.is_absolute() else Path(data_root) / path


def row_size(row):
    if row.get("height") and row.get("width"):
        return int(row["height"]), int(row["width"])
    raise ValueError("metadata row is missing required height/width")


def output_video_path(output_dir, row_id, row):
    return Path(output_dir) / f"{row_id:04d}_{Path(row['input_image']).stem}.mp4"


def configure_determinism(mode=DETERMINISTIC):
    if mode not in {"0", "off", "warn", "strict"}:
        raise ValueError("DETERMINISTIC must be one of: 0, off, warn, strict")
    if mode in {"0", "off"}:
        return

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=mode == "warn")


def main():
    if INFERENCE_MODE not in {"lora", "merged"}:
        raise ValueError("INFERENCE_MODE must be either 'lora' or 'merged'")
    if INFERENCE_MODE == "lora":
        if not LORA_PATH:
            raise ValueError("LORA_PATH is required when INFERENCE_MODE=lora")
        require_path(LORA_PATH, "LoRA checkpoint")

    configure_determinism()
    rows = read_metadata(METADATA_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from diffsynth.pipelines.wan_video import WanVideoPipeline

    model_configs, tokenizer_config = build_model_configs(MODEL_ROOT)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    if INFERENCE_MODE == "lora":
        pipe.load_lora(pipe.dit, LORA_PATH, alpha=LORA_ALPHA)

    print(
        f"inference mode={INFERENCE_MODE}, model_root={MODEL_ROOT}, "
        f"lora_alpha={LORA_ALPHA if INFERENCE_MODE == 'lora' else 'n/a'}, "
        f"deterministic={DETERMINISTIC}"
    )

    for row_id, row in enumerate(rows):
        height, width = row_size(row)
        image_path = require_path(input_image_path(DATA_ROOT, row), f"input image at row {row_id}")
        save_path = output_video_path(OUTPUT_DIR, row_id, row)
        print(f"[{row_id + 1}/{len(rows)}] {image_path} -> {save_path}")

        with Image.open(image_path) as image:
            input_image = image.convert("RGB").resize((width, height))
        video = pipe(
            prompt=row["prompt"],
            input_image=input_image,
            height=height,
            width=width,
            num_frames=NUM_FRAMES,
            seed=SEED,
            tiled=True,
        )
        save_video(video, str(save_path), fps=FPS, quality=VIDEO_QUALITY)


if __name__ == "__main__":
    main()
