import csv
import os
from pathlib import Path

import torch
from PIL import Image

from diffsynth.utils.data import save_video
from infer_cats_ti2v5b_lora import (
    LORA_PATH,
    MODEL_ROOT,
    NUM_FRAMES,
    OUTPUT_PATH,
    SEED,
    build_model_configs,
    require_path,
)


DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats"))
METADATA_PATH = Path(os.environ.get("METADATA_PATH", DATA_ROOT / "metadata.csv"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", OUTPUT_PATH))
FPS = int(os.environ.get("FPS", 24))
VIDEO_QUALITY = int(os.environ.get("VIDEO_QUALITY", 5))


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


def main():
    require_path(LORA_PATH, "LoRA checkpoint")
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
    pipe.load_lora(pipe.dit, str(LORA_PATH), alpha=1)

    for row_id, row in enumerate(rows):
        height, width = row_size(row)
        image_path = require_path(input_image_path(DATA_ROOT, row), f"input image at row {row_id}")
        save_path = output_video_path(OUTPUT_DIR, row_id, row)
        print(f"[{row_id + 1}/{len(rows)}] {image_path} -> {save_path}")

        input_image = Image.open(image_path).convert("RGB").resize((width, height))
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
