import csv
import glob
import os
from pathlib import Path

import torch
from PIL import Image

from diffsynth.core import ModelConfig
from diffsynth.utils.data import save_video


MODEL_ROOT = Path("/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B")
LORA_PATH = Path("results/lora_sft/Wan2.2-TI2V-5B_cats_LoRA_rank16_149clips_1e-4/epoch-26.safetensors")
INPUT_IMAGE = Path("/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/images_480x832/3_cgt-20260701172028-mw5wb.jpg")
OUTPUT_PATH = Path("/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/lora_sft/Wan2.2-TI2V-5B_cats_LoRA_rank16_149clips_1e-4/pred_videos")

HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 97
SEED = 1
METADATA_PATH = os.environ.get("METADATA_PATH")
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(METADATA_PATH).parent if METADATA_PATH else INPUT_IMAGE.parent.parent))
ROW_ID = int(os.environ.get("ROW_ID", "0"))

PROMPT = (
    "视频的首帧是一只银渐层的英国短毛猫，体态圆润微胖，毛发柔顺，全程摄像机静止，纯白背景，全景构图，猫咪居中。"
    "0-1秒：猫咪直立蹲坐，眼神明亮有神，耳朵竖立，胡须轻颤。"
    "1-3秒：猫咪轻微歪头，伴随频繁眨眼和耳朵轻微抖动，眼神保持灵动，胡须轻轻颤动。"
    "3-4秒：平滑恢复直立蹲坐，眼神保持灵动，恢复初始状态。"
    "视频过程中猫咪身体的所有部位必须 100% 始终保持在画面内部"
)


def require_path(path, label, is_dir=False):
    path = Path(path)
    exists = path.is_dir() if is_dir else path.is_file()
    if not exists:
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def detect_delimiter(metadata_path):
    sample = Path(metadata_path).read_text(encoding="utf-8")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        return "\t" if "\t" in first_line else ","


def load_case_from_metadata(metadata_path=METADATA_PATH, row_id=ROW_ID):
    if metadata_path is None:
        return PROMPT, INPUT_IMAGE, HEIGHT, WIDTH
    metadata_path = require_path(metadata_path, "metadata file")
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=detect_delimiter(metadata_path))
        rows = list(reader)
    if row_id < 0 or row_id >= len(rows):
        raise IndexError(f"ROW_ID={row_id} is out of range for {metadata_path} with {len(rows)} rows")
    row = rows[row_id]
    missing = [name for name in ("prompt", "input_image", "height", "width") if not row.get(name)]
    if missing:
        raise ValueError(f"metadata row {row_id} missing required columns: {', '.join(missing)}")
    input_image = Path(row["input_image"])
    if not input_image.is_absolute():
        input_image = DATA_ROOT / input_image
    return row["prompt"], input_image, int(float(row["height"])), int(float(row["width"]))


def build_model_configs(model_root=MODEL_ROOT):
    model_root = Path(model_root)
    dit_paths = sorted(glob.glob(str(model_root / "diffusion_pytorch_model*.safetensors")))
    if not dit_paths:
        raise FileNotFoundError(f"No DiT safetensors found at {model_root / 'diffusion_pytorch_model*.safetensors'}")

    text_encoder = require_path(model_root / "models_t5_umt5-xxl-enc-bf16.pth", "T5 text encoder weights")
    vae = require_path(model_root / "Wan2.2_VAE.pth", "Wan2.2 VAE weights")
    tokenizer = require_path(model_root / "google" / "umt5-xxl", "tokenizer directory", is_dir=True)

    return [
        ModelConfig(path=dit_paths),
        ModelConfig(path=str(text_encoder)),
        ModelConfig(path=str(vae)),
    ], ModelConfig(path=str(tokenizer))


def video_output_path(path=OUTPUT_PATH):
    path = Path(path)
    if path.suffix != ".mp4":
        path = path / f"wan22_ti2v5b_cats_lora_seed{SEED}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main():
    require_path(LORA_PATH, "LoRA checkpoint")
    prompt, input_image_path, height, width = load_case_from_metadata()
    require_path(input_image_path, "input image")
    model_configs, tokenizer_config = build_model_configs()

    from diffsynth.pipelines.wan_video import WanVideoPipeline

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    pipe.load_lora(pipe.dit, str(LORA_PATH), alpha=1)

    input_image = Image.open(input_image_path).convert("RGB").resize((width, height))
    video = pipe(
        prompt=prompt,
        input_image=input_image,
        height=height,
        width=width,
        num_frames=NUM_FRAMES,
        seed=SEED,
        tiled=True,
    )
    save_video(video, str(video_output_path()), fps=24, quality=5)


if __name__ == "__main__":
    main()
