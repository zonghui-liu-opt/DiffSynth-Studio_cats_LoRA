from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch
from PIL import Image

from diffsynth.core import ModelConfig
from diffsynth.utils.data import save_video


DEFAULT_MODEL_ROOT = "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B"
DEFAULT_LORA_PATH = "results/lora_sft/Wan2.2-TI2V-5B_cats_LoRA_rank16_149clips_1e-4/epoch-26.safetensors"
DEFAULT_INPUT_IMAGE = "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/images_480x832/3_cgt-20260701172028-mw5wb.jpg"
DEFAULT_OUTPUT_PATH = "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/lora_sft/Wan2.2-TI2V-5B_cats_LoRA_rank16_149clips_1e-4/pred_videos"
DEFAULT_PROMPT = (
    "视频的首帧是一只银渐层的英国短毛猫，体态圆润微胖，毛发柔顺，全程摄像机静止，纯白背景，全景构图，猫咪居中。"
    "0-1秒：猫咪直立蹲坐，眼神明亮有神，耳朵竖立，胡须轻颤。"
    "1-3秒：猫咪轻微歪头，伴随频繁眨眼和耳朵轻微抖动，眼神保持灵动，胡须轻轻颤动。"
    "3-4秒：平滑恢复直立蹲坐，眼神保持灵动，恢复初始状态。"
    "视频过程中猫咪身体的所有部位必须 100% 始终保持在画面内部"
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def find_dit_paths(model_root: str | Path) -> list[str]:
    pattern = Path(model_root) / "diffusion_pytorch_model*.safetensors"
    dit_paths = sorted(glob.glob(str(pattern)))
    if not dit_paths:
        raise FileNotFoundError(f"No DiT safetensors found at {pattern}")
    return dit_paths


def build_model_configs(
    model_root: str | Path,
    tokenizer_path: str | Path | None = None,
) -> tuple[list[ModelConfig], ModelConfig]:
    model_root = Path(model_root)
    dit_paths = find_dit_paths(model_root)
    text_encoder_path = _require_file(
        model_root / "models_t5_umt5-xxl-enc-bf16.pth",
        "T5 text encoder weights",
    )
    vae_path = _require_file(model_root / "Wan2.2_VAE.pth", "Wan2.2 VAE weights")
    tokenizer_dir = _require_dir(
        Path(tokenizer_path) if tokenizer_path is not None else model_root / "google" / "umt5-xxl",
        "tokenizer directory",
    )

    return (
        [
            ModelConfig(path=dit_paths),
            ModelConfig(path=str(text_encoder_path)),
            ModelConfig(path=str(vae_path)),
        ],
        ModelConfig(path=str(tokenizer_dir)),
    )


def resolve_output_path(output_path: str | Path, seed: int) -> Path:
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path / f"wan22_ti2v5b_cats_lora_seed{seed}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer Wan2.2-TI2V-5B cats LoRA on local H100 checkpoints.")
    parser.add_argument("--model_root", default=os.environ.get("MODEL_ROOT", DEFAULT_MODEL_ROOT))
    parser.add_argument("--tokenizer_path", default=os.environ.get("TOKENIZER_PATH"))
    parser.add_argument("--lora_path", default=os.environ.get("LORA_PATH", DEFAULT_LORA_PATH))
    parser.add_argument("--input_image", default=os.environ.get("INPUT_IMAGE", DEFAULT_INPUT_IMAGE))
    parser.add_argument("--output_path", default=os.environ.get("OUTPUT_PATH", DEFAULT_OUTPUT_PATH))
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--negative_prompt", default=os.environ.get("NEGATIVE_PROMPT"))
    parser.add_argument("--height", type=int, default=int(os.environ.get("HEIGHT", "480")))
    parser.add_argument("--width", type=int, default=int(os.environ.get("WIDTH", "832")))
    parser.add_argument("--num_frames", type=int, default=int(os.environ.get("NUM_FRAMES", "97")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "1")))
    parser.add_argument("--fps", type=float, default=float(os.environ.get("FPS", "24")))
    parser.add_argument("--quality", type=int, default=int(os.environ.get("QUALITY", "5")))
    parser.add_argument("--lora_alpha", type=float, default=float(os.environ.get("LORA_ALPHA", "1")))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument(
        "--tiled",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("TILED", True),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    model_configs, tokenizer_config = build_model_configs(args.model_root, args.tokenizer_path)
    lora_path = _require_file(Path(args.lora_path), "LoRA checkpoint")
    input_image_path = _require_file(Path(args.input_image), "input image")
    output_path = resolve_output_path(args.output_path, args.seed)

    from diffsynth.pipelines.wan_video import WanVideoPipeline

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    pipe.load_lora(pipe.dit, str(lora_path), alpha=args.lora_alpha)

    input_image = Image.open(input_image_path).convert("RGB").resize((args.width, args.height))
    pipe_kwargs = {
        "prompt": args.prompt,
        "input_image": input_image,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "tiled": args.tiled,
    }
    if args.negative_prompt:
        pipe_kwargs["negative_prompt"] = args.negative_prompt

    video = pipe(**pipe_kwargs)
    save_video(video, str(output_path), fps=args.fps, quality=args.quality)
    return output_path


if __name__ == "__main__":
    saved_path = main()
    print(f"Saved video to {saved_path}")
