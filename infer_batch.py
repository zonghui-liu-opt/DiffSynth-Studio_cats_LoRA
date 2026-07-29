import csv
import json
import os
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

try:
    from diffsynth.core import ModelConfig
except ImportError:
    from diffsynth.pipelines.wan_video import ModelConfig

from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import save_video


MODEL_ROOT = Path(os.environ.get(
    "MODEL_ROOT",
    "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B",
))
INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "lora").strip().lower()
DATA_ROOT = Path(os.environ.get(
    "DATA_ROOT",
    "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/testsets",
))
METADATA_PATH = Path(os.environ.get(
    "METADATA_PATH",
    DATA_ROOT / "metadata_6cases_480x832.csv",
))
LORA_PATH_VALUE = os.environ.get(
    "LORA_PATH",
    "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/lora_sft/Wan2.2-TI2V-5B_cats_LoRA_rank64_600clips_5e-5_ga4steps/epoch-52.safetensors",
).strip()
LORA_PATH = Path(LORA_PATH_VALUE) if LORA_PATH_VALUE else None
OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/lora_sft/Wan2.2-TI2V-5B_cats_LoRA_rank64_600clips_5e-5_ga4steps/pred_videos/epoch52_alpha1.0_compare/runtime_lora",
))

HEIGHT = int(os.environ.get("HEIGHT", 480))
WIDTH = int(os.environ.get("WIDTH", 832))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", 97))
SEED = int(os.environ.get("SEED", 1))
LORA_ALPHA = float(os.environ.get("LORA_ALPHA", 1.0))
FPS = int(os.environ.get("FPS", 24))
VIDEO_QUALITY = int(os.environ.get("VIDEO_QUALITY", 5))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "strict").strip().lower()
TILED = os.environ.get("TILED", "1").strip().lower() not in {"0", "false", "off", "no"}
EXPECTED_LORA_ALPHA_VALUE = os.environ.get("EXPECTED_LORA_ALPHA", "").strip()
EXPECTED_LORA_ALPHA = (
    float(EXPECTED_LORA_ALPHA_VALUE) if EXPECTED_LORA_ALPHA_VALUE else None
)


def require_path(path, name, is_dir=None):
    path = Path(path)

    if is_dir is True:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {name}: {path}")
    elif is_dir is False:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    else:
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    return path


def build_model_configs(model_root):
    model_root = Path(model_root)

    t5_path = require_path(
        model_root / "models_t5_umt5-xxl-enc-bf16.pth",
        "T5 encoder",
        is_dir=False,
    )
    vae_path = require_path(
        model_root / "Wan2.2_VAE.pth",
        "Wan2.2 VAE",
        is_dir=False,
    )
    tokenizer_path = require_path(
        model_root / "google/umt5-xxl",
        "tokenizer directory",
        is_dir=True,
    )

    dit_paths = sorted(model_root.glob("diffusion_pytorch_model*.safetensors"))
    if not dit_paths:
        raise FileNotFoundError(
            f"No DiT safetensors found at {model_root}/diffusion_pytorch_model*.safetensors"
        )

    print("========== MODEL FILES ==========")
    print(f"[T5]  {t5_path}")
    print(f"[VAE] {vae_path}")
    print(f"[Tokenizer] {tokenizer_path}")
    print(f"[DiT] shard count: {len(dit_paths)}")
    for path in dit_paths:
        print(f"  {path}")
    print("=================================")

    model_configs = [
        ModelConfig(path=str(t5_path)),

        # 关键修正：
        # 多个 DiT shard 必须作为同一个 ModelConfig 的 path list 传入。
        # 不能把每个 shard 单独写成一个 ModelConfig。
        ModelConfig(path=[str(path) for path in dit_paths]),

        ModelConfig(path=str(vae_path)),
    ]

    tokenizer_config = ModelConfig(path=str(tokenizer_path))

    return model_configs, tokenizer_config


def configure_determinism(mode=DETERMINISTIC):
    if mode not in {"0", "off", "warn", "strict"}:
        raise ValueError("DETERMINISTIC must be one of: 0, off, warn, strict")
    if mode in {"0", "off"}:
        return

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=mode == "warn")


def detect_delimiter(metadata_path):
    sample = Path(metadata_path).read_text(encoding="utf-8-sig")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        return "\t" if "\t" in first_line else ","


def read_metadata(metadata_path=METADATA_PATH):
    metadata_path = require_path(metadata_path, "metadata file", is_dir=False)

    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
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
    return HEIGHT, WIDTH


def output_video_path(output_dir, row_id, row):
    image_stem = Path(row["input_image"]).stem
    return Path(output_dir) / f"{row_id:04d}_{image_stem}.mp4"


def print_lora_rank(lora_path):
    state_dict = load_file(str(lora_path), device="cpu")

    rank_records = []

    for key, tensor in state_dict.items():
        shape = tuple(tensor.shape)

        if tensor.ndim != 2:
            continue

        if "lora_A" in key:
            rank_records.append((key, shape[0]))
        elif "lora_B" in key:
            rank_records.append((key, shape[1]))
        elif "lora_down" in key:
            rank_records.append((key, shape[0]))
        elif "lora_up" in key:
            rank_records.append((key, shape[1]))

    if not rank_records:
        print(f"[LoRA] No LoRA rank tensors detected in: {lora_path}")
        print("[LoRA] Please check the key names in the safetensors file.")
        return

    ranks = [rank for _, rank in rank_records]
    unique_ranks = sorted(set(ranks))

    print("========== LoRA INFO ==========")
    print(f"[LoRA] path: {lora_path}")
    print(f"[LoRA] detected tensor count: {len(rank_records)}")
    print(f"[LoRA] detected unique ranks: {unique_ranks}")

    if len(unique_ranks) == 1:
        print(f"[LoRA] rank = {unique_ranks[0]}")
    else:
        print("[LoRA] warning: multiple ranks detected")

    print("[LoRA] first few rank tensors:")
    for key, rank in rank_records[:10]:
        print(f"  rank={rank:<4} key={key}")
    print("================================")


def print_merged_model_info(model_root):
    manifest_path = Path(model_root) / "merge_manifest.json"
    if not manifest_path.is_file():
        print(f"[Merged] merge_manifest.json not found under {model_root}")
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    merged_alpha = manifest.get("lora_alpha")
    if EXPECTED_LORA_ALPHA is not None:
        if merged_alpha is None:
            raise ValueError(
                "Merged manifest has no lora_alpha, so equivalence cannot be verified"
            )
        if abs(float(merged_alpha) - EXPECTED_LORA_ALPHA) > 1e-12:
            raise ValueError(
                "Merged LoRA alpha does not match the expected runtime alpha: "
                f"merged={merged_alpha}, expected={EXPECTED_LORA_ALPHA}"
            )
    print("========== MERGED MODEL INFO ==========")
    print(f"[Merged] manifest: {manifest_path}")
    print(f"[Merged] LoRA alpha: {manifest.get('lora_alpha')}")
    print(f"[Merged] merge dtype: {manifest.get('merge_dtype')}")
    print(f"[Merged] matched LoRA layers: {manifest.get('matched_lora_layers')}")
    print(f"[Merged] state SHA256: {manifest.get('merged_state_sha256')}")
    print(f"[Merged] save verified: {manifest.get('save_checksum_verified')}")
    print(f"[Merged] reload verified: {manifest.get('reload_verified')}")
    print("=======================================")
    return manifest


def write_inference_manifest(rows, merged_manifest=None):
    manifest = {
        "inference_mode": INFERENCE_MODE,
        "model_root": str(MODEL_ROOT),
        "data_root": str(DATA_ROOT),
        "metadata_path": str(METADATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "case_count": len(rows),
        "height_fallback": HEIGHT,
        "width_fallback": WIDTH,
        "num_frames": NUM_FRAMES,
        "seed": SEED,
        "fps": FPS,
        "video_quality": VIDEO_QUALITY,
        "tiled": TILED,
        "deterministic": DETERMINISTIC,
        "torch_dtype": "torch.bfloat16",
        "lora_path": str(LORA_PATH) if INFERENCE_MODE == "lora" else None,
        "lora_alpha": LORA_ALPHA if INFERENCE_MODE == "lora" else None,
        "merged_state_sha256": (
            merged_manifest.get("merged_state_sha256") if merged_manifest else None
        ),
        "merged_lora_alpha": (
            merged_manifest.get("lora_alpha") if merged_manifest else None
        ),
        "expected_lora_alpha": EXPECTED_LORA_ALPHA,
    }
    path = OUTPUT_DIR / "inference_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[Manifest] {path}")


def main():
    if INFERENCE_MODE not in {"lora", "merged"}:
        raise ValueError("INFERENCE_MODE must be either 'lora' or 'merged'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan2.2-TI2V-5B inference")

    require_path(MODEL_ROOT, "model root", is_dir=True)
    require_path(DATA_ROOT, "data root", is_dir=True)
    require_path(METADATA_PATH, "metadata file", is_dir=False)
    if INFERENCE_MODE == "lora":
        if LORA_PATH is None:
            raise ValueError("LORA_PATH is required when INFERENCE_MODE=lora")
        require_path(LORA_PATH, "LoRA checkpoint", is_dir=False)

    configure_determinism()
    rows = read_metadata(METADATA_PATH)
    if not rows:
        raise ValueError(f"metadata has no rows: {METADATA_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_configs, tokenizer_config = build_model_configs(MODEL_ROOT)
    merged_manifest = print_merged_model_info(MODEL_ROOT) if INFERENCE_MODE == "merged" else None

    print(f"Starting Wan2.2-TI2V-5B {INFERENCE_MODE} batch inference")
    print(f"MODEL_ROOT={MODEL_ROOT}")
    print(f"DATA_ROOT={DATA_ROOT}")
    print(f"METADATA_PATH={METADATA_PATH}")
    if INFERENCE_MODE == "lora":
        print(f"LORA_PATH={LORA_PATH}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    print(f"HEIGHT={HEIGHT}, WIDTH={WIDTH}, NUM_FRAMES={NUM_FRAMES}")
    print(f"FPS={FPS}, SEED={SEED}, TILED={TILED}, DETERMINISTIC={DETERMINISTIC}")
    if INFERENCE_MODE == "lora":
        print(f"LORA_ALPHA={LORA_ALPHA}")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    if INFERENCE_MODE == "lora":
        pipe.load_lora(pipe.dit, str(LORA_PATH), alpha=LORA_ALPHA)
        print_lora_rank(LORA_PATH)

    write_inference_manifest(rows, merged_manifest=merged_manifest)

    for row_id, row in enumerate(rows):
        height, width = row_size(row)

        image_path = require_path(
            input_image_path(DATA_ROOT, row),
            f"input image at row {row_id}",
            is_dir=False,
        )
        save_path = output_video_path(OUTPUT_DIR, row_id, row)

        print(f"[{row_id + 1}/{len(rows)}] {image_path} -> {save_path}")
        print(f"[Prompt] {row['prompt']}")

        with Image.open(image_path) as image:
            input_image = image.convert("RGB").resize((width, height))

        video = pipe(
            prompt=row["prompt"],
            input_image=input_image,
            height=height,
            width=width,
            num_frames=NUM_FRAMES,
            seed=SEED,
            tiled=TILED,
        )

        save_video(video, str(save_path), fps=FPS, quality=VIDEO_QUALITY)

    print(f"Done. Saved videos to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
