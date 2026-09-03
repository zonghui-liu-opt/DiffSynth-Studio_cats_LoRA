import csv
import json
import math
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

# Enforce local-only loading for both the shell and direct Python entry points.
# These must be set before transformers/huggingface_hub are imported.
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
from PIL import Image
from safetensors import safe_open

from inference_timing import WanInferenceTimer
from inference_model_checks import inspect_dit_checkpoint, validate_loaded_pipeline, validate_lora_layout
from inference_runtime_checks import check_runtime

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
EXPECTED_LORA_RANK_VALUE = os.environ.get("EXPECTED_LORA_RANK", "").strip()
EXPECTED_LORA_RANK = int(EXPECTED_LORA_RANK_VALUE) if EXPECTED_LORA_RANK_VALUE else None
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "50"))
CFG_SCALE = float(os.environ.get("CFG_SCALE", "5.0"))
CFG_MERGE = os.environ.get("CFG_MERGE", "0").strip().lower() in {"1", "true", "on", "yes"}
SIGMA_SHIFT = float(os.environ.get("SIGMA_SHIFT", "5.0"))
NEGATIVE_PROMPT = os.environ.get("NEGATIVE_PROMPT", "")
TIMING_ENABLED = os.environ.get("TIMING_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}
WARMUP_RUNS = int(os.environ.get("WARMUP_RUNS", "0"))
REPEATS = int(os.environ.get("REPEATS", "1"))

TIMING_FIELDS = (
    "row_id", "repeat_id", "input_image", "output_path", "height", "width",
    "num_frames", "seed", "num_inference_steps", "cfg_scale", "cfg_merge",
    "dit_seconds", "dit_calls", "dit_mean_call_seconds", "dit_mean_step_seconds",
    "vae_decode_seconds", "vae_decode_calls", "pipeline_seconds",
    "other_pipeline_seconds", "save_video_seconds", "total_seconds",
)
SUMMARY_METRICS = (
    "dit_seconds", "dit_mean_call_seconds", "dit_mean_step_seconds",
    "vae_decode_seconds", "pipeline_seconds", "other_pipeline_seconds",
    "save_video_seconds", "total_seconds",
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


def build_model_configs(model_root, dit_paths=None):
    from diffsynth.core import ModelConfig

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

    if dit_paths is None:
        dit_paths, _, _ = inspect_dit_checkpoint(model_root)

    print("========== MODEL FILES ==========")
    print(f"[T5]  {t5_path}")
    print(f"[VAE] {vae_path}")
    print(f"[Tokenizer] {tokenizer_path}")
    print(f"[DiT] shard count: {len(dit_paths)}")
    for path in dit_paths:
        print(f"  {path}")
    print("=================================")

    model_configs = [
        ModelConfig(path=str(t5_path), skip_download=True),

        # 关键修正：
        # 多个 DiT shard 必须作为同一个 ModelConfig 的 path list 传入。
        # 不能把每个 shard 单独写成一个 ModelConfig。
        ModelConfig(path=[str(path) for path in dit_paths], skip_download=True),

        ModelConfig(path=str(vae_path), skip_download=True),
    ]

    tokenizer_config = ModelConfig(path=str(tokenizer_path), skip_download=True)

    return model_configs, tokenizer_config


def configure_determinism(mode=DETERMINISTIC):
    if mode not in {"0", "off", "warn", "strict"}:
        raise ValueError("DETERMINISTIC must be one of: 0, off, warn, strict")
    if mode in {"0", "off"}:
        return

    # Also covers direct `python infer_batch.py` before the first CUDA operation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
    def dimension(value):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer() or number <= 0:
            raise ValueError(f"Invalid image dimension: {value}")
        return int(number)

    if bool(row.get("height")) != bool(row.get("width")):
        raise ValueError("height and width must be provided together")
    height, width = (
        (dimension(row["height"]), dimension(row["width"]))
        if row.get("height") else (dimension(HEIGHT), dimension(WIDTH))
    )
    if height % 32 or width % 32:
        raise ValueError(f"TI2V-5B height/width must be multiples of 32: {height}x{width}")
    return height, width


def validate_rows(rows):
    if not rows:
        raise ValueError("metadata has no rows")
    for row_id, row in enumerate(rows):
        for name in ("prompt", "input_image"):
            if not row.get(name) or not row[name].strip():
                raise ValueError(f"metadata row {row_id} missing {name}")
        row_size(row)
        path = require_path(input_image_path(DATA_ROOT, row), f"input image at row {row_id}", is_dir=False)
        with Image.open(path) as image:
            image.verify()


def validate_settings():
    if NUM_FRAMES < 1 or NUM_FRAMES % 4 != 1:
        raise ValueError("NUM_FRAMES must be positive and have the form 4*n+1 (e.g. 97)")
    if NUM_INFERENCE_STEPS < 1 or REPEATS < 1 or WARMUP_RUNS < 0:
        raise ValueError("NUM_INFERENCE_STEPS/REPEATS must be positive; WARMUP_RUNS must be >= 0")
    if FPS <= 0:
        raise ValueError("FPS must be positive")
    if not 0 <= VIDEO_QUALITY <= 10:
        raise ValueError("VIDEO_QUALITY must be between 0 and 10")
    if not math.isfinite(CFG_SCALE) or CFG_SCALE < 0:
        raise ValueError("CFG_SCALE must be finite and >= 0")
    if CFG_SCALE == 1.0 and CFG_MERGE:
        # Wan's CFG merger doubles the latent batch, but scale=1 skips splitting it.
        raise ValueError("Use CFG_MERGE=0 when CFG_SCALE=1 (guidance is disabled)")
    if not math.isfinite(SIGMA_SHIFT) or SIGMA_SHIFT <= 0:
        raise ValueError("SIGMA_SHIFT must be finite and positive")
    if not math.isfinite(LORA_ALPHA):
        raise ValueError("LORA_ALPHA must be finite")
    if EXPECTED_LORA_RANK is not None and EXPECTED_LORA_RANK < 1:
        raise ValueError("EXPECTED_LORA_RANK must be positive or empty")


def output_video_path(output_dir, row_id, row, repeat_id=None):
    image_stem = Path(row["input_image"]).stem
    suffix = f"_r{repeat_id + 1:02d}" if repeat_id is not None else ""
    return Path(output_dir) / f"{row_id:04d}_{image_stem}{suffix}.mp4"


def print_lora_rank(lora_path):
    rank_records = []
    # Read tensor shapes without loading another full copy of the checkpoint.
    with safe_open(str(lora_path), framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            shape = checkpoint.get_slice(key).get_shape()
            if len(shape) != 2:
                continue
            if "lora_A" in key or "lora_down" in key:
                rank_records.append((key, shape[0]))
            elif "lora_B" in key or "lora_up" in key:
                rank_records.append((key, shape[1]))

    if not rank_records:
        if EXPECTED_LORA_RANK is not None:
            raise ValueError(f"Cannot verify rank {EXPECTED_LORA_RANK}: no LoRA tensors in {lora_path}")
        print(f"[LoRA] No LoRA rank tensors detected in: {lora_path}")
        print("[LoRA] Please check the key names in the safetensors file.")
        return []

    ranks = [rank for _, rank in rank_records]
    unique_ranks = sorted(set(ranks))
    if EXPECTED_LORA_RANK is not None and unique_ranks != [EXPECTED_LORA_RANK]:
        raise ValueError(f"Expected LoRA rank {EXPECTED_LORA_RANK}, detected ranks {unique_ranks}")

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
    return unique_ranks


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


def write_inference_manifest(rows, merged_manifest=None, lora_ranks=None, model_load_seconds=None,
                             lora_targets=(), model_structure_hash=None, runtime_report=None):
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
        "expected_lora_rank": EXPECTED_LORA_RANK,
        "detected_lora_ranks": lora_ranks,
        "matched_lora_layers": len(lora_targets),
        "model_structure_hash": model_structure_hash,
        "runtime_preflight": runtime_report,
        "lora_execution": "fused_at_load" if INFERENCE_MODE == "lora" else "merged_checkpoint",
        "batch_size": 1,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "cfg_scale": CFG_SCALE,
        "cfg_merge": CFG_MERGE,
        "negative_prompt": NEGATIVE_PROMPT,
        "sigma_shift": SIGMA_SHIFT,
        "timing_enabled": TIMING_ENABLED,
        "warmup_runs_per_resolution": WARMUP_RUNS,
        "repeats": REPEATS,
        "model_load_seconds": model_load_seconds,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    path = OUTPUT_DIR / "inference_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[Manifest] {path}")


def summarize_timings(records):
    summary = {"count": len(records)}
    for name in SUMMARY_METRICS:
        values = sorted(float(row[name]) for row in records)
        if values:
            # Linear interpolation, including well-defined one-sample statistics.
            position = (len(values) - 1) * 0.95
            low, high = math.floor(position), math.ceil(position)
            summary[name] = {
                "total": sum(values), "mean": statistics.mean(values),
                "min": values[0], "max": values[-1], "median": statistics.median(values),
                "p95": values[low] + (values[high] - values[low]) * (position - low),
            }
    return summary


def write_timing_summary(records, planned_samples, warmed_resolutions, status, error=None):
    groups = {}
    for record in records:
        key = f"{record['height']}x{record['width']}"
        groups.setdefault(key, []).append(record)
    summary = {
        "status": status,
        "completed_samples": len(records),
        "planned_samples": planned_samples,
        "warmup_runs_per_resolution": WARMUP_RUNS,
        "warmed_resolutions": [f"{h}x{w}" for h, w in sorted(warmed_resolutions)],
        "repeats": REPEATS,
        "units": "seconds",
        "timing_method": "CUDA-synchronized perf_counter wall time",
        "overall": summarize_timings(records),
        "by_resolution": {key: summarize_timings(group) for key, group in groups.items()},
        "error": error,
    }
    path = OUTPUT_DIR / "timing_summary.json"
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def save_video(frames, save_path, fps, quality):
    # Defer the full DiffSynth import until after the runtime API checks.
    from diffsynth.utils.data import save_video as write_video

    return write_video(frames, save_path, fps=fps, quality=quality)


def run_batch(pipe, rows):
    """Sequential TI2V jobs; warmups are full calls, once per resolution bucket."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records, warmed_resolutions = [], set()
    planned_samples = len(rows) * REPEATS
    csv_context = (
        (OUTPUT_DIR / "timing.csv").open("w", newline="", encoding="utf-8")
        if TIMING_ENABLED else nullcontext(None)
    )
    with csv_context as handle:
        if TIMING_ENABLED:
            writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
            writer.writeheader()
            handle.flush()
            write_timing_summary(records, planned_samples, warmed_resolutions, "running")
        try:
            for row_id, row in enumerate(rows):
                height, width = row_size(row)
                image_path = require_path(input_image_path(DATA_ROOT, row), f"input image at row {row_id}", is_dir=False)
                with Image.open(image_path) as image:
                    input_image = image.convert("RGB").resize((width, height))
                kwargs = dict(
                    prompt=row["prompt"], negative_prompt=row.get("negative_prompt") or NEGATIVE_PROMPT,
                    input_image=input_image, height=height, width=width,
                    num_frames=NUM_FRAMES, seed=SEED, tiled=TILED,
                    num_inference_steps=NUM_INFERENCE_STEPS, cfg_scale=CFG_SCALE,
                    cfg_merge=CFG_MERGE, sigma_shift=SIGMA_SHIFT,
                )
                resolution = (height, width)
                if resolution not in warmed_resolutions:
                    for warmup_id in range(WARMUP_RUNS):
                        print(f"[Warmup {warmup_id + 1}/{WARMUP_RUNS}] {height}x{width}", flush=True)
                        warmup_video = pipe(**kwargs)
                        del warmup_video
                    if WARMUP_RUNS:
                        warmed_resolutions.add(resolution)

                for repeat_id in range(REPEATS):
                    save_path = output_video_path(OUTPUT_DIR, row_id, row, repeat_id if REPEATS > 1 else None)
                    print(f"[{row_id + 1}/{len(rows)}, repeat {repeat_id + 1}/{REPEATS}] {image_path} -> {save_path}", flush=True)
                    print(f"[Prompt] {row['prompt']}")
                    with WanInferenceTimer(pipe, enabled=TIMING_ENABLED) as timer:
                        video = pipe(**kwargs)
                    metrics = timer.metrics()
                    if TIMING_ENABLED:
                        if not metrics["dit_calls"] or not metrics["vae_decode_calls"]:
                            raise RuntimeError("No DiT/VAE decode calls captured; check the pipeline timing boundaries")
                        actual_steps = len(pipe.scheduler.timesteps)
                        if actual_steps == 0:
                            raise RuntimeError("Pipeline completed without any denoising steps")
                        save_start = time.perf_counter()
                    save_video(video, str(save_path), fps=FPS, quality=VIDEO_QUALITY)
                    if TIMING_ENABLED:
                        save_seconds = time.perf_counter() - save_start
                        record = {
                            "row_id": row_id, "repeat_id": repeat_id,
                            "input_image": str(image_path), "output_path": str(save_path),
                            "height": height, "width": width, "num_frames": NUM_FRAMES,
                            "seed": SEED, "num_inference_steps": actual_steps,
                            "cfg_scale": CFG_SCALE, "cfg_merge": CFG_MERGE,
                            **metrics,
                            "dit_mean_step_seconds": metrics["dit_seconds"] / actual_steps,
                            "save_video_seconds": save_seconds,
                            "total_seconds": metrics["pipeline_seconds"] + save_seconds,
                        }
                        writer.writerow(record)
                        handle.flush()
                        records.append(record)
                        write_timing_summary(records, planned_samples, warmed_resolutions, "running")
                        print(
                            f"[Timing] DiT={metrics['dit_seconds']:.3f}s ({metrics['dit_calls']} calls), "
                            f"VAE decode={metrics['vae_decode_seconds']:.3f}s, "
                            f"pipeline={metrics['pipeline_seconds']:.3f}s, save={save_seconds:.3f}s",
                            flush=True,
                        )
                    del video
        except BaseException as error:
            if TIMING_ENABLED:
                write_timing_summary(records, planned_samples, warmed_resolutions, "failed", f"{type(error).__name__}: {error}")
            raise
        if TIMING_ENABLED:
            write_timing_summary(records, planned_samples, warmed_resolutions, "complete")
            summary = summarize_timings(records)
            if records:
                print(
                    f"[Timing mean, n={len(records)}] DiT={summary['dit_seconds']['mean']:.3f}s, "
                    f"VAE decode={summary['vae_decode_seconds']['mean']:.3f}s, "
                    f"pipeline={summary['pipeline_seconds']['mean']:.3f}s"
                )
            print(f"[Timing reports] {OUTPUT_DIR / 'timing.csv'}, {OUTPUT_DIR / 'timing_summary.json'}")
    return records


def main():
    validate_settings()
    if INFERENCE_MODE not in {"lora", "merged"}:
        raise ValueError("INFERENCE_MODE must be either 'lora' or 'merged'")
    require_path(MODEL_ROOT, "model root", is_dir=True)
    require_path(DATA_ROOT, "data root", is_dir=True)
    require_path(METADATA_PATH, "metadata file", is_dir=False)
    if INFERENCE_MODE == "lora":
        if LORA_PATH is None:
            raise ValueError("LORA_PATH is required when INFERENCE_MODE=lora")
        require_path(LORA_PATH, "LoRA checkpoint", is_dir=False)

    configure_determinism()
    rows = read_metadata(METADATA_PATH)
    validate_rows(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[Preflight] Checking CUDA, tokenizer and video encoder...", flush=True)
    runtime_report = check_runtime(MODEL_ROOT / "google/umt5-xxl", OUTPUT_DIR, FPS, VIDEO_QUALITY)
    print("[Preflight] Checking model shards and LoRA tensor headers...", flush=True)
    dit_paths, dit_shapes, model_structure_hash = inspect_dit_checkpoint(MODEL_ROOT)
    model_configs, tokenizer_config = build_model_configs(MODEL_ROOT, dit_paths=dit_paths)
    merged_manifest = print_merged_model_info(MODEL_ROOT) if INFERENCE_MODE == "merged" else None
    lora_ranks = print_lora_rank(LORA_PATH) if INFERENCE_MODE == "lora" else None
    lora_targets = validate_lora_layout(LORA_PATH, dit_shapes) if INFERENCE_MODE == "lora" else []
    if INFERENCE_MODE == "lora":
        print(f"[Preflight] LoRA targets matched: {len(lora_targets)}", flush=True)
    preflight_path = OUTPUT_DIR / "preflight_report.json"
    preflight_path.write_text(json.dumps(runtime_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[Preflight] Passed; report: {preflight_path}. Starting model load.", flush=True)

    print(f"Starting Wan2.2-TI2V-5B {INFERENCE_MODE} batch inference")
    print(f"MODEL_ROOT={MODEL_ROOT}")
    print(f"DATA_ROOT={DATA_ROOT}")
    print(f"METADATA_PATH={METADATA_PATH}")
    if INFERENCE_MODE == "lora":
        print(f"LORA_PATH={LORA_PATH}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    print(f"HEIGHT={HEIGHT}, WIDTH={WIDTH}, NUM_FRAMES={NUM_FRAMES}")
    print(f"FPS={FPS}, SEED={SEED}, TILED={TILED}, DETERMINISTIC={DETERMINISTIC}")
    print(f"STEPS={NUM_INFERENCE_STEPS}, CFG_SCALE={CFG_SCALE}, CFG_MERGE={CFG_MERGE}, SIGMA_SHIFT={SIGMA_SHIFT}")
    print(f"TIMING_ENABLED={TIMING_ENABLED}, WARMUP_RUNS={WARMUP_RUNS} per resolution, REPEATS={REPEATS}")
    if INFERENCE_MODE == "lora":
        print(f"LORA_ALPHA={LORA_ALPHA}")

    torch.cuda.synchronize()
    model_load_start = time.perf_counter()
    from diffsynth.pipelines.wan_video import WanVideoPipeline

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
    )

    validate_loaded_pipeline(pipe, lora_targets)
    if INFERENCE_MODE == "lora":
        pipe.load_lora(pipe.dit, str(LORA_PATH), alpha=LORA_ALPHA, hotload=False)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_start
    print(f"[Model load + LoRA fusion] {model_load_seconds:.3f}s (excluded from sample timings)")

    write_inference_manifest(
        rows, merged_manifest=merged_manifest, lora_ranks=lora_ranks,
        model_load_seconds=model_load_seconds, lora_targets=lora_targets,
        model_structure_hash=model_structure_hash, runtime_report=runtime_report,
    )
    run_batch(pipe, rows)

    print(f"Done. Saved videos to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
