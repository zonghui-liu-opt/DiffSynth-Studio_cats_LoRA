import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DIT_GLOB = "diffusion_pytorch_model*.safetensors"
AUXILIARY_PATHS = (
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.2_VAE.pth",
    "google/umt5-xxl",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse a DiffSynth Wan2.2-TI2V-5B LoRA into a bf16 baseline DiT."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-shard-size-gb", type=float, default=4.0)
    parser.add_argument(
        "--aux-files-mode",
        choices=("symlink", "copy", "none"),
        default="symlink",
        help="How to place the text encoder, VAE, and tokenizer in OUTPUT_ROOT.",
    )
    parser.add_argument(
        "--skip-save-verification",
        action="store_true",
        help="Skip the exact tensor checksum verification after serialization.",
    )
    parser.add_argument(
        "--skip-reload-verification",
        action="store_true",
        help="Skip reloading the serialized DiT through DiffSynth's model loader.",
    )
    parser.add_argument(
        "--deterministic",
        choices=("warn", "strict"),
        default="strict",
    )
    return parser.parse_args()


def require_model_layout(model_root):
    model_root = Path(model_root).expanduser().resolve()
    dit_paths = sorted(model_root.glob(DIT_GLOB))
    if not dit_paths:
        raise FileNotFoundError(f"No DiT weights found: {model_root / DIT_GLOB}")
    for relative_path in AUXILIARY_PATHS:
        path = model_root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline model asset: {path}")
    return model_root, dit_paths


def validate_output_root(model_root, output_root):
    output_root = Path(output_root).expanduser().resolve()
    if output_root == model_root:
        raise ValueError("OUTPUT_ROOT must differ from MODEL_ROOT; the baseline is never modified in place")
    if output_root.exists():
        raise FileExistsError(
            f"OUTPUT_ROOT already exists: {output_root}. Choose a new path so no model is overwritten."
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    return output_root


def validate_lora_targets(model, converted_lora):
    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"
    a_names = {name[: -len(a_suffix)] for name in converted_lora if name.endswith(a_suffix)}
    b_names = {name[: -len(b_suffix)] for name in converted_lora if name.endswith(b_suffix)}
    if not a_names or not b_names:
        raise ValueError("The checkpoint does not contain any recognizable LoRA A/B pairs")
    if a_names != b_names:
        missing_a = sorted(b_names - a_names)
        missing_b = sorted(a_names - b_names)
        raise ValueError(f"Incomplete LoRA pairs: missing_A={missing_a}, missing_B={missing_b}")

    modules = dict(model.named_modules())
    unmatched = sorted(a_names - modules.keys())
    if unmatched:
        preview = ", ".join(unmatched[:8])
        raise ValueError(f"{len(unmatched)} LoRA targets do not exist in the baseline DiT: {preview}")

    for name in sorted(a_names):
        module = modules[name]
        if not hasattr(module, "weight"):
            raise ValueError(f"LoRA target has no weight tensor: {name}")
        weight_up = converted_lora[name + b_suffix]
        weight_down = converted_lora[name + a_suffix]
        if weight_up.ndim == 4:
            weight_up = weight_up.squeeze(3).squeeze(2)
            weight_down = weight_down.squeeze(3).squeeze(2)
        if weight_up.ndim != 2 or weight_down.ndim != 2:
            raise ValueError(f"Unsupported LoRA tensor rank at {name}")
        expected_shape = (weight_up.shape[0], weight_down.shape[1])
        if tuple(module.weight.shape) != expected_shape:
            raise ValueError(
                f"LoRA shape mismatch at {name}: update={expected_shape}, "
                f"baseline={tuple(module.weight.shape)}"
            )
    return sorted(a_names)


def build_shard_plan(state_dict, max_shard_size_bytes):
    if max_shard_size_bytes <= 0:
        raise ValueError("max_shard_size_bytes must be positive")
    shards = []
    current_names = []
    current_size = 0
    for name in sorted(state_dict):
        tensor = state_dict[name]
        tensor_size = tensor.numel() * tensor.element_size()
        if current_names and current_size + tensor_size > max_shard_size_bytes:
            shards.append(current_names)
            current_names = []
            current_size = 0
        current_names.append(name)
        current_size += tensor_size
    if current_names:
        shards.append(current_names)
    return shards


def update_tensor_checksum(digest, name, tensor):
    tensor = tensor.detach().contiguous().cpu()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(memoryview(tensor.view(torch.uint8).numpy()))


def save_sharded_state_dict(state_dict, output_root, max_shard_size_bytes):
    output_root = Path(output_root)
    shard_plan = build_shard_plan(state_dict, max_shard_size_bytes)
    shard_count = len(shard_plan)
    weight_map = {}
    total_size = 0
    expected_digest = hashlib.sha256()
    output_names = []

    for shard_id, tensor_names in enumerate(shard_plan, start=1):
        if shard_count == 1:
            filename = "diffusion_pytorch_model.safetensors"
        else:
            filename = f"diffusion_pytorch_model-{shard_id:05d}-of-{shard_count:05d}.safetensors"
        cpu_tensors = {}
        for name in tensor_names:
            tensor = state_dict[name].detach().contiguous().cpu()
            cpu_tensors[name] = tensor
            tensor_size = tensor.numel() * tensor.element_size()
            total_size += tensor_size
            weight_map[name] = filename
            update_tensor_checksum(expected_digest, name, tensor)
        save_file(cpu_tensors, str(output_root / filename), metadata={"format": "pt"})
        output_names.append(filename)
        del cpu_tensors

    if shard_count > 1:
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        index_path = output_root / "diffusion_pytorch_model.safetensors.index.json"
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return output_names, weight_map, expected_digest.hexdigest(), total_size


def checksum_saved_state_dict(output_root, weight_map):
    digest = hashlib.sha256()
    files_to_names = {}
    for name, filename in weight_map.items():
        files_to_names.setdefault(filename, []).append(name)
    for filename in sorted(files_to_names):
        with safe_open(output_root / filename, framework="pt", device="cpu") as handle:
            for name in sorted(files_to_names[filename]):
                update_tensor_checksum(digest, name, handle.get_tensor(name))
    return digest.hexdigest()


def materialize_auxiliary_files(model_root, output_root, mode):
    if mode == "none":
        return []
    materialized = []
    for relative_path in AUXILIARY_PATHS:
        source = model_root / relative_path
        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == "symlink":
            destination.symlink_to(source, target_is_directory=source.is_dir())
        elif source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        materialized.append(relative_path)
    return materialized


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.max_shard_size_gb <= 0:
        raise ValueError("--max-shard-size-gb must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the bf16 runtime LoRA fusion path")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(
        True, warn_only=args.deterministic == "warn"
    )

    model_root, dit_paths = require_model_layout(args.model_root)
    lora_path = args.lora_path.expanduser().resolve()
    if not lora_path.is_file():
        raise FileNotFoundError(f"Missing LoRA checkpoint: {lora_path}")
    output_root = validate_output_root(model_root, args.output_root)
    staging_root = output_root.parent / f".{output_root.name}.merging-{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(f"Temporary merge directory already exists: {staging_root}")
    staging_root.mkdir(parents=True)

    try:
        from diffsynth.core import ModelConfig, load_state_dict
        from diffsynth.core.loader import hash_model_file
        from diffsynth.pipelines.wan_video import WanVideoPipeline
        from diffsynth.utils.lora import GeneralLoRALoader

        print(f"Loading baseline DiT from {model_root}")
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[ModelConfig(path=[str(path) for path in dit_paths])],
            tokenizer_config=None,
        )
        if pipe.dit is None:
            raise RuntimeError("The baseline files were not detected as a Wan video DiT")

        raw_lora = load_state_dict(
            str(lora_path), torch_dtype=torch.bfloat16, device="cuda"
        )
        converted_lora = GeneralLoRALoader(
            torch_dtype=torch.bfloat16, device="cuda"
        ).convert_state_dict(raw_lora)
        matched_layers = validate_lora_targets(pipe.dit, converted_lora)
        print(f"Validated {len(matched_layers)} LoRA target layers")

        # Use the exact same DiffSynth fusion entry point as runtime LoRA inference.
        pipe.load_lora(pipe.dit, state_dict=raw_lora, alpha=args.alpha)
        del raw_lora, converted_lora

        max_shard_size_bytes = int(args.max_shard_size_gb * 1024**3)
        output_names, weight_map, expected_checksum, total_size = save_sharded_state_dict(
            pipe.dit.state_dict(), staging_root, max_shard_size_bytes
        )
        if not args.skip_save_verification:
            saved_checksum = checksum_saved_state_dict(staging_root, weight_map)
            if saved_checksum != expected_checksum:
                raise RuntimeError(
                    "Saved merged weights failed exact checksum verification: "
                    f"memory={expected_checksum}, disk={saved_checksum}"
                )
        else:
            saved_checksum = None

        baseline_model_hash = hash_model_file([str(path) for path in dit_paths])
        merged_paths = [str(staging_root / name) for name in output_names]
        merged_model_hash = hash_model_file(merged_paths)
        if merged_model_hash != baseline_model_hash:
            raise RuntimeError(
                "Merged checkpoint keys/shapes differ from the baseline: "
                f"baseline={baseline_model_hash}, merged={merged_model_hash}"
            )

        if not args.skip_reload_verification:
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            reloaded_pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[ModelConfig(path=merged_paths)],
                tokenizer_config=None,
            )
            if reloaded_pipe.dit is None:
                raise RuntimeError("DiffSynth failed to reload the serialized merged DiT")
            del reloaded_pipe
            gc.collect()
            torch.cuda.empty_cache()

        auxiliary_files = materialize_auxiliary_files(
            model_root, staging_root, args.aux_files_mode
        )
        manifest = {
            "format": "DiffSynth-Studio Wan2.2-TI2V-5B merged LoRA",
            "baseline_model_root": str(model_root),
            "baseline_dit_files": [str(path) for path in dit_paths],
            "lora_path": str(lora_path),
            "lora_sha256": sha256_file(lora_path),
            "lora_alpha": args.alpha,
            "merge_dtype": "torch.bfloat16",
            "matched_lora_layers": len(matched_layers),
            "merged_dit_files": output_names,
            "merged_total_size_bytes": total_size,
            "merged_state_sha256": expected_checksum,
            "save_checksum_verified": not args.skip_save_verification,
            "reload_verified": not args.skip_reload_verification,
            "model_key_hash": merged_model_hash,
            "deterministic": args.deterministic,
            "aux_files_mode": args.aux_files_mode,
            "auxiliary_paths": auxiliary_files,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(),
        }
        (staging_root / "merge_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        staging_root.rename(output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    print(f"Merged model ready: {output_root}")
    print(f"Merged state checksum: {expected_checksum}")
    if args.aux_files_mode == "none":
        print("Note: auxiliary files were not materialized; this directory is not standalone for inference.")


if __name__ == "__main__":
    main()
