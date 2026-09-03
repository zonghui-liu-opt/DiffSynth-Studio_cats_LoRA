"""Read checkpoint headers before allocating Wan models on the GPU."""

import json
import re
from pathlib import Path

from safetensors import safe_open


def read_tensor_shapes(paths):
    shapes, locations = {}, {}
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key in shapes:
                    raise ValueError(f"Duplicate tensor {key}: {locations[key]} and {path}")
                shapes[key] = tuple(checkpoint.get_slice(key).get_shape())
                locations[key] = Path(path).name
    if not shapes:
        raise ValueError("Checkpoint contains no tensors")
    return shapes, locations


def inspect_dit_checkpoint(model_root):
    """Resolve complete shards and recognize the same model as DiffSynth's loader."""
    from diffsynth.configs import MODEL_CONFIGS
    from diffsynth.core.loader import hash_model_file

    model_root = Path(model_root)
    index_path = model_root / "diffusion_pytorch_model.safetensors.index.json"
    weight_map = None
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Empty or invalid weight_map in {index_path}")
        filenames = set(weight_map.values())
        for filename in filenames:
            if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".safetensors"):
                raise ValueError(f"Invalid shard filename in {index_path}: {filename}")
        paths = [model_root / filename for filename in sorted(filenames)]
    else:
        paths = sorted(model_root.glob("diffusion_pytorch_model*.safetensors"))
        shards = [re.fullmatch(r"diffusion_pytorch_model-(\d+)-of-(\d+)\.safetensors", path.name) for path in paths]
        if any(shards):
            totals = {int(match[2]) for match in shards if match}
            if not all(shards) or len(totals) != 1:
                raise ValueError("Mixed DiT shard sets; use one complete model directory")
            total = totals.pop()
            indices = {int(match[1]) for match in shards}
            if total < 1 or len(paths) != total or indices != set(range(1, total + 1)):
                raise ValueError(f"Incomplete DiT shards: found {len(paths)}, expected {total}")
    if not paths:
        raise FileNotFoundError(f"No DiT safetensors found under {model_root}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing DiT shard: {path}")
    shapes, locations = read_tensor_shapes(paths)
    if weight_map is not None and locations != weight_map:
        raise ValueError(f"DiT shard contents do not match weight_map: {index_path}")
    model_hash = hash_model_file([str(path) for path in paths])
    supported = {
        config["model_hash"] for config in MODEL_CONFIGS
        if config["model_name"] == "wan_video_dit"
        and config.get("extra_kwargs", {}).get("dim") == 3072
        and config.get("extra_kwargs", {}).get("out_dim") == 48
        and config.get("extra_kwargs", {}).get("fuse_vae_embedding_in_latents")
    }
    if model_hash not in supported:
        raise ValueError(
            f"DiT is not a complete Wan2.2-TI2V-5B checkpoint supported by this checkout "
            f"(structure hash={model_hash}); check model variant and copied shards"
        )
    return paths, shapes, model_hash


def validate_lora_layout(lora_path, dit_shapes):
    """Reject partial, wrong-model and zero-match adapters using the loader's names."""
    from diffsynth.utils.lora import GeneralLoRALoader

    shapes, _ = read_tensor_shapes([lora_path])
    names = GeneralLoRALoader().get_name_dict(shapes)
    if not names:
        raise ValueError("The checkpoint contains no complete recognizable LoRA A/B pairs")
    adapter_keys = {
        key for key in shapes
        if any(part in key for part in (".lora_A.", ".lora_B.", ".lora_down.", ".lora_up."))
    }
    paired_keys = {key for up, down, _ in names.values() for key in (up, down)}
    if adapter_keys != paired_keys:
        raise ValueError("Incomplete or duplicate LoRA pairs; every A/down and B/up must have one partner")
    for target, (up_key, down_key, _) in names.items():
        if up_key not in shapes or down_key not in shapes:
            raise ValueError(f"Incomplete LoRA pair at {target}")
        up, down = shapes[up_key], shapes[down_key]
        if len(up) != 2 or len(down) != 2 or up[1] != down[0]:
            raise ValueError(f"Invalid LoRA A/B matrix shapes at {target}: A={down}, B={up}")
        base_shape = dit_shapes.get(target + ".weight")
        if base_shape is None:
            raise ValueError(f"LoRA target does not exist in this DiT: {target}; check checkpoint model and saved key prefix")
        if tuple(base_shape) != (up[0], down[1]):
            raise ValueError(f"LoRA shape mismatch at {target}: A={down}, B={up}, DiT={base_shape}")
    return sorted(names)


def validate_loaded_pipeline(pipe, lora_targets=()):
    missing = [name for name in ("dit", "vae", "text_encoder", "tokenizer") if getattr(pipe, name, None) is None]
    if missing:
        raise ValueError(f"Required Wan components failed to load: {', '.join(missing)}")
    if not getattr(pipe.dit, "fuse_vae_embedding_in_latents", False) or pipe.vae.upsampling_factor != 16:
        raise ValueError("Loaded DiT/VAE do not form a Wan2.2-TI2V-5B pipeline")
    modules = dict(pipe.dit.named_modules())
    unmatched = [name for name in lora_targets if name not in modules or not hasattr(modules[name], "weight")]
    if unmatched:
        raise ValueError(f"LoRA targets missing from loaded DiT: {unmatched[:8]}")
