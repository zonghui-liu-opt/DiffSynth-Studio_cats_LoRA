import json
import shutil

import pytest
import torch
from safetensors.torch import save_file

from inference_model_checks import inspect_dit_checkpoint, validate_lora_layout


def save_weights(path, tensors):
    save_file({key: torch.zeros(shape) for key, shape in tensors.items()}, path)


def register_tiny_model(monkeypatch, paths):
    import diffsynth.configs
    from diffsynth.core.loader import hash_model_file

    model_hash = hash_model_file([str(path) for path in paths])
    monkeypatch.setattr(diffsynth.configs, "MODEL_CONFIGS", [{
        "model_hash": model_hash, "model_name": "wan_video_dit",
        "extra_kwargs": {"dim": 3072, "out_dim": 48, "fuse_vae_embedding_in_latents": True},
    }])
    return model_hash


def test_index_selects_exact_shards_and_checks_tensor_locations(tmp_path, monkeypatch):
    paths = [tmp_path / f"diffusion_pytorch_model-{i:05d}-of-00002.safetensors" for i in (1, 2)]
    save_weights(paths[0], {"first.weight": (2, 3)})
    save_weights(paths[1], {"last.weight": (3, 2)})
    expected_hash = register_tiny_model(monkeypatch, paths)
    index = tmp_path / "diffusion_pytorch_model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"first.weight": paths[0].name, "last.weight": paths[1].name}}))
    # An old checkpoint in the same directory must not be mixed into an indexed model.
    save_weights(tmp_path / "diffusion_pytorch_model.safetensors", {"old.weight": (1,)})

    selected, shapes, model_hash = inspect_dit_checkpoint(tmp_path)
    assert selected == paths
    assert shapes == {"first.weight": (2, 3), "last.weight": (3, 2)}
    assert model_hash == expected_hash
    index.write_text(json.dumps({"weight_map": {"first.weight": paths[1].name, "last.weight": paths[0].name}}))
    with pytest.raises(ValueError, match="do not match weight_map"):
        inspect_dit_checkpoint(tmp_path)


def test_missing_shard_fails_before_model_loading(tmp_path):
    path = tmp_path / "diffusion_pytorch_model-00001-of-00002.safetensors"
    save_weights(path, {"a": (1,)})
    with pytest.raises(ValueError, match="Incomplete DiT shards"):
        inspect_dit_checkpoint(tmp_path)
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": path.name, "b": "diffusion_pytorch_model-00002-of-00002.safetensors"},
    }))
    with pytest.raises(FileNotFoundError, match="Missing DiT shard"):
        inspect_dit_checkpoint(tmp_path)


def test_duplicate_tensor_and_wrong_model_are_rejected(tmp_path):
    first = tmp_path / "diffusion_pytorch_model-00001-of-00002.safetensors"
    second = tmp_path / "diffusion_pytorch_model-00002-of-00002.safetensors"
    save_weights(first, {"a": (1,)})
    shutil.copy(first, second)
    with pytest.raises(ValueError, match="Duplicate tensor"):
        inspect_dit_checkpoint(tmp_path)
    first.rename(tmp_path / "diffusion_pytorch_model.safetensors")
    second.unlink()
    with pytest.raises(ValueError, match="not a complete Wan2.2-TI2V-5B"):
        inspect_dit_checkpoint(tmp_path)


@pytest.mark.parametrize("a,b", [("lora_A", "lora_B"), ("lora_A.default", "lora_B.default"), ("lora_down", "lora_up")])
def test_lora_header_validation_uses_actual_loader_names(tmp_path, a, b):
    path = tmp_path / "lora.safetensors"
    save_weights(path, {f"diffusion_model.blocks.0.q.{a}.weight": (64, 8), f"diffusion_model.blocks.0.q.{b}.weight": (4, 64)})
    assert validate_lora_layout(path, {"blocks.0.q.weight": (4, 8)}) == ["blocks.0.q"]


@pytest.mark.parametrize("case", ["unknown_target", "wrong_shape", "missing_a", "missing_b", "wrong_rank"])
def test_lora_mismatch_cannot_silently_run_base_model(tmp_path, case):
    path = tmp_path / "lora.safetensors"
    shapes = {"q.lora_A.weight": (64, 8), "q.lora_B.weight": (4, 64)}
    base = {"q.weight": (4, 8)}
    if case == "unknown_target":
        base = {"k.weight": (4, 8)}
    elif case == "wrong_shape":
        base = {"q.weight": (5, 8)}
    elif case == "missing_a":
        del shapes["q.lora_A.weight"]
    elif case == "missing_b":
        del shapes["q.lora_B.weight"]
    else:
        shapes["q.lora_B.weight"] = (4, 32)
    save_weights(path, shapes)
    with pytest.raises(ValueError):
        validate_lora_layout(path, base)
