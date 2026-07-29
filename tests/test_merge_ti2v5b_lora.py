import torch

from merge_ti2v5b_lora import (
    build_shard_plan,
    checksum_saved_state_dict,
    save_sharded_state_dict,
    validate_lora_targets,
)


def test_save_sharded_state_dict_round_trips_exactly(tmp_path):
    state_dict = {
        "a.weight": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
        "b.weight": torch.arange(24, dtype=torch.bfloat16).reshape(6, 4),
    }

    output_names, weight_map, expected_checksum, total_size = save_sharded_state_dict(
        state_dict, tmp_path, max_shard_size_bytes=40
    )

    assert output_names == [
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    ]
    assert checksum_saved_state_dict(tmp_path, weight_map) == expected_checksum
    assert total_size == 80
    assert (tmp_path / "diffusion_pytorch_model.safetensors.index.json").is_file()


def test_build_shard_plan_keeps_an_oversized_tensor_whole():
    state_dict = {
        "large": torch.zeros(100, dtype=torch.float32),
        "small": torch.zeros(1, dtype=torch.float32),
    }

    assert build_shard_plan(state_dict, max_shard_size_bytes=16) == [
        ["large"],
        ["small"],
    ]


def test_validate_lora_targets_checks_names_and_shapes():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    converted_lora = {
        "0.lora_A.weight": torch.zeros(1, 3),
        "0.lora_B.weight": torch.zeros(2, 1),
    }

    assert validate_lora_targets(model, converted_lora) == ["0"]
