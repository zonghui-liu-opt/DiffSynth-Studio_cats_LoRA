from pathlib import Path


def test_build_model_configs_uses_training_launcher_model_root_layout(tmp_path):
    from infer_cats_ti2v5b_lora import build_model_configs

    model_root = tmp_path / "Wan2.2-TI2V-5B"
    model_root.mkdir()
    (model_root / "diffusion_pytorch_model-00001-of-00002.safetensors").touch()
    (model_root / "diffusion_pytorch_model-00002-of-00002.safetensors").touch()
    (model_root / "models_t5_umt5-xxl-enc-bf16.pth").touch()
    (model_root / "Wan2.2_VAE.pth").touch()
    (model_root / "google" / "umt5-xxl").mkdir(parents=True)

    model_configs, tokenizer_config = build_model_configs(model_root)

    assert model_configs[0].path == [
        str(model_root / "diffusion_pytorch_model-00001-of-00002.safetensors"),
        str(model_root / "diffusion_pytorch_model-00002-of-00002.safetensors"),
    ]
    assert model_configs[1].path == str(model_root / "models_t5_umt5-xxl-enc-bf16.pth")
    assert model_configs[2].path == str(model_root / "Wan2.2_VAE.pth")
    assert tokenizer_config.path == str(model_root / "google" / "umt5-xxl")


def test_build_model_configs_reports_missing_dit_glob(tmp_path):
    from infer_cats_ti2v5b_lora import build_model_configs

    model_root = tmp_path / "Wan2.2-TI2V-5B"
    model_root.mkdir()

    try:
        build_model_configs(model_root)
    except FileNotFoundError as exc:
        assert str(model_root / "diffusion_pytorch_model*.safetensors") in str(exc)
    else:
        raise AssertionError("Expected missing DiT weights to raise FileNotFoundError")
