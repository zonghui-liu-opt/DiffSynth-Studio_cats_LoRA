"""Exercise batch orchestration without loading Wan weights or invoking ffmpeg."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

import infer_batch


class FakeVAE:
    def __init__(self):
        self.calls = 0

    def decode(self, latents, **kwargs):
        self.calls += 1
        return [Image.new("RGB", (8, 8))]


class FakePipeline:
    device = "cpu"

    def __init__(self, fail_at=None, before_call=None):
        self.vae = FakeVAE()
        self.scheduler = SimpleNamespace(timesteps=[])
        self.calls = []
        self.model_calls = 0
        self.fail_at = fail_at
        self.before_call = before_call

    def model_fn(self, *args, **kwargs):
        self.model_calls += 1
        return "noise prediction"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.before_call is not None:
            self.before_call(len(self.calls))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("synthetic inference failure")

        self.scheduler.timesteps = list(range(kwargs["num_inference_steps"]))
        for timestep in self.scheduler.timesteps:
            self.model_fn(timestep=timestep)
            if kwargs["cfg_scale"] != 1 and not kwargs["cfg_merge"]:
                self.model_fn(timestep=timestep)
        return self.vae.decode("latents", tiled=kwargs["tiled"])


@pytest.fixture
def batch_env(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_root.mkdir()
    output_dir.mkdir()
    settings = {
        "DATA_ROOT": data_root,
        "OUTPUT_DIR": output_dir,
        "TIMING_ENABLED": True,
        "WARMUP_RUNS": 0,
        "REPEATS": 1,
        "HEIGHT": 96,
        "WIDTH": 160,
        "NUM_FRAMES": 9,
        "SEED": 17,
        "FPS": 24,
        "VIDEO_QUALITY": 5,
        "NUM_INFERENCE_STEPS": 2,
        "CFG_SCALE": 5.0,
        "CFG_MERGE": False,
        "NEGATIVE_PROMPT": "blurry",
        "SIGMA_SHIFT": 5.0,
        "TILED": True,
    }
    for name, value in settings.items():
        monkeypatch.setattr(infer_batch, name, value)

    saved = []

    def fake_save_video(video, path, **kwargs):
        saved.append((Path(path), kwargs))
        Path(path).write_bytes(b"fake mp4")

    monkeypatch.setattr(infer_batch, "save_video", fake_save_video)

    def make_row(stem, height=None, width=None):
        image_path = data_root / f"{stem}.png"
        # Deliberately differ from the inference size and mode to verify preparation.
        Image.new("RGBA", (21, 13), color=(128, 64, 32, 255)).save(image_path)
        row = {"prompt": f"A cat named {stem}", "input_image": image_path.name}
        if height is not None:
            row["height"] = str(height)
        if width is not None:
            row["width"] = str(width)
        return row

    return SimpleNamespace(
        data_root=data_root, output_dir=output_dir, saved=saved, make_row=make_row
    )


def read_timing_csv(output_dir):
    with (output_dir / "timing.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_summary(output_dir):
    return json.loads((output_dir / "timing_summary.json").read_text(encoding="utf-8"))


def assert_metric_summary(summary, records, metric):
    values = [float(record[metric]) for record in records]
    stats = summary[metric]
    assert set(stats) >= {"total", "mean", "min", "max", "median", "p95"}
    assert stats["total"] == pytest.approx(sum(values))
    assert stats["mean"] == pytest.approx(sum(values) / len(values))
    assert stats["min"] == pytest.approx(min(values))
    assert stats["max"] == pytest.approx(max(values))
    assert 0 <= stats["min"] <= stats["median"] <= stats["p95"] <= stats["max"]


def test_single_run_preserves_filename_and_forwards_inference_settings(batch_env):
    row = batch_env.make_row("cat")
    pipe = FakePipeline()

    records = infer_batch.run_batch(pipe, [row])

    assert [path.name for path, _ in batch_env.saved] == ["0000_cat.mp4"]
    assert batch_env.saved[0][1] == {"fps": 24, "quality": 5}
    assert len(records) == len(pipe.calls) == 1
    call = pipe.calls[0]
    for name, expected in {
        "prompt": row["prompt"],
        "negative_prompt": "blurry",
        "height": 96,
        "width": 160,
        "num_frames": 9,
        "seed": 17,
        "num_inference_steps": 2,
        "cfg_scale": 5.0,
        "cfg_merge": False,
        "sigma_shift": 5.0,
        "tiled": True,
    }.items():
        assert call[name] == expected
    assert call["input_image"].mode == "RGB"
    assert call["input_image"].size == (160, 96)
    assert pipe.model_calls == 4  # Both CFG branches are part of DiT timing.
    assert pipe.vae.calls == 1
    assert len(read_timing_csv(batch_env.output_dir)) == 1


def test_warmups_are_per_resolution_and_repeats_have_separate_reports(batch_env, monkeypatch):
    monkeypatch.setattr(infer_batch, "WARMUP_RUNS", 2)
    monkeypatch.setattr(infer_batch, "REPEATS", 2)
    rows = [
        batch_env.make_row("landscape", 96, 160),
        batch_env.make_row("portrait", 160, 96),
        batch_env.make_row("landscape_again", 96, 160),
    ]
    pipe = FakePipeline()

    records = infer_batch.run_batch(pipe, rows)

    assert [call["prompt"] for call in pipe.calls] == (
        [rows[0]["prompt"]] * 4 + [rows[1]["prompt"]] * 4 + [rows[2]["prompt"]] * 2
    )
    assert {call["seed"] for call in pipe.calls} == {17}
    assert [path.name for path, _ in batch_env.saved] == [
        "0000_landscape_r01.mp4",
        "0000_landscape_r02.mp4",
        "0001_portrait_r01.mp4",
        "0001_portrait_r02.mp4",
        "0002_landscape_again_r01.mp4",
        "0002_landscape_again_r02.mp4",
    ]
    assert len(records) == len(read_timing_csv(batch_env.output_dir)) == 6
    assert pipe.vae.calls == 10
    assert pipe.model_calls == 40

    summary = read_summary(batch_env.output_dir)
    assert summary["status"] == "complete"
    assert summary["completed_samples"] == summary["planned_samples"] == 6
    assert summary["overall"]["count"] == 6
    assert summary["by_resolution"]["96x160"]["count"] == 4
    assert summary["by_resolution"]["160x96"]["count"] == 2
    for metric in ("dit_seconds", "vae_decode_seconds"):
        assert_metric_summary(summary["overall"], records, metric)
        for resolution, group in summary["by_resolution"].items():
            height, width = map(int, resolution.split("x"))
            group_records = [
                record for record in records
                if int(record["height"]) == height and int(record["width"]) == width
            ]
            assert_metric_summary(group, group_records, metric)


@pytest.mark.parametrize("cfg_scale,cfg_merge,expected_calls", [(1.0, False, 2), (5.0, True, 2)])
def test_cfg_modes_are_forwarded_without_forcing_two_model_calls(
    batch_env, monkeypatch, cfg_scale, cfg_merge, expected_calls
):
    monkeypatch.setattr(infer_batch, "CFG_SCALE", cfg_scale)
    monkeypatch.setattr(infer_batch, "CFG_MERGE", cfg_merge)
    pipe = FakePipeline()

    infer_batch.run_batch(pipe, [batch_env.make_row("cat")])

    assert pipe.calls[0]["cfg_scale"] == cfg_scale
    assert pipe.calls[0]["cfg_merge"] is cfg_merge
    assert pipe.model_calls == expected_calls


def test_failure_flushes_successful_rows_and_writes_failed_summary(batch_env):
    rows = [batch_env.make_row("first"), batch_env.make_row("second")]

    def check_flushed_progress(call_number):
        if call_number == 2:
            # The file must already be readable while the batch is still running.
            assert len(read_timing_csv(batch_env.output_dir)) == 1

    pipe = FakePipeline(fail_at=2, before_call=check_flushed_progress)
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        infer_batch.run_batch(pipe, rows)

    assert [path.name for path, _ in batch_env.saved] == ["0000_first.mp4"]
    assert len(read_timing_csv(batch_env.output_dir)) == 1
    summary = read_summary(batch_env.output_dir)
    assert summary["status"] == "failed"
    assert summary["completed_samples"] == 1
    assert summary["planned_samples"] == 2
    assert summary["overall"]["count"] == 1


def test_disabled_timing_still_saves_videos_without_timing_artifacts(batch_env, monkeypatch):
    monkeypatch.setattr(infer_batch, "TIMING_ENABLED", False)
    pipe = FakePipeline()

    infer_batch.run_batch(pipe, [batch_env.make_row("cat")])

    assert [path.name for path, _ in batch_env.saved] == ["0000_cat.mp4"]
    assert len(pipe.calls) == 1
    assert not (batch_env.output_dir / "timing.csv").exists()
    assert not (batch_env.output_dir / "timing_summary.json").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt": "   "},
        {"input_image": ""},
        {"height": "96"},
        {"height": "zero", "width": "160"},
        {"height": "0", "width": "160"},
        {"height": "95", "width": "160"},
    ],
)
def test_validate_rows_rejects_bad_input_before_model_loading(batch_env, changes):
    row = batch_env.make_row("cat")
    row.update(changes)

    with pytest.raises(ValueError):
        infer_batch.validate_rows([row])


def test_validate_rows_rejects_missing_images(batch_env):
    row = {"prompt": "A cat", "input_image": "missing.png"}

    with pytest.raises(FileNotFoundError):
        infer_batch.validate_rows([row])


def test_rejects_cfg_merge_without_guidance_before_loading_model(batch_env, monkeypatch):
    monkeypatch.setattr(infer_batch, "CFG_SCALE", 1.0)
    monkeypatch.setattr(infer_batch, "CFG_MERGE", True)
    with pytest.raises(ValueError, match="CFG_MERGE=0"):
        infer_batch.validate_settings()


def test_print_lora_rank_accepts_rank64_in_both_checkpoint_naming_styles(tmp_path, monkeypatch):
    monkeypatch.setattr(infer_batch, "EXPECTED_LORA_RANK", 64)
    checkpoint = tmp_path / "rank64.safetensors"
    save_file(
        {
            "dit.blocks.0.self_attn.q.lora_A.weight": torch.zeros(64, 128),
            "dit.blocks.0.self_attn.q.lora_B.weight": torch.zeros(128, 64),
            "dit.blocks.0.self_attn.k.lora_down.weight": torch.zeros(64, 128),
            "dit.blocks.0.self_attn.k.lora_up.weight": torch.zeros(128, 64),
        },
        checkpoint,
    )

    assert infer_batch.print_lora_rank(checkpoint) == [64]


@pytest.mark.parametrize("ranks", [(32,), (32, 64)])
def test_print_lora_rank_rejects_wrong_or_mixed_rank(tmp_path, monkeypatch, ranks):
    monkeypatch.setattr(infer_batch, "EXPECTED_LORA_RANK", 64)
    checkpoint = tmp_path / "wrong_rank.safetensors"
    save_file(
        {
            f"dit.blocks.{index}.self_attn.q.lora_A.weight": torch.zeros(rank, 128)
            for index, rank in enumerate(ranks)
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="Expected LoRA rank 64"):
        infer_batch.print_lora_rank(checkpoint)


def test_print_lora_rank_rejects_checkpoint_without_lora_tensors(tmp_path, monkeypatch):
    monkeypatch.setattr(infer_batch, "EXPECTED_LORA_RANK", 64)
    checkpoint = tmp_path / "base_model.safetensors"
    save_file({"dit.blocks.0.self_attn.q.weight": torch.zeros(128, 128)}, checkpoint)

    with pytest.raises(ValueError, match="no LoRA tensors"):
        infer_batch.print_lora_rank(checkpoint)
