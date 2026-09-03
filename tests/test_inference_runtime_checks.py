"""Failure-path tests with explicit doubles; CUDA tests do not execute a GPU."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import inference_runtime_checks as runtime


def replace_imports(monkeypatch, **modules):
    original = runtime.import_module
    monkeypatch.setattr(
        runtime, "import_module", lambda name: modules[name] if name in modules else original(name)
    )


@pytest.fixture
def fake_video_io(monkeypatch):
    reader = SimpleNamespace(
        get_data=Mock(return_value=np.zeros((32, 32, 3), dtype=np.uint8)),
        close=Mock(),
    )

    def encode(frames, destination, **kwargs):
        assert len(frames) == 2
        assert all(frame.size == (32, 32) for frame in frames)
        Path(destination).write_bytes(b"synthetic non-empty MP4 for a mocked decoder")

    save_video = Mock(side_effect=encode)
    get_reader = Mock(return_value=reader)
    replace_imports(
        monkeypatch,
        **{
            "diffsynth.utils.data": SimpleNamespace(save_video=save_video),
            "imageio": SimpleNamespace(get_reader=get_reader),
        },
    )
    return SimpleNamespace(save_video=save_video, get_reader=get_reader, reader=reader)


def test_video_writer_encodes_and_reads_two_frames_with_doubles(tmp_path, fake_video_io):
    output = tmp_path / "new_output"
    report = runtime.check_video_writer(output, fps=24, quality=5)
    assert report["status"] == "passed"
    assert report["decoded_frames"] == 2
    assert report["encoded_bytes"] > 0
    assert list(output.iterdir()) == []
    assert fake_video_io.save_video.call_args.kwargs == {"fps": 24, "quality": 5}
    assert fake_video_io.get_reader.call_args.kwargs == {"format": "ffmpeg"}
    assert [call.args for call in fake_video_io.reader.get_data.call_args_list] == [(0,), (1,)]
    fake_video_io.reader.close.assert_called_once()
    json.dumps(report)


def test_video_encoder_error_keeps_cause_and_cleans_temp_directory(tmp_path, fake_video_io):
    original = OSError("ffmpeg executable unavailable")
    fake_video_io.save_video.side_effect = original
    with pytest.raises(RuntimeError, match="MP4 encoding.*ffmpeg executable unavailable") as error:
        runtime.check_video_writer(tmp_path, 24, 5)
    assert error.value.__cause__ is original
    assert list(tmp_path.iterdir()) == []
    fake_video_io.get_reader.assert_not_called()


@pytest.mark.parametrize("create_empty_file", [True, False])
def test_video_encoder_empty_or_missing_file_fails(tmp_path, fake_video_io, create_empty_file):
    def bad_encode(frames, destination, **kwargs):
        if create_empty_file:
            Path(destination).touch()

    fake_video_io.save_video.side_effect = bad_encode
    with pytest.raises(RuntimeError, match="MP4 encoding.*non-empty MP4"):
        runtime.check_video_writer(tmp_path, 24, 5)
    assert list(tmp_path.iterdir()) == []
    fake_video_io.get_reader.assert_not_called()


def test_corrupt_encoded_video_fails_decode_and_cleans_up(tmp_path, fake_video_io):
    original = OSError("invalid MP4 data")
    fake_video_io.get_reader.side_effect = original
    with pytest.raises(RuntimeError, match="MP4 decoding.*invalid MP4 data") as error:
        runtime.check_video_writer(tmp_path, 24, 5)
    assert error.value.__cause__ is original
    assert list(tmp_path.iterdir()) == []


def test_video_requires_second_frame_and_closes_reader(tmp_path, fake_video_io):
    original = IndexError("second frame missing")
    fake_video_io.reader.get_data.side_effect = [np.zeros((32, 32, 3)), original]
    with pytest.raises(RuntimeError, match="MP4 decoding.*second frame missing") as error:
        runtime.check_video_writer(tmp_path, 24, 5)
    assert error.value.__cause__ is original
    fake_video_io.reader.close.assert_called_once()
    assert list(tmp_path.iterdir()) == []


def test_video_output_path_must_be_a_directory(tmp_path):
    output = tmp_path / "is_a_file"
    output.write_text("existing output", encoding="utf-8")
    with pytest.raises(RuntimeError, match="output directory") as error:
        runtime.check_video_writer(output, 24, 5)
    assert isinstance(error.value.__cause__, FileExistsError)
    assert output.read_text(encoding="utf-8") == "existing output"


def test_tokenizer_is_local_only_and_exercises_english_chinese(tmp_path, monkeypatch):
    ids = torch.ones((2, 512), dtype=torch.int64)
    tokenizer = Mock(return_value=(ids, ids))
    tokenizer.vocab_size = 256000
    constructor = Mock(return_value=tokenizer)
    replace_imports(
        monkeypatch,
        **{
            "diffsynth.models.wan_video_text_encoder": SimpleNamespace(
                HuggingfaceTokenizer=constructor
            )
        },
    )
    report = runtime.check_tokenizer(tmp_path)
    constructor.assert_called_once_with(
        name=str(tmp_path.resolve()), seq_len=512, clean="whitespace", local_files_only=True
    )
    prompts = tokenizer.call_args.args[0]
    assert len(prompts) == 2
    assert prompts[0].isascii()
    assert not prompts[1].isascii()
    assert tokenizer.call_args.kwargs == {"return_mask": True}
    assert report["token_counts"] == [512, 512]
    json.dumps(report)


def test_tokenizer_failure_keeps_original_cause(tmp_path, monkeypatch):
    original = OSError("local sentencepiece tokenizer is incomplete")
    replace_imports(
        monkeypatch,
        **{
            "diffsynth.models.wan_video_text_encoder": SimpleNamespace(
                HuggingfaceTokenizer=Mock(side_effect=original)
            )
        },
    )
    with pytest.raises(RuntimeError, match="tokenizer local load.*incomplete") as error:
        runtime.check_tokenizer(tmp_path)
    assert error.value.__cause__ is original


def test_tokenizer_rejects_a_nonexistent_local_directory(tmp_path):
    with pytest.raises(RuntimeError, match="Local tokenizer directory not found") as error:
        runtime.check_tokenizer(tmp_path / "missing")
    assert isinstance(error.value.__cause__, FileNotFoundError)


@pytest.fixture
def fake_cuda_runtime(monkeypatch):
    class Model:
        def load_state_dict(self, state_dict, strict=True, assign=False):
            pass

    cuda = SimpleNamespace(
        is_available=Mock(return_value=True),
        is_bf16_supported=Mock(return_value=True),
        current_device=Mock(return_value=1),
        get_device_properties=Mock(return_value=SimpleNamespace(name="Mock GPU", major=9, minor=0)),
        mem_get_info=Mock(return_value=(123456, 789012)),
        synchronize=Mock(),
    )

    # Tensor calculations below execute on CPU; the CUDA dispatch is a double.
    def cpu_ones(shape, *, device, dtype):
        assert device == "cuda:1"
        assert dtype == torch.bfloat16
        return torch.ones(shape, dtype=dtype)

    fake_torch = SimpleNamespace(
        __version__="test-double",
        version=SimpleNamespace(cuda="mock CUDA"),
        nn=SimpleNamespace(Module=Model),
        cuda=cuda,
        no_grad=torch.no_grad,
        ones=Mock(side_effect=cpu_ones),
        isfinite=torch.isfinite,
        bfloat16=torch.bfloat16,
    )
    attention = SimpleNamespace(
        FLASH_ATTN_3_AVAILABLE=False,
        FLASH_ATTN_2_AVAILABLE=False,
        SAGE_ATTN_AVAILABLE=False,
        flash_attention=Mock(side_effect=lambda q, k, v, num_heads: q.clone()),
    )
    replace_imports(monkeypatch, **{"torch": fake_torch, "diffsynth.models.wan_video_dit": attention})
    tokenizer_check = Mock(return_value={"status": "passed"})
    video_check = Mock(return_value={"status": "passed"})
    monkeypatch.setattr(runtime, "check_tokenizer", tokenizer_check)
    monkeypatch.setattr(runtime, "check_video_writer", video_check)
    return SimpleNamespace(
        torch=fake_torch, cuda=cuda, attention=attention,
        tokenizer_check=tokenizer_check, video_check=video_check,
    )


def test_runtime_rejects_missing_cuda_before_tokenizer_and_encoding(tmp_path, fake_cuda_runtime):
    env = fake_cuda_runtime
    env.cuda.is_available.return_value = False
    with pytest.raises(RuntimeError, match="CUDA availability.*CUDA is unavailable"):
        runtime.check_runtime(tmp_path, tmp_path, 24, 5)
    env.cuda.synchronize.assert_not_called()
    env.tokenizer_check.assert_not_called()
    env.video_check.assert_not_called()


def test_runtime_rejects_missing_assign_api_before_cuda(tmp_path, fake_cuda_runtime):
    env = fake_cuda_runtime
    env.torch.nn.Module.load_state_dict = lambda self, state_dict, strict=True: None
    with pytest.raises(RuntimeError, match="PyTorch model loader API.*assign"):
        runtime.check_runtime(tmp_path, tmp_path, 24, 5)
    env.cuda.is_available.assert_not_called()


def test_runtime_rejects_cuda_without_bf16(tmp_path, fake_cuda_runtime):
    env = fake_cuda_runtime
    env.cuda.is_bf16_supported.return_value = False
    with pytest.raises(RuntimeError, match="CUDA availability.*BF16 dtype"):
        runtime.check_runtime(tmp_path, tmp_path, 24, 5)
    env.torch.ones.assert_not_called()


@pytest.mark.parametrize(
    "available, expected",
    [
        ((True, True, True), "flash3"),
        ((False, True, True), "flash2"),
        ((False, False, True), "sage"),
        ((False, False, False), "sdpa"),
    ],
)
def test_runtime_kernel_checks_and_backend_priority_with_cuda_double(
    tmp_path, fake_cuda_runtime, available, expected
):
    env = fake_cuda_runtime
    (
        env.attention.FLASH_ATTN_3_AVAILABLE,
        env.attention.FLASH_ATTN_2_AVAILABLE,
        env.attention.SAGE_ATTN_AVAILABLE,
    ) = available
    report = runtime.check_runtime(tmp_path, tmp_path, 24, 5)
    assert report["attention_backend"] == expected
    assert report["device"] == "cuda:1"
    assert report["free_memory_bytes"] == 123456  # No invented VRAM threshold.
    assert report["total_memory_bytes"] == 789012
    assert report["bf16_matmul"] == report["attention_kernel"] == "passed"
    assert env.cuda.synchronize.call_count == 2
    assert all(call.args == (1,) for call in env.cuda.synchronize.call_args_list)
    call = env.attention.flash_attention.call_args
    assert tuple(call.args[0].shape) == (1, 16, 256)
    assert call.kwargs == {"num_heads": 2}
    env.tokenizer_check.assert_called_once_with(tmp_path)
    env.video_check.assert_called_once_with(tmp_path, 24, 5)
    json.dumps(report)


def test_attention_kernel_error_keeps_cause_and_stops_early(tmp_path, fake_cuda_runtime):
    env = fake_cuda_runtime
    original = RuntimeError("no kernel image is available for execution on the device")
    env.attention.flash_attention.side_effect = original
    with pytest.raises(RuntimeError, match="CUDA Wan attention kernel.*no kernel image") as error:
        runtime.check_runtime(tmp_path, tmp_path, 24, 5)
    assert error.value.__cause__ is original
    env.tokenizer_check.assert_not_called()


def test_asynchronous_cuda_failure_is_reported_in_its_stage(tmp_path, fake_cuda_runtime):
    env = fake_cuda_runtime
    original = RuntimeError("CUDA kernel failure at synchronization")
    env.cuda.synchronize.side_effect = original
    with pytest.raises(RuntimeError, match="BF16 matrix multiplication.*synchronization") as error:
        runtime.check_runtime(tmp_path, tmp_path, 24, 5)
    assert error.value.__cause__ is original
    env.attention.flash_attention.assert_not_called()
