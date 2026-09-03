import sys
from types import SimpleNamespace

import pytest

import inference_timing
from inference_timing import WanInferenceTimer


class FakeVAE:
    def decode(self, value):
        return value + 1


class FakePipeline:
    def __init__(self, device="cpu"):
        self.device = device
        self.vae = FakeVAE()
        self.model_fn = lambda value: value * 2

    def __call__(self, value):
        positive = self.model_fn(value)
        negative = self.model_fn(0)
        return self.vae.decode(positive - negative)


def test_records_cfg_calls_and_restores_class_and_instance_attributes(monkeypatch):
    pipe = FakePipeline()
    original_model_fn = pipe.model_fn
    # Outer duration 12s, the two model calls 2s + 3s, decode 4s.
    timestamps = iter([0, 1, 3, 4, 7, 7, 11, 12])
    monkeypatch.setattr(inference_timing, "perf_counter", lambda: next(timestamps))

    timer = WanInferenceTimer(pipe)
    with timer:
        assert pipe(5) == 11

    assert timer.metrics() == {
        "dit_seconds": 5,
        "dit_calls": 2,
        "vae_decode_seconds": 4,
        "vae_decode_calls": 1,
        "pipeline_seconds": 12,
        "dit_mean_call_seconds": 2.5,
        "other_pipeline_seconds": 3,
    }
    assert pipe.model_fn is original_model_fn
    assert "decode" not in vars(pipe.vae)
    assert pipe.vae.decode.__func__ is FakeVAE.decode


def test_cuda_syncs_the_configured_device_at_all_timing_boundaries(monkeypatch):
    events = []
    fake_cuda = SimpleNamespace(synchronize=lambda device: events.append(("sync", device)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(inference_timing, "perf_counter", lambda: events.append(("clock",)) or 0)
    pipe = FakePipeline(device="cuda:2")
    pipe.model_fn = lambda value: events.append(("dit",)) or value
    pipe.vae.decode = lambda value: events.append(("decode",)) or value
    original_decode = pipe.vae.decode

    with WanInferenceTimer(pipe):
        pipe.model_fn(1)
        pipe.vae.decode(1)

    assert events == [
        ("sync", "cuda:2"), ("clock",),
        ("sync", "cuda:2"), ("clock",), ("dit",), ("sync", "cuda:2"), ("clock",),
        ("sync", "cuda:2"), ("clock",), ("decode",), ("sync", "cuda:2"), ("clock",),
        ("sync", "cuda:2"), ("clock",),
    ]
    assert pipe.vae.decode is original_decode


def test_exception_is_propagated_and_methods_are_restored():
    pipe = FakePipeline()

    def failing_model(value):
        raise RuntimeError("inference failed")

    pipe.model_fn = failing_model
    timer = WanInferenceTimer(pipe)
    with pytest.raises(RuntimeError, match="inference failed"):
        with timer:
            pipe(1)

    assert pipe.model_fn is failing_model
    assert "decode" not in vars(pipe.vae)
    assert timer.metrics()["dit_calls"] == 1
    assert timer.metrics()["vae_decode_calls"] == 0
    assert timer.metrics()["pipeline_seconds"] >= timer.metrics()["dit_seconds"]


def test_reusing_context_resets_previous_measurements():
    pipe = FakePipeline()
    timer = WanInferenceTimer(pipe)
    with timer:
        pipe(1)
    assert timer.metrics()["dit_calls"] == 2

    with timer:
        pipe.vae.decode(1)
    assert timer.metrics()["dit_calls"] == 0
    assert timer.metrics()["vae_decode_calls"] == 1
    assert timer.metrics()["dit_mean_call_seconds"] == 0


def test_disabled_has_no_wrappers_clock_or_sync(monkeypatch):
    pipe = FakePipeline(device="mps")
    original_model_fn = pipe.model_fn

    def unexpected_call(*args, **kwargs):
        pytest.fail("disabled timer must not read the clock or synchronize")

    monkeypatch.setattr(inference_timing, "perf_counter", unexpected_call)
    monkeypatch.setattr(WanInferenceTimer, "_synchronize", unexpected_call)
    timer = WanInferenceTimer(pipe, enabled=False)
    with timer:
        assert pipe.model_fn is original_model_fn
        assert "decode" not in vars(pipe.vae)
        assert pipe(2) == 5
    assert timer.metrics() == {}


def test_rejects_nested_contexts_without_breaking_outer_timer():
    pipe = FakePipeline()
    timer = WanInferenceTimer(pipe)
    with timer:
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with timer:
                pass
        with pytest.raises(RuntimeError, match="already wrapped"):
            with WanInferenceTimer(pipe):
                pass
        pipe(1)
    assert timer.metrics()["dit_calls"] == 2
    assert "decode" not in vars(pipe.vae)


def test_partial_setup_failure_restores_model_fn():
    pipe = FakePipeline()
    original_model_fn = pipe.model_fn
    pipe.vae.decode = None
    with pytest.raises(TypeError, match="decode must be callable"):
        with WanInferenceTimer(pipe):
            pass
    assert pipe.model_fn is original_model_fn
    assert pipe.vae.decode is None


def test_synchronization_failure_also_restores_wrappers(monkeypatch):
    pipe = FakePipeline(device="cuda:1")
    original_model_fn = pipe.model_fn
    calls = 0

    def synchronize(device):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("CUDA synchronization failed")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize)))
    timer = WanInferenceTimer(pipe)
    with pytest.raises(RuntimeError, match="CUDA synchronization failed"):
        with timer:
            pass
    assert pipe.model_fn is original_model_fn
    assert "decode" not in vars(pipe.vae)


def test_rejects_unsupported_device_without_patching():
    pipe = FakePipeline(device="mps")
    original_model_fn = pipe.model_fn
    with pytest.raises(ValueError, match="only CUDA and CPU"):
        with WanInferenceTimer(pipe):
            pass
    assert pipe.model_fn is original_model_fn
    assert "decode" not in vars(pipe.vae)
