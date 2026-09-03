"""Exercise the real TI2V pipeline and DiT dispatch without pretrained weights.

Only the VAE and prompt models are synthetic; the scheduler, pipeline units,
denoising loop, CFG branches, and Wan DiT implementation run on the CPU.
"""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import inference_timing
from diffsynth.diffusion import FlowMatchScheduler
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.pipelines.wan_video import (
    WanVideoPipeline,
    WanVideoUnit_CfgMerger,
    WanVideoUnit_ImageEmbedderFused,
    WanVideoUnit_InputVideoEmbedder,
    WanVideoUnit_NoiseInitializer,
    WanVideoUnit_PromptEmbedder,
    WanVideoUnit_ShapeChecker,
    model_fn_wan_video,
)
from inference_timing import WanInferenceTimer


class WorkClock:
    """Deterministic durations distinguish image encoding from video decoding."""

    def __init__(self):
        self.seconds = 0.0

    def __call__(self):
        return self.seconds

    def advance(self, seconds):
        self.seconds += seconds


class TinyVAE(torch.nn.Module):
    upsampling_factor = 16

    def __init__(self, clock):
        super().__init__()
        self.model = SimpleNamespace(z_dim=4)
        self.clock = clock
        self.encoded_latents = []
        self.decoded_latents = []
        self.fail_decode = False

    def encode(self, images, **kwargs):
        # ImageEmbedderFused supplies a list of C,T,H,W tensors.
        image = torch.stack(images).mean(dim=1, keepdim=True)
        latent = torch.nn.functional.interpolate(
            image,
            size=(1, image.shape[-2] // 16, image.shape[-1] // 16),
            mode="nearest",
        ).repeat(1, self.model.z_dim, 1, 1, 1)
        self.encoded_latents.append(latent.clone())
        self.clock.advance(11.0)
        return latent

    def decode(self, latents, **kwargs):
        self.decoded_latents.append(latents.clone())
        self.clock.advance(3.0)
        if self.fail_decode:
            raise RuntimeError("synthetic VAE decode failure")
        # Keep the batch dimension visible so CFG mistakes cannot be hidden by
        # the production PIL conversion's mean reduction across that dimension.
        return torch.nn.functional.interpolate(
            latents[:, :3],
            size=(
                (latents.shape[2] - 1) * 4 + 1,
                latents.shape[3] * self.upsampling_factor,
                latents.shape[4] * self.upsampling_factor,
            ),
            mode="nearest",
        )


class TinyTextEncoder(torch.nn.Module):
    def forward(self, ids, mask):
        return ids.float().unsqueeze(-1).repeat(1, 1, 4)


def tiny_tokenizer(prompt, **kwargs):
    ids = [1, 2] if prompt == "a cat" else [0, 1]
    return torch.tensor([ids]), torch.ones((1, 2), dtype=torch.long)


@pytest.fixture
def tiny_pipe(monkeypatch):
    clock = WorkClock()
    monkeypatch.setattr(inference_timing, "perf_counter", clock)
    pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.float32)
    pipe.vae = TinyVAE(clock)
    pipe.tokenizer = tiny_tokenizer
    pipe.text_encoder = TinyTextEncoder()
    # Locally initialize a tiny one-block DiT, including Wan 2.2 TI2V's
    # separated timestep and first-frame fusion. No model downloads are used.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        pipe.dit = WanModel(
            dim=16,
            in_dim=4,
            ffn_dim=32,
            out_dim=4,
            text_dim=4,
            freq_dim=8,
            eps=1e-6,
            patch_size=(1, 2, 2),
            num_heads=2,
            num_layers=1,
            has_image_input=False,
            seperated_timestep=True,
            require_vae_embedding=False,
            require_clip_embedding=False,
            fuse_vae_embedding_in_latents=True,
        )
    pipe.units = [
        WanVideoUnit_ShapeChecker(),
        WanVideoUnit_NoiseInitializer(),
        WanVideoUnit_PromptEmbedder(),
        WanVideoUnit_InputVideoEmbedder(),
        WanVideoUnit_ImageEmbedderFused(),
        WanVideoUnit_CfgMerger(),
    ]
    pipe.post_units = []
    pipe.eval()
    pipe.observed_dit_batches = []

    def record_dit_call(module, args):
        pipe.observed_dit_batches.append(args[0].shape[0])
        clock.advance(2.0)

    pipe.dit.patch_embedding.register_forward_pre_hook(record_dit_call)
    assert isinstance(pipe.scheduler, FlowMatchScheduler)
    assert pipe.model_fn is model_fn_wan_video
    return pipe


def infer(pipe, cfg_scale=5.0, cfg_merge=False):
    return pipe(
        prompt="a cat",
        negative_prompt="blurred",
        input_image=Image.new("RGB", (32, 32), (200, 100, 50)),
        height=32,
        width=32,
        num_frames=9,
        num_inference_steps=3,
        cfg_scale=cfg_scale,
        cfg_merge=cfg_merge,
        seed=42,
        rand_device="cpu",
        tiled=False,
        output_type="floatpoint",
        progress_bar_cmd=lambda timesteps: timesteps,
    )


@pytest.mark.parametrize(
    "cfg_scale,cfg_merge,dit_calls,dit_batch",
    [(1.0, False, 3, 1), (5.0, False, 6, 1), (5.0, True, 3, 2)],
)
def test_real_ti2v_cfg_timing_and_seed_invariance(
    tiny_pipe, cfg_scale, cfg_merge, dit_calls, dit_batch
):
    pipe = tiny_pipe
    before = infer(pipe, cfg_scale, cfg_merge)
    with WanInferenceTimer(pipe) as timer:
        # Encoding remains a class method, outside the timed decode wrapper.
        assert "encode" not in vars(pipe.vae)
        timed = infer(pipe, cfg_scale, cfg_merge)

    assert pipe.model_fn is model_fn_wan_video
    assert "decode" not in vars(pipe.vae)
    assert pipe.vae.decode.__func__ is TinyVAE.decode
    after = infer(pipe, cfg_scale, cfg_merge)
    assert before.shape == timed.shape == after.shape == (1, 3, 9, 32, 32)
    assert torch.isfinite(timed).all()
    assert torch.equal(before, timed)
    assert torch.equal(timed, after)

    # Merged CFG runs a batch of two through DiT but decodes only one video.
    assert pipe.observed_dit_batches == [dit_batch] * (3 * dit_calls)
    assert len(pipe.vae.encoded_latents) == len(pipe.vae.decoded_latents) == 3
    for encoded, decoded in zip(pipe.vae.encoded_latents, pipe.vae.decoded_latents):
        assert decoded.shape == (1, 4, 3, 2, 2)
        assert torch.equal(decoded[:, :, :1], encoded)

    assert timer.metrics() == {
        "dit_seconds": 2.0 * dit_calls,
        "dit_calls": dit_calls,
        "vae_decode_seconds": 3.0,
        "vae_decode_calls": 1,
        "pipeline_seconds": 11.0 + 2.0 * dit_calls + 3.0,
        "dit_mean_call_seconds": 2.0,
        "other_pipeline_seconds": 11.0,
    }


def test_real_pipeline_decode_failure_restores_module_methods(tiny_pipe):
    pipe = tiny_pipe
    pipe.vae.fail_decode = True
    timer = WanInferenceTimer(pipe)
    with pytest.raises(RuntimeError, match="synthetic VAE decode failure"):
        with timer:
            infer(pipe)

    assert pipe.model_fn is model_fn_wan_video
    assert "decode" not in vars(pipe.vae)
    assert pipe.vae.decode.__func__ is TinyVAE.decode
    assert timer.metrics()["dit_calls"] == 6
    assert timer.metrics()["vae_decode_calls"] == 1
    assert timer.metrics()["vae_decode_seconds"] == 3.0
    pipe.vae.fail_decode = False
    assert infer(pipe).shape == (1, 3, 9, 32, 32)
