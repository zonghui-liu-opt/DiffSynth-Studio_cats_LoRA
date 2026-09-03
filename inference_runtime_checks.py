"""Offline runtime smoke checks to run before allocating Wan model weights.

These checks execute small kernels and a real MP4 encode/decode, but do not
estimate whether the full requested video will fit in GPU memory.
"""

from contextlib import contextmanager
from importlib import import_module
import inspect
from pathlib import Path
import platform
from tempfile import TemporaryDirectory


class RuntimePreflightError(RuntimeError):
    """A runtime check failed; the original exception remains in __cause__."""


@contextmanager
def _stage(name):
    try:
        yield
    except RuntimePreflightError:
        raise
    except Exception as exc:
        raise RuntimePreflightError(f"Runtime preflight [{name}] failed: {exc}") from exc


def check_tokenizer(tokenizer_path):
    """Load the pipeline's local tokenizer and tokenize English and Chinese."""
    with _stage("tokenizer local load and tokenization"):
        tokenizer_path = Path(tokenizer_path).expanduser().resolve()
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"Local tokenizer directory not found: {tokenizer_path}")
        tokenizer_class = import_module(
            "diffsynth.models.wan_video_text_encoder"
        ).HuggingfaceTokenizer
        tokenizer = tokenizer_class(
            name=str(tokenizer_path),
            seq_len=512,
            clean="whitespace",
            local_files_only=True,
        )
        ids, mask = tokenizer(
            ["A cat looks at the camera.", "一只猫看着镜头。"], return_mask=True
        )
        if tuple(ids.shape) != (2, 512) or tuple(mask.shape) != (2, 512):
            raise ValueError(
                f"Unexpected tokenizer shapes: ids={tuple(ids.shape)}, "
                f"mask={tuple(mask.shape)}; expected (2, 512)"
            )
        token_counts = [int(value) for value in mask.sum(dim=1).tolist()]
        if any(count <= 0 for count in token_counts):
            raise ValueError(f"Tokenizer returned an empty sequence: {token_counts}")
        return {
            "status": "passed",
            "path": str(tokenizer_path),
            "local_files_only": True,
            "vocab_size": int(tokenizer.vocab_size),
            "token_counts": token_counts,
            "shape": list(ids.shape),
        }


def check_video_writer(output_dir, fps, quality):
    """Write and read a tiny MP4 in the actual output filesystem, then remove it."""
    with _stage("video writer dependencies and output directory"):
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        image_class = import_module("PIL.Image")
        imageio = import_module("imageio")
        save_video = import_module("diffsynth.utils.data").save_video
        frames = [
            image_class.new("RGB", (32, 32), (255, 0, 0)),
            image_class.new("RGB", (32, 32), (0, 0, 255)),
        ]

    # TemporaryDirectory also checks write access to OUTPUT_DIR. Keep the
    # encoder and decoder on the same filesystem used by the full run.
    with _stage("video output write access and temporary file cleanup"):
        with TemporaryDirectory(prefix=".wan_runtime_check_", dir=output_dir) as temp_dir:
            video_path = Path(temp_dir) / "smoke.mp4"
            with _stage("MP4 encoding with diffsynth.save_video"):
                save_video(frames, str(video_path), fps=fps, quality=quality)
                if not video_path.is_file() or video_path.stat().st_size == 0:
                    raise ValueError(f"Video encoder did not create a non-empty MP4: {video_path}")
                encoded_bytes = video_path.stat().st_size

            with _stage("MP4 decoding with imageio/ffmpeg"):
                reader = imageio.get_reader(str(video_path), format="ffmpeg")
                try:
                    for index in range(2):
                        frame = reader.get_data(index)
                        if tuple(frame.shape[:2]) != (32, 32) or frame.size == 0:
                            raise ValueError(
                                f"Decoded frame {index} has invalid shape {frame.shape}; "
                                "expected a non-empty 32x32 image"
                            )
                finally:
                    reader.close()

    return {
        "status": "passed",
        "output_dir": str(output_dir),
        "fps": fps,
        "quality": quality,
        "decoded_frames": 2,
        "frame_size": [32, 32],
        "encoded_bytes": int(encoded_bytes),
    }


def check_runtime(tokenizer_path, output_dir, fps, quality):
    """Return a JSON-safe report after checking the full local inference runtime.

    Set offline environment variables before calling this function. It never
    loads model weights, installs dependencies, or downloads a tokenizer.
    Failure includes the check stage and preserves the original exception chain.
    CUDA success means tiny BF16 kernels work; it is not a full-model GPU test.
    """
    with _stage("PyTorch model loader API"):
        torch = import_module("torch")
        if "assign" not in inspect.signature(torch.nn.Module.load_state_dict).parameters:
            raise RuntimeError(
                f"Installed torch {torch.__version__} lacks "
                "torch.nn.Module.load_state_dict(assign=...). "
                "This repository's model loader requires that API; use an offline "
                "PyTorch build compatible with the repository and your CUDA driver."
            )

    with _stage("CUDA availability and BF16 support"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is unavailable (torch={torch.__version__}, "
                f"torch CUDA build={torch.version.cuda}). "
                "Check the CUDA PyTorch build, NVIDIA driver, and CUDA_VISIBLE_DEVICES."
            )
        if not hasattr(torch.cuda, "is_bf16_supported") or not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support the required BF16 dtype.")
        device_index = torch.cuda.current_device()
        device = f"cuda:{device_index}"
        properties = torch.cuda.get_device_properties(device_index)
        free_memory, total_memory = torch.cuda.mem_get_info(device_index)
        report = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": device,
            "device_name": str(properties.name),
            "device_capability": [int(properties.major), int(properties.minor)],
            "free_memory_bytes": int(free_memory),
            "total_memory_bytes": int(total_memory),
            "bf16_supported": True,
        }

    with _stage("CUDA BF16 matrix multiplication"):
        with torch.no_grad():
            matrix = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
            product = matrix @ matrix
            torch.cuda.synchronize(device_index)
            if not bool(torch.isfinite(product).all().item()):
                raise ValueError("BF16 matrix multiplication produced non-finite values.")
        del matrix, product
        report["bf16_matmul"] = "passed"

    with _stage("Wan attention backend import"):
        wan_dit = import_module("diffsynth.models.wan_video_dit")
        # Match the exact dispatch order in wan_video_dit.flash_attention.
        report["attention_backend"] = next(
            (
                label
                for attribute, label in (
                    ("FLASH_ATTN_3_AVAILABLE", "flash3"),
                    ("FLASH_ATTN_2_AVAILABLE", "flash2"),
                    ("SAGE_ATTN_AVAILABLE", "sage"),
                )
                if getattr(wan_dit, attribute, False)
            ),
            "sdpa",
        )

    with _stage(f"CUDA Wan attention kernel ({report['attention_backend']})"):
        with torch.no_grad():
            # Wan2.2 TI2V 5B uses 128 channels per head; this tests the same
            # head dimension with a tiny sequence and only two heads.
            q = torch.ones((1, 16, 256), device=device, dtype=torch.bfloat16)
            k, v = q.clone(), q.clone()
            attention = wan_dit.flash_attention(q, k, v, num_heads=2)
            torch.cuda.synchronize(device_index)
            if tuple(attention.shape) != tuple(q.shape):
                raise ValueError(f"Unexpected attention output shape: {tuple(attention.shape)}")
            if not bool(torch.isfinite(attention).all().item()):
                raise ValueError("Attention produced non-finite values.")
        del q, k, v, attention
        report["attention_kernel"] = "passed"

    report["tokenizer"] = check_tokenizer(tokenizer_path)
    report["video_writer"] = check_video_writer(output_dir, fps, quality)
    report["status"] = "passed"
    return report
