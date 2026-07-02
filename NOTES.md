# Wan2.2-TI2V-5B LoRA Stage B Checklist

## Required Local Files

Place these under `MODEL_ROOT` before running on the offline H100 host:

- `diffusion_pytorch_model*.safetensors` - all Wan2.2-TI2V-5B DiT shard files.
- `models_t5_umt5-xxl-enc-bf16.pth` - umT5 text encoder weights.
- `Wan2.2_VAE.pth` - Wan2.2 VAE weights.
- `google/umt5-xxl/` - tokenizer directory for `--tokenizer_path`; without this, `train.py` defaults to a ModelScope/HF tokenizer path and will try to download.

The training launcher uses `--model_paths` with local paths. Do not use `--model_id_with_origin_paths` on the offline host.

## Data Check

Run this first on the real dataset. The real `metadata.csv` is tab-separated, while `UnifiedDataset` reads CSV through `pandas.read_csv()` without `sep`, so training should use the generated comma-separated `metadata_fixed.csv`.

```bash
DATA_ROOT=/path/to/dataset
python3 check_dataset.py \
  --dataset_root "$DATA_ROOT" \
  --metadata_path "$DATA_ROOT/metadata.csv" \
  --height 480 \
  --width 832 \
  --num_frames 121 | tee "$DATA_ROOT/check_dataset.log"
```

Edit only the top block of `train_ti2v5b_lora.sh` after checking the reported `recommended_height`, `recommended_width`, frame counts, and `tokens_per_video`.

## Smoke Run

Create a small metadata file. The repo has no `--max_steps` flag, so use `DATASET_REPEAT=3` to get roughly 20 optimizer steps from 16 rows on 2 GPUs.

```bash
DATA_ROOT=/path/to/dataset
python3 - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("/path/to/dataset")
df = pd.read_csv(root / "metadata_fixed.csv")
df.head(16).to_csv(root / "metadata_smoke16.csv", index=False)
PY

MODEL_ROOT=/path/to/local/wan \
TOKENIZER_PATH=/path/to/local/wan/google/umt5-xxl \
DATA_ROOT=/path/to/dataset \
METADATA_PATH=/path/to/dataset/metadata_smoke16.csv \
OUTPUT_ROOT=./models/train/Wan2.2-TI2V-5B_lora_smoke \
NUM_GPUS=2 \
HEIGHT=480 WIDTH=832 NUM_FRAMES=121 \
NUM_EPOCHS=1 DATASET_REPEAT=3 DATASET_NUM_WORKERS=4 \
bash train_ti2v5b_lora.sh
```

Run commands from the repository root. The launcher exports `PYTHONPATH=$PWD` so `examples/wanvideo/model_training/train.py` can import the local `diffsynth` package without requiring an editable install.

Expected observations:

- `metrics.jsonl` appears under `OUTPUT_ROOT` and grows one line per optimizer step on rank 0.
- `training_args.json` appears under `OUTPUT_ROOT`.
- `epoch-0.safetensors` appears at epoch end.
- If DDP reports unused parameters, rerun after adding `--find_unused_parameters` to the launcher command. The official TI2V LoRA example does not use it, so keep it off unless the failure appears.
- If OOM occurs, lower resolution/frames for the smoke, set `NUM_GPUS=4`, or enable gradient checkpointing offload by adding `--use_gradient_checkpointing_offload`.

## Full Run

After smoke passes:

```bash
MODEL_ROOT=/path/to/local/wan \
TOKENIZER_PATH=/path/to/local/wan/google/umt5-xxl \
DATA_ROOT=/path/to/dataset \
METADATA_PATH=/path/to/dataset/metadata_fixed.csv \
OUTPUT_ROOT=./models/train/Wan2.2-TI2V-5B_lora \
NUM_GPUS=4 \
HEIGHT=480 WIDTH=832 NUM_FRAMES=121 \
NUM_EPOCHS=5 DATASET_REPEAT=1 DATASET_NUM_WORKERS=8 \
bash train_ti2v5b_lora.sh
```

The current trainer now uses `data["input_image"][0]` when `--data_file_keys "video,input_image"` loads that column; otherwise it falls back to the official behavior of using `data["video"][0]`.

## Plot Metrics

```bash
python3 plot_metrics.py \
  --metrics_path ./models/train/Wan2.2-TI2V-5B_lora/metrics.jsonl \
  --output_dir ./models/train/Wan2.2-TI2V-5B_lora/plots \
  --warmup_steps 3
```

Expected observations:

- `loss.png` shows raw loss plus EMA.
- `throughput.png` shows global tokens/s and videos/hour.
- The printed summary reports total steps, last 20% average loss, steady tokens/s, steady videos/hour, and observed wall-clock hours.

## H100 Tuning Sequence

Record `tokens/s`, `videos/hour`, and `torch.cuda.max_memory_allocated()` after each change. Keep only changes with clear throughput or memory benefit.

1. Confirm flash attention or the repository's available attention backend is active. If the import/backend fails, fall back to the default PyTorch attention and compare tokens/s.
2. Start with gradient checkpointing on. `WanTrainingModule` forcibly enables it if disabled. Try disabling only if you intentionally patch that behavior and have memory headroom; otherwise use `--use_gradient_checkpointing_offload` only for OOM recovery.
3. Tune `DATASET_NUM_WORKERS`: if GPU utilization periodically drops to near zero, increase workers; if CPU RAM or dataloader startup becomes unstable, reduce workers.
4. Check whether cached data processing tasks are useful for your run. The training code supports `metadata_path=None` to load cached `.pth` files, and tasks ending with `:data_process` can write preprocessed `.pth` files. For 600 samples, precomputing is likely cheap; validate outputs before switching full training to cached data.
5. If memory is still low after stable 4-GPU training, try increasing micro batch size only if you add real batching support. Current `DataLoader` uses `collate_fn=lambda x: x[0]`, so `samples_per_step` assumes micro batch size 1.

## Validation Video

Use the official validation pattern after at least one checkpoint exists:

```python
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

MODEL_ROOT = "/path/to/local/wan"
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=f"{MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path=f"{MODEL_ROOT}/diffusion_pytorch_model-00001-of-00001.safetensors"),
        ModelConfig(path=f"{MODEL_ROOT}/Wan2.2_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(path=f"{MODEL_ROOT}/google/umt5-xxl"),
)
pipe.load_lora(pipe.dit, "models/train/Wan2.2-TI2V-5B_lora/epoch-0.safetensors", alpha=1)
input_image = Image.open("/path/to/dataset/images_480x832/example.jpg").convert("RGB").resize((832, 480))
video = pipe(prompt="your validation prompt", input_image=input_image, height=480, width=832, num_frames=121, seed=1, tiled=True)
save_video(video, "validate_wan22_ti2v5b_lora.mp4", fps=15, quality=5)
```

If the DiT checkpoint has multiple shards, replace the single DiT `ModelConfig(path=...)` with the exact local shard list.
