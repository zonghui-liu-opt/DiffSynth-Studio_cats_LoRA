# 猫视频 LoRA SFT 内网训练命令

## 1. 准备路径

```bash
export MODEL_ROOT=/path/to/local/wan
export TOKENIZER_PATH=$MODEL_ROOT/google/umt5-xxl
export DATA_ROOT=/path/to/dataset
```

`MODEL_ROOT` 必须包含：

- `diffusion_pytorch_model*.safetensors`
- `models_t5_umt5-xxl-enc-bf16.pth`
- `Wan2.2_VAE.pth`
- `google/umt5-xxl/`

## 2. 检查数据

```bash
python3 check_dataset.py \
  --dataset_root "$DATA_ROOT" \
  --metadata_path "$DATA_ROOT/metadata.csv" \
  --num_frames 121
```

确认：

- `bad_samples: 0`
- 生成了 `$DATA_ROOT/metadata_fixed.csv`
- 输出包含 `resolution_counts` 和 `bucket_counts`
- `$DATA_ROOT/metadata_fixed.csv` 包含 `height,width,bucket`

`bucket` 使用 `HxW` 命名，例如 `480x832`、`832x480`、`480x480`。后续训练不再指定全局 `HEIGHT/WIDTH`，直接按 metadata 每行的 `height,width` resize。

## 3. 冒烟训练

先取 16 条数据：

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("/path/to/dataset")
df = pd.read_csv(root / "metadata_fixed.csv")
if "bucket" in df:
    smoke = pd.concat([group.head(8) for _, group in df.groupby("bucket", sort=False)])
else:
    smoke = df.head(16)
smoke.head(16).to_csv(root / "metadata_smoke16.csv", index=False)
PY
```

跑 2 卡小测试：

```bash
MODEL_ROOT=$MODEL_ROOT \
TOKENIZER_PATH=$TOKENIZER_PATH \
DATA_ROOT=$DATA_ROOT \
METADATA_PATH=$DATA_ROOT/metadata_smoke16.csv \
OUTPUT_ROOT=./models/train/cats_Wan2.2-TI2V-5B_lora_smoke \
NUM_GPUS=2 \
NUM_FRAMES=121 \
ENABLE_RESOLUTION_BUCKETS=1 \
SAVE_STEPS= \
NUM_EPOCHS=1 DATASET_REPEAT=3 DATASET_NUM_WORKERS=4 \
bash train_ti2v5b_lora.sh
```

检查：

```bash
ls ./models/train/cats_Wan2.2-TI2V-5B_lora_smoke
tail -n 5 ./models/train/cats_Wan2.2-TI2V-5B_lora_smoke/metrics.jsonl
```

应看到：

- `epoch-0.safetensors`
- `metrics.jsonl`
- `training_args.json`

## 4. 正式训练

```bash
MODEL_ROOT=$MODEL_ROOT \
TOKENIZER_PATH=$TOKENIZER_PATH \
DATA_ROOT=$DATA_ROOT \
METADATA_PATH=$DATA_ROOT/metadata_fixed.csv \
OUTPUT_ROOT=./models/train/cats_Wan2.2-TI2V-5B_lora \
NUM_GPUS=4 \
NUM_FRAMES=121 \
ENABLE_RESOLUTION_BUCKETS=1 \
SAVE_STEPS= \
NUM_EPOCHS=5 DATASET_REPEAT=1 DATASET_NUM_WORKERS=8 \
bash train_ti2v5b_lora.sh
```

只改变量，不改代码。需要临时退回上游 center-crop 行为时，追加 `ENABLE_RESOLUTION_BUCKETS=0`。旧变量 `ENABLE_ORIENTATION_BUCKETS` 仍作为兼容别名可用。

默认 `SAVE_STEPS=` 为空，训练每个 epoch 保存一次 `epoch-*.safetensors`。如果要按 step 保存，例如每 500 step 存一次：

```bash
SAVE_STEPS=500 bash train_ti2v5b_lora.sh
```

设置后会保存 `step-500.safetensors`、`step-1000.safetensors` 等；训练结束时如果最后 step 不是 500 的整数倍，还会额外保存最终 step ckpt。设置 `SAVE_STEPS` 后不再保存 `epoch-*.safetensors`。

## 5. 看训练曲线

```bash
python3 plot_metrics.py \
  --metrics_path ./models/train/cats_Wan2.2-TI2V-5B_lora/metrics.jsonl \
  --output_dir ./models/train/cats_Wan2.2-TI2V-5B_lora/plots
```

查看：

- `plots/loss.png`
- `plots/throughput.png`
- 终端输出的 tokens/s 和 videos/hour

## 6. 常见回退

- OOM：先降 `NUM_FRAMES` 或分辨率确认链路，再恢复正式参数。
- GPU 利用率周期性掉底：增大 `DATASET_NUM_WORKERS`。
- DDP 报 unused parameters：在 `train_ti2v5b_lora.sh` 的训练参数中临时追加 `--find_unused_parameters`。
- `metadata.csv` 读取异常：训练一律改用 `metadata_fixed.csv`。
- 多分辨率混合：确认 `check_dataset.py` 输出 `bucket_counts`，训练脚本默认 `ENABLE_RESOLUTION_BUCKETS=1`，新增分辨率只需要在 metadata 中写入对应 `height,width`。
