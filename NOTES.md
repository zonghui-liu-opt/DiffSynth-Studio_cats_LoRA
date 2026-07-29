# Wan2.2-TI2V-5B LoRA 内网上机说明

## 权重目录

`MODEL_ROOT` 下必须有：

- `diffusion_pytorch_model*.safetensors`：Wan2.2-TI2V-5B DiT 权重分片。
- `models_t5_umt5-xxl-enc-bf16.pth`：umT5 文本编码器权重。
- `Wan2.2_VAE.pth`：Wan2.2 VAE 权重。
- `google/umt5-xxl/`：本地 tokenizer 目录。

训练必须使用 `--model_paths` 读本地文件。内网不要用 `--model_id_with_origin_paths`，否则会尝试下载。

## 数据检查

真实 `metadata.csv` 是 tab 分隔；DiffSynth 默认按逗号读 CSV，所以先生成 `metadata_fixed.csv`：

```bash
DATA_ROOT=/path/to/dataset

python3 check_dataset.py \
  --dataset_root "$DATA_ROOT" \
  --metadata_path "$DATA_ROOT/metadata.csv" \
  --num_frames 121 | tee "$DATA_ROOT/check_dataset.log"
```

看输出里的 `resolution_counts`、`bucket_counts`、帧数分布和 `tokens_per_bucket`。训练时使用 `metadata_fixed.csv`；该文件会额外包含或规范化 `height,width,bucket`，bucket 名固定为 `HxW`，例如 `480x832`、`832x480`、`480x480`。新增训练分辨率时，只需要让 metadata 每行写对 `height,width`；如果原始 metadata 没有这两列，`check_dataset.py` 会用视频实测尺寸补齐。

## 冒烟训练

先取 16 条数据跑小规模测试。若数据里有竖屏样本，优先从每个 bucket 各取一部分，避免冒烟只覆盖横屏：

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

```bash
MODEL_ROOT=/path/to/local/wan \
TOKENIZER_PATH=/path/to/local/wan/google/umt5-xxl \
DATA_ROOT=/path/to/dataset \
METADATA_PATH=/path/to/dataset/metadata_smoke16.csv \
OUTPUT_ROOT=./models/train/Wan2.2-TI2V-5B_lora_smoke \
NUM_GPUS=2 \
NUM_FRAMES=121 \
ENABLE_RESOLUTION_BUCKETS=1 \
SAVE_STEPS= \
NUM_EPOCHS=1 DATASET_REPEAT=3 DATASET_NUM_WORKERS=4 \
bash train_ti2v5b_lora.sh
```

预期：

- `OUTPUT_ROOT/metrics.jsonl` 持续写入。
- `OUTPUT_ROOT/training_args.json` 存在。
- `OUTPUT_ROOT/epoch-0.safetensors` 存在。

如果 OOM，先改小分辨率/帧数做冒烟；正式训练再恢复。若 DDP 报 unused parameters，再考虑给训练命令追加 `--find_unused_parameters`。

## 正式训练

```bash
MODEL_ROOT=/path/to/local/wan \
TOKENIZER_PATH=/path/to/local/wan/google/umt5-xxl \
DATA_ROOT=/path/to/dataset \
METADATA_PATH=/path/to/dataset/metadata_fixed.csv \
OUTPUT_ROOT=./models/train/Wan2.2-TI2V-5B_lora \
NUM_GPUS=4 \
NUM_FRAMES=121 \
ENABLE_RESOLUTION_BUCKETS=1 \
SAVE_STEPS= \
NUM_EPOCHS=5 DATASET_REPEAT=1 DATASET_NUM_WORKERS=8 \
bash train_ti2v5b_lora.sh
```

开启 `ENABLE_RESOLUTION_BUCKETS=1` 后，训练会直接读取 metadata 每行的 `height,width`，按 `bucket=HxW` 分组采样，并把 `video` 与 `input_image` resize 到该行目标尺寸，不再需要在启动脚本里指定全局 `HEIGHT/WIDTH`。旧变量 `ENABLE_ORIENTATION_BUCKETS` 仍作为兼容别名可用；需要退回上游 center-crop 行为时设置 `ENABLE_RESOLUTION_BUCKETS=0`。`train_ti2v5b_lora.sh` 会导出 `PYTHONPATH=$PWD`，请从仓库根目录执行。

## Checkpoint 保存频率

默认 `SAVE_STEPS=` 为空，此时每个 epoch 保存一次：

- `epoch-0.safetensors`
- `epoch-1.safetensors`

如果要按 step 保存，在启动命令里设置：

```bash
SAVE_STEPS=500 bash train_ti2v5b_lora.sh
```

开启后会保存 `step-500.safetensors`、`step-1000.safetensors` 等；训练结束时如果最后一步不是 `SAVE_STEPS` 的整数倍，还会保存最终的 `step-{最后步数}.safetensors`。设置 `SAVE_STEPS` 后不再保存 `epoch-*.safetensors`。

## 指标绘图

```bash
python3 plot_metrics.py \
  --metrics_path ./models/train/Wan2.2-TI2V-5B_lora/metrics.jsonl \
  --output_dir ./models/train/Wan2.2-TI2V-5B_lora/plots \
  --warmup_steps 3
```

输出：

- `loss.png`：loss 原始曲线和 EMA 曲线。
- `throughput.png`：tokens/s 和 videos/hour。
- 终端摘要：总 step、末 20% loss、稳态 tokens/s、videos/hour。

## 注意事项

- 当前训练脚本显式传 `--data_file_keys "video,input_image"`。
- trainer 会优先使用 metadata 中的 `input_image`；没有该列时回退为视频首帧。
- 当前 DataLoader 每个 micro step 实际只取 1 条样本；bucket sampler 控制采样顺序和尺寸路径，不代表 dense tensor batch size > 1。
- 指标里的 `samples_per_step = grad_accum * world_size`；`tokens_per_sample` 会按当前样本实际 resize 后的 `height,width,num_frames` 计算，混合 `480x832/832x480/480x480` 时不会再固定用全局尺寸。
- 调 `DATASET_NUM_WORKERS` 时看 GPU 利用率：周期性掉底通常是解码瓶颈。
- 600 条数据规模较小，可先跑完冒烟再决定是否启用缓存数据流程。

## 验证出片

训练至少产出一个 `epoch-*.safetensors` 后，用官方验证方式加载 LoRA：

```python
import glob
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

MODEL_ROOT = "/path/to/local/wan"
DIT_PATHS = sorted(glob.glob(f"{MODEL_ROOT}/diffusion_pytorch_model*.safetensors"))
if not DIT_PATHS:
    raise FileNotFoundError(f"No DiT safetensors found at {MODEL_ROOT}/diffusion_pytorch_model*.safetensors")

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=DIT_PATHS),
        ModelConfig(path=f"{MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path=f"{MODEL_ROOT}/Wan2.2_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(path=f"{MODEL_ROOT}/google/umt5-xxl"),
)

pipe.load_lora(pipe.dit, "models/train/Wan2.2-TI2V-5B_lora/epoch-0.safetensors", alpha=1)
height, width = 480, 480
input_image = Image.open("/path/to/dataset/images_480x480/example.jpg").convert("RGB").resize((width, height))

video = pipe(
    prompt="一只猫在镜头前自然活动",
    input_image=input_image,
    height=height,
    width=width,
    num_frames=121,
    seed=1,
    tiled=True,
)
save_video(video, "validate_wan22_ti2v5b_lora.mp4", fps=15, quality=5)
```

批量推理建议直接用 `infer_ti2v5b_lora_batch.sh`，它会从 metadata 每行读取 `height,width`；单样本推理可设置 `METADATA_PATH=/path/to/metadata_fixed.csv ROW_ID=0 DATA_ROOT=/path/to/dataset python3 infer_cats_ti2v5b_lora.py`。

## LoRA 离线 Merge 与等价性对比

离线 merge 必须和运行时加载 LoRA 使用相同的 `LORA_ALPHA`（当前默认都是 `1`），并在将要做对比的 H100 软件/硬件环境中执行。merge 脚本会只加载 DiT，以 bf16 调用与运行时推理相同的 `pipe.load_lora` 融合路径，然后保存为可被 DiffSynth 自动识别的 safetensors 分片。

```bash
MODEL_ROOT=/path/to/local/Wan2.2-TI2V-5B \
LORA_PATH=/path/to/epoch-26.safetensors \
MERGED_MODEL_ROOT=/path/to/models/Wan2.2-TI2V-5B-cats-lora-merged \
LORA_ALPHA=1 \
CUDA_VISIBLE_DEVICES=0 \
bash merge_ti2v5b_lora.sh
```

默认 `AUX_FILES_MODE=symlink`，因此输出目录中的 T5、VAE 和 tokenizer 指向 baseline，节省磁盘且可以直接推理；需要把 merged 目录整体搬走时设置 `AUX_FILES_MODE=copy`。脚本拒绝覆盖已有 `MERGED_MODEL_ROOT`。成功后查看：

- `merge_manifest.json`：baseline/LoRA 路径、LoRA SHA256、alpha、融合层数、软件版本与 merged state checksum。
- `diffusion_pytorch_model-*.safetensors`：融合后的 bf16 DiT。
- 保存验证会逐 tensor 检查内存与磁盘完全一致，并释放原模型后通过 DiffSynth 再加载一次 merged DiT；仅在明确接受跳过时设置 `VERIFY_SAVE=0` 或 `VERIFY_RELOAD=0`。

使用完全相同的 metadata、seed、帧数、FPS 和 deterministic 设置分别推理：

```bash
COMMON_DATA_ROOT=/path/to/testsets
COMMON_METADATA=/path/to/testsets/metadata_6cases_480x832.csv

MODEL_ROOT=/path/to/local/Wan2.2-TI2V-5B \
LORA_PATH=/path/to/epoch-26.safetensors \
LORA_ALPHA=1 \
DATA_ROOT="$COMMON_DATA_ROOT" METADATA_PATH="$COMMON_METADATA" \
OUTPUT_DIR=./results/compare/lora \
NUM_FRAMES=97 SEED=1 DETERMINISTIC=strict \
bash infer_ti2v5b_lora_batch.sh

MERGED_MODEL_ROOT=/path/to/models/Wan2.2-TI2V-5B-cats-lora-merged \
DATA_ROOT="$COMMON_DATA_ROOT" METADATA_PATH="$COMMON_METADATA" \
OUTPUT_DIR=./results/compare/merged \
NUM_FRAMES=97 SEED=1 DETERMINISTIC=strict \
bash infer_ti2v5b_merged_batch.sh
```

两条推理命令最终共用 `infer_cats_ti2v5b_lora_batch.py`，唯一模型差异是前者启动时执行 LoRA fusion、后者读取已融合 DiT。比较 MP4 文件字节与解码后每个像素：

```bash
python3 compare_ti2v5b_batch_outputs.py \
  --lora-dir ./results/compare/lora \
  --merged-dir ./results/compare/merged \
  --comparison-dir ./results/compare/comparison_videos \
  --lora-flag "RUNTIME LORA" \
  --merged-flag "MERGED MODEL" \
  --report ./results/compare/report.json
```

脚本默认生成左右并排的标注视频：左侧标注 `RUNTIME LORA`，表示推理时加载 LoRA；右侧标注 `MERGED MODEL`，表示直接加载 merged 权重。视频画面中不显示逐帧差异，差异统计只写入终端与 JSON report。未指定 `--comparison-dir` 时输出到 merged 目录同级的 `comparison_videos/`；只需要 JSON 指标时可传 `--skip-comparison-videos`。

理想结果是 `all_file_bytes_equal=true`；即使 MP4 容器字节因为编码器元数据不同，也应至少满足 `all_decoded_frames_equal=true`。如果 strict deterministic 模式遇到当前 CUDA/attention 后端不支持的算子，脚本会直接失败而不是静默产出不可严格对比的结果；可用 `DETERMINISTIC=warn` 诊断，但该模式不再承诺逐像素一致。

## 内网 batch 推理入口

`infer_batch.py` 是两种权重加载方式共用的推理核心；两个 shell 只负责选择模型根目录、模式与输出目录，从而保证 metadata、图像 resize、seed、采样和视频保存代码完全一致。

推理时加载 LoRA：

```bash
BASE_MODEL_ROOT=/path/to/Wan2.2-TI2V-5B \
LORA_PATH=/path/to/epoch-52.safetensors \
DATA_ROOT=/path/to/testsets \
METADATA_PATH=/path/to/metadata_6cases_480x832.csv \
LORA_ALPHA=1.0 \
bash infer_batch_lora.sh
```

直接加载 merged 模型：

```bash
MERGED_MODEL_ROOT=/path/to/Wan2.2-TI2V-5B-merged \
DATA_ROOT=/path/to/testsets \
METADATA_PATH=/path/to/metadata_6cases_480x832.csv \
EXPECTED_LORA_ALPHA=1.0 \
bash infer_batch_merged.sh
```

merged 目录必须包含 merged DiT，以及 T5、VAE、tokenizer 的文件或有效软链接。若存在 `merge_manifest.json`，merged 入口会检查其中的 `lora_alpha` 是否与 `EXPECTED_LORA_ALPHA` 一致。两种入口都会在各自输出目录写入 `inference_manifest.json`，便于核对比较参数。
