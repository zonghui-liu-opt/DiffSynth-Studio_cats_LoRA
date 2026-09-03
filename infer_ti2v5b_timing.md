# Wan2.2-TI2V-5B rank64 LoRA 批量推理与分段计时

入口为 `infer_batch_lora.sh` → `infer_batch.sh` → `infer_batch.py`。一次加载本地模型和 LoRA，按 CSV/TSV 顺序逐条输入 `prompt + input_image`，每次生成一段视频（batch size = 1）。支持横竖屏混合任务列表；不把多条样本堆成一个 GPU tensor batch。

## 运行

沿用仓库原有内网模型目录、rank64 `epoch-52.safetensors` 和 Wan Python 环境时，更新整份仓库后，在仓库目录运行这一条即可：

```bash
bash infer_batch_lora.sh
```

脚本会自动预检，通过后直接开始正式批量推理，不要求先执行另一条测试命令。默认仍是 97 帧、50 步、CFG=5、分块 VAE、每条生成一次。默认 `WARMUP_RUNS=0`；需要去掉冷启动影响时用 `WARMUP_RUNS=1 bash infer_batch_lora.sh`。

模型目录和原内网 Python 路径沿用已有配置；`testsets`、`results` 默认按脚本所在仓库定位，移动仓库或从其他工作目录启动都可用。脚本优先使用原内网 Wan Python，原路径不存在时使用当前 `python3`，并打印最终解释器。目录不同可在 **`infer_batch_lora.sh` 顶部**修改，或一次性传入环境变量：

```bash
BASE_MODEL_ROOT=/path/to/Wan2.2-TI2V-5B \
LORA_PATH=/path/to/rank64/epoch-52.safetensors \
DATA_ROOT="$PWD/testsets" \
METADATA_PATH="$PWD/testsets/metadata_6cases_480x832.csv" \
OUTPUT_DIR="$PWD/results/ti2v5b_rank64_timing" \
PYTHON_BIN=/path/to/wan_environment/bin/python \
CUDA_VISIBLE_DEVICES=0 \
NUM_FRAMES=97 \
NUM_INFERENCE_STEPS=50 \
WARMUP_RUNS=1 \
REPEATS=1 \
bash infer_batch_lora.sh
```

直接执行 Python 时用 `MODEL_ROOT` 指定基础模型目录；LoRA launcher 支持 `BASE_MODEL_ROOT` 和 `MODEL_ROOT`，同时设置时以前者为准。不要为了运行此脚本切换到未安装 Wan 依赖的系统 Python。

基础模型目录需包含 `diffusion_pytorch_model*.safetensors`、`models_t5_umt5-xxl-enc-bf16.pth`、`Wan2.2_VAE.pth` 和 `google/umt5-xxl/`。LoRA launcher 默认检查所有识别出的 LoRA A/B（或 down/up）矩阵的 rank 必须为 64。其他 rank 可通过 `EXPECTED_LORA_RANK=16` 指定，或设为空字符串仅记录检测结果。

预检在加载大模型前执行：

- 检查当前 PyTorch 的模型加载 API、CUDA/BF16，实际运行小型 BF16 矩阵乘法及当前 Wan attention 后端。
- 强制离线，从本地加载 tokenizer 并执行中英文分词。脚本不安装包，也不下载模型或 tokenizer。
- 在实际输出目录临时编码两帧 MP4 并读回，确认 FFmpeg 和写权限；检查文件会自动清理。
- 按 safetensors index 精确选择 DiT 分片；没有 index 时检查分片编号完整性，拒绝重复 tensor、缺片和非 TI2V-5B 结构。
- 使用实际 LoRA 加载器的命名规则核对每一对 A/B、矩阵形状及 DiT 目标层，拒绝缺配对、错误模型或零匹配，避免静默运行基础模型。

所有 Python 标准输出和错误自动保存到输出目录的 `inference.log`，错误退出码会正常向上传递。预检通过报告写入 `preflight_report.json`，包含实际软件、CUDA、设备、attention 后端和编码检查结果。

metadata 示例：

```csv
input_image,prompt,height,width
images/cat_landscape.png,一只猫轻轻歪头然后恢复坐姿,480,832
images/cat_portrait.png,一只猫抬起前爪并自然眨眼,832,480
```

相对图片路径基于 `DATA_ROOT`；也支持绝对路径。`prompt`、`input_image` 必填；省略两列尺寸时使用 `HEIGHT/WIDTH`，提供尺寸时必须成对且为正的 32 倍数。可选列 `negative_prompt` 覆盖全局负向提示词。会在加载大模型前检查所有图片、提示词和尺寸。

## 参数

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `TIMING_ENABLED` | `1` | 记录分段耗时；`0` 关闭计时包装与报告 |
| `WARMUP_RUNS` | `0` | 每种分辨率首次出现时完整预热次数，建议测性能时设为 `1` |
| `REPEATS` | `1` | 每条样本正式测量次数；可设 `3` 观察波动 |
| `NUM_INFERENCE_STEPS` | `50` | 去噪步数 |
| `CFG_SCALE` / `CFG_MERGE` | `5.0` / `0` | 引导强度；合并 CFG 可设 `CFG_MERGE=1`，可能增加显存占用 |
| `SIGMA_SHIFT` | `5.0` | 调度器 shift |
| `NEGATIVE_PROMPT` | 空字符串 | 全局负向提示词 |
| `NUM_FRAMES` / `FPS` | `97` / `24` | 帧数必须满足 `4*n+1`；FPS 为视频保存帧率 |
| `SEED` | `1` | 每次调用使用相同 seed，包括预热和重复测量 |
| `LORA_ALPHA` | `1.0` | LoRA 权重缩放 |
| `TILED` | `1` | VAE 分块编码/解码；设 `0` 关闭，需足够显存 |
| `DETERMINISTIC` | `strict` | 保留原确定性设置；可选 `warn`、`off`，比较性能时保持一致 |

预热使用该分辨率首次出现的样本和完整推理参数，不保存视频，也不计入 CSV 或均值。若列表同时包含 `480x832` 和 `832x480`，`WARMUP_RUNS=1` 会额外运行两次完整推理。默认 `0` 保留冷启动样本，报告中会明确记录该设置。

关闭 guidance 时请用 `CFG_SCALE=1 CFG_MERGE=0`。当前 Wan pipeline 在 `CFG_SCALE=1 CFG_MERGE=1` 下仍合并双份 latent，此入口会提前拒绝该组合以避免多余计算和解码。

## 输出与计时定义

输出目录包含：

- `0000_图片名.mp4` 等逐条视频。`REPEATS>1` 时使用 `_r01`、`_r02` 等后缀。
- `timing.csv`：每条正式测量一行，记录样本编号、重复编号、尺寸、seed、步数、CFG 和耗时。CSV 编号从 0 开始，文件重复后缀从 1 开始。
- `timing_summary.json`：总体与 `by_resolution` 分组统计，包含数量及 total / mean / min / max / median / p95，时间单位均为秒。P95 使用线性插值。
- `inference_manifest.json`：模型和 LoRA 路径、检测 rank、加载方式、GPU、PyTorch/CUDA 版本、生成参数、预热及重复设置、模型加载耗时。
- `preflight_report.json` / `inference.log`：启动预检通过结果与完整 Python 运行日志；manifest 还会记录 LoRA 实际匹配层数。

| CSV 字段 | 计时范围 |
| --- | --- |
| `dit_seconds` | 全部 `pipe.model_fn` 调用耗时之和，包含 DiT embedding、transformer blocks、输出头，以及启用时的正/负 CFG 两次模型计算；不包含模型函数外的 CFG 合成和 scheduler 更新 |
| `dit_calls` | `model_fn` 调用次数。默认 50 步、CFG=5、CFG_MERGE=0 时为 100；CFG=1 或合并 CFG 时为 50 |
| `dit_mean_call_seconds` | `dit_seconds / dit_calls` |
| `dit_mean_step_seconds` | `dit_seconds / 实际去噪步数`，包含每步的全部 CFG 模型调用 |
| `vae_decode_seconds` | 整个 `pipe.vae.decode` 调用，包括内部数据搬运、分块解码和 CPU 拼接；不包含首帧 VAE encode 或调用前的显式模型切换 |
| `vae_decode_calls` | 通常为 1；内部多个 tile 不单独计数 |
| `pipeline_seconds` | 完整 `pipe(...)` 调用，包括文本编码、首帧 VAE encode、DiT、调度器、VAE decode、输出帧转换及调用内模型管理 |
| `other_pipeline_seconds` | `pipeline_seconds - dit_seconds - vae_decode_seconds` |
| `save_video_seconds` | `save_video(...)` 的视频编码与文件保存 |
| `total_seconds` | `pipeline_seconds + save_video_seconds`；不含读图/缩放、模型加载、预热、控制台输出或报告写入 |

各边界使用 `torch.cuda.synchronize(pipe.device)` 配合 `time.perf_counter()`，以获得 GPU 工作完成后的墙钟耗时。CUDA 异步调用需要同步后计时，参见 [PyTorch CUDA timing 文档](https://docs.pytorch.org/docs/main/notes/cuda.html#asynchronous-execution)。这是包含 Python 调度和内部数据搬运的阶段耗时。同步本身会带来测量开销，完整 pipeline 数值也是开启分段计时后的耗时；需要原始吞吐对照时可设 `TIMING_ENABLED=0`。

当前入口显式使用 `load_lora(..., hotload=False)`，在模型加载时把 rank64 LoRA 融合进 DiT 权重；因此记录的是融合了 LoRA 的 DiT 耗时。`inference_manifest.json` 的 `model_load_seconds` 包含模型装载和 LoRA 融合，未计入上述每条样本时间。

每成功生成并保存一条视频就 flush CSV 并更新 JSON。运行状态为 `running`、`complete` 或 `failed`；发生可捕获异常时保留已完成样本统计并抛出原异常。进程被强制终止时，状态可能保留为 `running`，已 flush 的记录仍在。重新执行会覆盖同名输出，不会自动续跑；比较不同配置时使用不同 `OUTPUT_DIR`。

## 验证

本地无需预训练权重的测试覆盖同步边界、CFG 计数、异常恢复、按尺寸预热、重复输出、报告统计、分片完整性及 LoRA 匹配，另外使用真实 Wan pipeline、调度器、TI2V units 和本地初始化的小型 DiT 验证计时前后同 seed 输出一致。shell 测试覆盖带空格路径、从其他目录启动、离线变量和错误码/日志传递：

```bash
python3 -m pytest -q tests/
bash -n infer_batch.sh infer_batch_lora.sh infer_batch_merged.sh
```

本轮 71 项测试通过，并实际完成 CPU 上的 MP4 编解码和合成本地 tokenizer 分词冒烟；本机没有 CUDA，未使用目标 Wan/LoRA 权重完成 GPU 全量验收。启动小算子预检可检查当前 GPU 后端是否工作，不能替代完整 5B 模型的显存和推理验证；真实耗时由内网机器执行获得。
