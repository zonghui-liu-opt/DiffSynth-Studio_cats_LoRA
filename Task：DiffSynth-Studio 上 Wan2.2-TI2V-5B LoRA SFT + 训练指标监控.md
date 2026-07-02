# Task：DiffSynth-Studio 上 Wan2.2-TI2V-5B LoRA SFT + 训练指标监控

**（Stage A：MacBook 本地调试 → push GitHub → Stage B：内网 H100 正式训练）**

## 0. 角色、工作流与总原则（必须遵守）

你是资深 ML infra 工程师，在 `modelscope/DiffSynth-Studio`（main 分支）上完成开发。工作流分两阶段：

- **Stage A（当前，你全程参与）**：MacBook（Apple Silicon，CPU-only，无 CUDA / 无 flash_attn / 无真实数据 / 无模型权重）。你需要自己生成合成调试数据，把一切可离线验证的逻辑测通，然后 push GitHub。
- **Stage B（你不在场）**：内网 H100 80G × 2~4，**无外网**（不能访问 HF/ModelScope 自动下载权重），真实数据与权重均在本地磁盘。用户将手动按你写的 checklist 操作。

硬性约束：

1. **不重复造轮子**：训练主体复用官方 `examples/wanvideo/model_training/train.py` 与官方 `lora/Wan2.2-TI2V-5B.sh` 示例参数（rank=32、`lora_target_modules "q,k,v,o,ffn.0,ffn.2"`、`--extra_inputs "input_image"`、`--remove_prefix_in_ckpt "pipe.dit."`）。动手前先通读 `train.py` 的 argparse 与 `diffsynth/trainers/` 训练循环，已有功能（日志、按 epoch 保存、缓存机制等）一律直接启用。
2. **原地最小侵入**：新增逻辑用 `if args.xxx:` 分支挂在现有路径上，默认关闭时与上游行为完全一致。禁止另起训练循环。
3. **Stage B 只改配置不改代码**：所有环境差异（路径、卡数）必须收敛到脚本顶部变量块。任何散落的硬编码路径都算 bug。
4. **新增的工具模块必须可脱离 GPU/权重独立 import 和测试**（不得在 import 时触发 CUDA 初始化或加载模型）。
5. 逻辑精准：所有统计量定义写进注释；参数名以仓库当前源码为准，不凭记忆猜 API。每完成一个子任务，输出 diff 摘要 + 自测结果。

## 1. 环境与数据背景

- **真实数据（仅 Stage B 存在）**：约 600 条视频。`metadata.csv` 为 **tab 分隔**，三列：`video`（如 `videos_480x832/51_cgt-xxx.mp4`）、`prompt`、`input_image`（如 `images_480x832/51_cgt-xxx.jpg`，TI2V 首帧条件图）。
- **真实权重（仅 Stage B 存在）**：Wan2.2-TI2V-5B 全套（DiT safetensors、umT5 encoder、Wan2.2 VAE），Stage A 中一律用占位符 `/path/to/local/wan/...` 表示，你需要读源码确认 train.py 用哪个参数接收**本地权重路径**（内网无法走 `model_id_with_origin_paths` 的自动下载路径；确认是否有 `--model_paths` 之类的本地加载参数），并在 NOTES.md 里列出 Stage B 需要在 `MODEL_ROOT` 下放置的确切文件清单。
- 训练目标：LoRA rank=32 SFT，按 epoch 保存 ckpt，产出 loss 曲线与 token 吞吐曲线（tokens/s），折算"每小时训练视频条数"。

## 2. 子任务

### T0 配置收敛与占位

所有 shell 脚本顶部统一变量块：`MODEL_ROOT`（默认 `/path/to/local/wan`）、`DATA_ROOT`、`OUTPUT_ROOT`、`NUM_GPUS`（2/4 可切）、`METRICS_PATH`、`HEIGHT/WIDTH/NUM_FRAMES` (数据分辨率可能是480x832或832x480)。新增 `.gitignore`：`debug_data/`、`models/`、`*.jsonl`、输出 PNG 不入库（生成器脚本入库，二进制产物不入库）。

### T1 合成数据生成器 `make_debug_dataset.py`（新文件）

Stage A 没有真实数据，你要自己造，且造出来的数据必须**在格式上精确复刻真实数据的坑**：

1. 参数：`--num_videos`（默认 8）、`--height/--width`（默认 96×160，必须是 32 的倍数）、`--num_frames`（默认 13，满足 `(F-1) % 4 == 0`）、`--output_dir`（默认 `debug_data/`）、`--with_bad_samples`。
2. 用仓库已有依赖（优先 `imageio`，其次 cv2）生成随机移动色块的 mp4 + 对应首帧 jpg，目录结构与真实数据一致（`videos_HxW/`、`images_HxW/`）。
3. **metadata.csv 必须用 tab 分隔**（复刻真实文件），使 T2 的分隔符探测/转换逻辑得到真实测试。
4. `--with_bad_samples` 时额外注入 2 条坏样本：一条帧数不足、一条 `input_image` 路径不存在——用于测试 T2 的报错路径。

### T2 数据体检脚本 `check_dataset.py`（新文件，~100 行）

只读脚本，做且只做：

1. 探测 `metadata.csv` 分隔符（tab/逗号）；先读 DiffSynth 数据集源码确认其解析方式，若真实文件（tab）不兼容，则输出转换后的标准 `metadata_fixed.csv`。校验三列齐全、所有 `video`/`input_image` 路径存在。
2. 统计全部视频真实 宽×高、帧数、fps 分布（**优先用 imageio/cv2 读取，ffprobe 仅作为可选 fallback**——Mac 上未必装了 ffmpeg）。目录名 `videos_480x832` 无法确定是 W×H 还是 H×W，以实测为准，输出训练用 `--height/--width` 建议值（32 的倍数）。
3. 输出每条视频 token 数（调用 T4 的 `metrics_utils.tokens_per_sample()`，不要重复实现公式）与全数据集 token 总量。

**Stage A 验收**：对 `make_debug_dataset.py --with_bad_samples` 的产物运行，正常样本统计正确、2 条坏样本被准确报出。

### T3 训练启动脚本 `train_ti2v5b_lora.sh`（新文件）

以官方 TI2V-5B LoRA 示例为唯一基准，权重改为本地路径占位（参数名以仓库实际 argparse 为准）：

```bash
# ====== Stage B 上机时只改这一块 ======
MODEL_ROOT=/path/to/local/wan
DATA_ROOT=/path/to/dataset          # Stage A 调试时指向 debug_data/
OUTPUT_ROOT=./models/train/Wan2.2-TI2V-5B_lora
NUM_GPUS=4
HEIGHT=480; WIDTH=832; NUM_FRAMES=121   # 以 check_dataset.py 实测为准
METRICS_PATH=$OUTPUT_ROOT/metrics.jsonl
# =====================================

accelerate launch --num_processes $NUM_GPUS --mixed_precision bf16 \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path $DATA_ROOT \
  --dataset_metadata_path $DATA_ROOT/metadata.csv \
  --height $HEIGHT --width $WIDTH --num_frames $NUM_FRAMES \
  <本地权重加载参数，指向 $MODEL_ROOT 下的 DiT/T5/VAE，读源码定> \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path $OUTPUT_ROOT \
  --lora_base_model dit \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --metrics_path $METRICS_PATH
```

要求：确认 `input_image` 是否需显式加入 `--data_file_keys`（官方示例未传，读源码确认默认值；不确定就显式传 `"video,input_image"`）；确认 ckpt 是否按 epoch 保存（600 条 × 4 卡一个 epoch 仅 ~150 step，按 epoch 即可）；确认 LoRA + DDP 是否需要 `--find_unused_parameters`。

**Stage A 只做两件事，不要试图在 Mac 上真正跑训练**：(a) `bash -n` 语法检查 + argparse `--help` 能列出所有用到的参数名（包括你新增的 `--metrics_path`）；(b) 见 T7 的数据集迭代 smoke。

### T4 指标工具模块 `metrics_utils.py`（新文件）+ trainer 最小 diff

为了让核心逻辑在 Mac 上可测，**计算与 I/O 全部放进独立纯函数模块**，trainer 里的 if 分支只负责调用：

1. `metrics_utils.py` 内容（无任何 torch/CUDA 依赖）：

   - `tokens_per_sample(num_frames, height, width)`：

     ```python
     # Wan2.2-TI2V-5B: VAE 压缩 4x16x16, 经 DiT patchify 后总压缩比 4x32x32
     latent_frames = (num_frames - 1) // 4 + 1
     return latent_frames * (height // 32) * (width // 32)
     # 例: 121帧 480x832 -> 31 * 15 * 26 = 12090 tokens/视频
     ```

   - `MetricsWriter`：append 模式写 JSONL，一行一 step，字段：`step`、`epoch`、`loss`、`step_time_sec`、`tokens_per_sample`、`samples_per_step`、`lr`。

2. trainer diff：新增 CLI 参数 `--metrics_path`（默认 `None`，None 时零行为变化）。开启时仅 rank 0（`accelerator.is_main_process`）在每个 optimizer step 后写一行；`loss` 取 rank0 本地 `.item()`（注释说明未做 all-reduce 及原因）；`step_time_sec` 用 `time.perf_counter()` 差值；`samples_per_step = micro_batch_size * grad_accum * world_size`，全部从 accelerator/args 取真实值，不硬编码。

3. `tests/test_metrics_utils.py`（pytest）：token 公式对拍（含 121/480/832→12090 这组已知值）、JSONL 写入-读回一致性、append/resume 场景 step 去重逻辑。

**Stage A 验收**：`pytest tests/` 全绿，且 `metrics_utils.py` 可在未安装 flash_attn 的 Mac 环境单独 import。

### T5 绘图脚本 `plot_metrics.py`（新文件，仅依赖 matplotlib）

1. **loss 曲线**：x=step，raw（浅色）+ EMA 平滑（深色，默认系数 0.98）。
2. **吞吐曲线**：双 y 轴——左轴 `tokens_per_step / step_time_sec`（全局 tokens/s），右轴 `videos_per_hour = 3600 * samples_per_step / step_time_sec`。丢弃前 K 个 warmup step（默认 3）后标注稳态均值。支持 `--tokens_per_video` 覆盖参数（未来数据 token 异构时用 `tokens/s ÷ tokens_per_video` 折算）。
3. 结尾打印文本摘要：总 step、末 20% 平均 loss、稳态 tokens/s、videos/hour、预计全程耗时。
4. 支持训练中途重复执行；resume 时 JSONL 为 append，绘图按 step 去重取最后一条。

**Stage A 验收**：写一个 `tests/gen_fake_metrics.py`（或 pytest fixture）生成 ~300 行带噪声下降 loss + 含 warmup 慢 step + 含 resume 重复 step 的假 JSONL，跑 `plot_metrics.py` 产出两张 PNG，人工核对一行数值：`tokens_per_step / step_time` 与图上一致。

### T6 数据集加载 smoke test（Mac 可跑，不需要权重）

从 diffsynth 中直接 import 训练数据集类（读源码确认类名与构造签名），用 `debug_data/` 的合成 metadata 实例化，迭代 2 个样本，断言：`video` 张量形状符合 `num_frames/height/width`、`prompt` 为 str、`input_image` 被正确加载。**这一步是 Stage A 能对"训练输入正确性"做的最强验证**。若该类 import 链上不可避免地依赖 GPU-only 包，降级为对 metadata 解析逻辑的针对性测试，并在 NOTES.md 说明降级原因。

### T7 Stage B 上机 Checklist（写入 `NOTES.md`，给人看的，必须可照做）

你不在场，所以每条都要精确到命令级：

1. **权重清单**：`MODEL_ROOT` 下需放置的确切文件名（DiT `diffusion_pytorch_model*.safetensors`、`models_t5_umt5-xxl-enc-bf16.pth`、`Wan2.2_VAE.pth`，以源码加载逻辑为准）。
2. **上机顺序**：先对真实数据跑 `check_dataset.py` → 按输出修正脚本顶部 `HEIGHT/WIDTH/NUM_FRAMES` → 先 `NUM_GPUS=2` + 真实数据前 16 条（`head` 出一个 mini metadata）跑 20 step 冒烟 → 确认 `metrics.jsonl` 逐行正常、无 OOM → 切 4 卡全量。
3. **榨干 H100 的实验序列**（逐项记录 tokens/s 与 `torch.cuda.max_memory_allocated()` 峰值，收益不明的改动不保留）：确认 flash attention 生效；gradient checkpointing 先关（换速度）OOM 再开；dataloader `num_workers` 调优（GPU util 周期性掉底 = 解码瓶颈）；**调研仓库是否已有 T5/VAE 预缓存机制，有则启用**（600 条缓存成本极低），没有则不自行实现；显存富余时尝试 micro batch > 1。
4. **验证出片**：参考仓库 `validate_lora` 示例写 5 行冒烟脚本，`pipe.load_lora(pipe.dit, ...)` 加载 `epoch-*.safetensors` 出一段视频。
5. 每项预期观测值 + 失败时的回退动作。

## 3. 交付物

新文件：`make_debug_dataset.py`、`check_dataset.py`、`train_ti2v5b_lora.sh`、`metrics_utils.py`、`plot_metrics.py`、`tests/`（含 fake metrics 生成）、`.gitignore`、`NOTES.md`（Stage B checklist + 权重清单 + 降级说明）；对 trainer 的最小 diff（`--metrics_path` 分支）。

## 4. 验收标准（分阶段）

**Stage A（Mac，你必须全部自证后才允许 push）**：

- `make_debug_dataset.py` → `check_dataset.py` 全链路在合成数据上输出正确，坏样本被准确识别；
- `pytest tests/` 全绿；`plot_metrics.py` 在假 JSONL 上产出两张自洽的 PNG；
- T6 数据集迭代 smoke 通过（或有书面降级说明）；
- 代码审查确认：`--metrics_path` 不传时，运行行为与上游逐行一致；无任何散落的硬编码路径。

**Stage B（H100，人工按 NOTES.md 执行）**：

- 4 卡跑通 ≥1 epoch，`output_path` 下出现 `epoch-*.safetensors`，`metrics.jsonl` 每 step 一行无缺失，两张 PNG 数值自洽；
- LoRA 可被 pipeline 加载出片；2↔4 卡切换只改脚本顶部变量。