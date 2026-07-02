# 任务计划：Wan2.2-TI2V-5B LoRA SFT 与训练指标监控

## 目标
在当前 DiffSynth-Studio 仓库中完成 Stage A 可离线验证的 Wan2.2-TI2V-5B LoRA 训练辅助工具、指标记录、绘图、数据集 smoke test 与 Stage B 上机 checklist。

## 当前阶段
完成

## 各阶段

### 阶段 1：需求与源码发现
- [x] 阅读任务书
- [x] 通读 `examples/wanvideo/model_training/train.py` 的 argparse
- [x] 通读实际训练循环、保存、日志、数据集入口
- [x] 确认 Wan2.2-TI2V-5B 官方 LoRA 示例参数与本地权重参数
- [x] 将关键发现记录到 `findings.md`
- **状态：** complete

### 阶段 2：计划与测试先行
- [x] 写入实施计划
- [x] 为 metrics、绘图去重、数据生成/体检、数据集 smoke 准备 pytest 覆盖
- [x] 先运行目标测试确认失败
- **状态：** complete

### 阶段 3：实现 Stage A 交付物
- [x] 新增 `metrics_utils.py`
- [x] 最小修改 trainer 增加 `--metrics_path`
- [x] 新增 `make_debug_dataset.py`
- [x] 新增 `check_dataset.py`
- [x] 新增 `plot_metrics.py`
- [x] 新增 `train_ti2v5b_lora.sh`
- [x] 新增/更新 `.gitignore`
- [x] 新增 `NOTES.md`
- **状态：** complete

### 阶段 4：测试与验证
- [x] `pytest tests/` 全绿
- [x] 生成 synthetic debug dataset，并用 `check_dataset.py` 验证坏样本报告
- [x] fake metrics 生成与 `plot_metrics.py` PNG 输出验证
- [x] `bash -n train_ti2v5b_lora.sh`
- [x] `train.py --help` 覆盖脚本中使用的参数名
- [x] 数据集迭代 smoke test 通过或写明降级原因
- **状态：** complete

### 阶段 5：审查、提交与交付
- [x] 审查 diff 确保默认关闭时上游行为不变
- [x] 确认无散落硬编码路径
- [x] 记录最终自测结果
- [x] 视远端状态提交并推送 GitHub
- **状态：** complete

## 关键问题
1. `train.py` 是否已有本地权重参数，参数名和传参形式是什么？答：已有 `--model_paths`，JSON 列表，本地 path/glob。
2. `input_image` 是否需要加入 `--data_file_keys`？答：建议显式加入以加载真实列，但当前 TI2V trainer 仍用 `data["video"][0]`。
3. 现有训练循环中 optimizer step 的准确位置、step/epoch 变量名、world size/batch size 来源是什么？答：见 `diffsynth/diffusion/runner.py`，需新增真实 optimizer step 计数。
4. 数据集类能否在 Mac CPU-only 环境直接 import 和迭代？答：可以，`tests/test_dataset_smoke.py` 已通过。

## 已做决策
| 决策 | 理由 |
|------|------|
| 任务书作为已批准实现规格 | 用户明确要求逐步精确完成任务，不再额外阻塞等待设计确认 |
| 所有新增工具模块保持 CPU-only/import-safe | 任务硬性约束要求 Stage A 无 GPU/权重可验证 |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| 试图新建 goal 时发现已有活跃 goal | 1 | 沿用当前活跃 goal |

## 备注
- 每完成阶段更新本文件与 `progress.md`。
- 外部/不可信内容不写入本文件；源码发现写入 `findings.md`。
