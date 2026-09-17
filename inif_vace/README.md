# P39 single-scene quality push：可复现代码整理

本目录是服务器上已完成的 P39 实验的“代码与契约”整理包，位于：

```text
/root/autodl-tmp/outputs/p39_single_scene_quality_push/09_reproducible_code
```

它把完整路径串起来：

```text
topdown RGB
  → 官方 InfiniSplat RGB Gaussian artifact / scene.ply
  → P32 source-canonical camera transfer
  → 72 帧 condition RGB
  → condition/clean 同轨迹 pair manifest
  → 33-frame cyclic paired dataset
  → Wan2.1-VACE-1.3B rank-16 LoRA training
  → validation / metrics / full72 generation / QA
```

## 目录

- `00_topdown_assets/`：上游资产指针、P32 审计快照和 topdown→condition wrapper。
- `01_infinisplat/`：官方 InfiniSplat 推理、相机 payload、72 帧渲染、pair QA。
- `02_pair_dataset/`：P39 使用的 33 帧 cyclic 数据构造器和构造 shell。
- `03_training/`：训练器、固定配置 wrapper、历史命令脚本。
- `04_validation/`：多 start 验证 wrapper 和历史验证脚本。
- `05_metrics/`：指标、对比、full72 生成和 QA 脚本。
- `tests/`：原 P38/P39 单元测试；`audit_pipeline.py` 是本整理包的只读总审计。
- `PIPELINE_MANIFEST.json`：所有绝对路径、commit、checkpoint、数据契约和禁止事项。
- `CODE_INDEX.md`：逐脚本入口与来源对应表。

## 重要数据与防泄漏边界

- 选定 topdown 是 `topdown_normalized/topdown.png`，不是同目录下的其他 topdown 候选。
- condition 生成只读 topdown RGB、topdown camera、clean camera JSON 和已审计的标量 `final_lambda`；不读取 clean RGB/depth/normal/semantic 来生成几何或 condition。
- clean RGB 只在 pair dataset 的 target、validation 和 metric 阶段出现。
- condition frame `i` 使用 clean trajectory 的 camera `i`，不做 camera fitting、optical flow、homography 或 image registration。
- source-canonical translation 只接受 P32 已审计的 `final_lambda=0.3343257405583537`；缺少 scale audit 时 wrapper 直接失败，不会自行猜测尺度。
- 72 帧、896×896、帧 stem 和 camera pairing 是 fail-closed 条件。
- 33 帧 clip 是 cyclic `[start, start+1, …] mod 72`；训练 68 条，验证 start 为 `0,17,31,50` 共 4 条。

## 已运行资产与可复现入口

原实验代码和产物保持不变；本目录的 wrapper 不会覆盖原 P32/P39 输出。

1. 生成 topdown→condition（只在需要重新生成时执行）：

```bash
./01_infinisplat/run_topdown_to_condition.sh
```

默认会在本目录 `00_topdown_assets/reproduced/` 写入新 artifact、72 帧 PNG、视频、完整 pair manifest 和供 full72 推理使用的 condition-only manifest。它不会修改原始 P32 condition。

2. 生成 P39 paired dataset（默认输出到本目录 `datasets/`）：

```bash
./02_pair_dataset/build_p39_dataset.sh
```

也可通过 `PAIR_MANIFEST`, `DATASET_OUT`, `WIDTH`, `HEIGHT` 覆盖路径和统一 resize；不要对 condition 与 clean 使用不同设置。

3. 训练/验证入口：

```bash
./03_training/train_h672_900_950.sh
./03_training/train_h704_925_950.sh
./04_validation/validate_all_starts_672.sh
./04_validation/validate_all_starts_704.sh
```

这些是会产生大型 GPU 运行的命令；默认不执行。已有 checkpoint 和结果在原 P39 目录中。
若先在本整理包重新跑 H672，再跑 H704，请用 `SOURCE_LORA`/`SOURCE_OPTIMIZER` 指向新生成的 step925 文件；默认值保留原 P39 已运行 warm-start 的来源。

4. 只读审计：

```bash
/root/autodl-tmp/envs/p37_blackwell/bin/python audit_pipeline.py \
  --manifest PIPELINE_MANIFEST.json \
  --report audit_report.json
```

审计会检查 topdown、pair manifest 72 行、condition/clean 路径独立、`camera_exact_pair=true`、672/704 数据集的 68/4 manifests 和全部整理脚本的 Python 语法。

5. 已验证的 P39 full72 direct73 生成与 QA（会运行 GPU 推理，默认不执行）：

```bash
./05_metrics/run_full72_direct73.sh
./05_metrics/run_full72_qa.sh
```

`make_condition_only_manifest.py` 将 pair manifest 投影为 `p39_full72.py` 要求的严格三字段 condition-only manifest；clean RGB 不会被传入 full72 condition generation。

## P39 已验证训练配置

```text
base model       Wan2.1-VACE-1.3B
LoRA rank        16
clip             33 frames
batch            1 × gradient accumulation 4 = effective batch 4
precision        bf16
learning rate    1e-4
weight decay     1e-2
seed             20260911
gradient ckpt    false
offload          false
```

P39 是 warm-start：先从 P38 step900 运行 672×672 到 step950（选择 step925），再从 step925 运行 704×704 到 step950。最终 handoff 与 checkpoint 仍在原目录：

```text
/root/autodl-tmp/outputs/p39_single_scene_quality_push/final/HANDOFF_P39.md
/root/autodl-tmp/outputs/p39_single_scene_quality_push/checkpoints/P39_H672_STEP0925_BEST.safetensors
```

## 来源仓库状态

InfiniSplat commit 为 `c56965cdf32b4e8e1178ee14410e3fbf26a87705`，DiffSynth-Studio commit 为 `afd101f3452c9ecae0c87b79adfa2e22d65ffdc3`。两者工作区在盘点时均为 dirty；本整理包不 reset、checkout、clean 或覆盖这些用户修改。

P39 运行环境记录为 Python 3.10.12、PyTorch 2.7.1+cu128（CUDA 12.8），GPU 为 NVIDIA RTX PRO 6000 Blackwell Server Edition；详见 manifest。版本信息来自已有训练运行记录。

## 历史脚本与新 wrapper

`03_training/legacy/`、`04_validation/legacy/`、`05_metrics/legacy/` 是从原 P39 `01_code` 复制的已运行脚本，便于逐字追溯；同级 wrapper 使用本整理包的路径和环境变量，便于重新执行。历史 P20/P23/P24 脚本没有被冒充成 P32 exact producer；P32 风格的 producer 是 `01_infinisplat/` 中明确标注的 wrapper。
