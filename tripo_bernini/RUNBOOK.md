# 端到端运行手册

本手册说明真实代码入口和权威输入输出。它不自动执行命令；运行前必须检查 GPU、路径和现有 checkpoint，避免覆盖既有任务。

## 0. 环境与固定 revision

- 项目根：`/fs1/private/user/baitongyuan/projects/liuzh`
- TripoSplat revision：见 `SNAPSHOT_METADATA.json`
- Bernini revision：见 `SNAPSHOT_METADATA.json`
- VeOmni source：`repos/VeOmni-v0.1.11`
- Bernini 权重：`models/Bernini-R-1.3B-Diffusers`
- 正式训练使用 `MODELING_BACKEND=veomni`，在 `ulysses_size=4` 时解析为 `veomni_flash_attention_2_with_sp`。

## 1. Topdown 与 TripoSplat condition

1. 读取 `local/topdown_condition/00_topdown_assets/` 内的 camera 和 scene 合同。
2. 运行 `01_infinisplat/build_condition_cameras.py`、`infer_topdown.py`、`render_condition_72.py` 或 `run_topdown_to_condition.sh` 生成 72-frame condition 及 manifest。
3. TripoSplat 源码位于 `upstream/TripoSplat/`；其 Gaussian PLY 是后续 exact renderer 的几何输入。
4. 使用 `local/condition_pipeline/scripts/render_tripo_clean_path.py` 通过 P48 clean camera path 渲染 PLY。输入 camera 与 Clean RGB 的 frame stem 必须精确保持 F00 到 F71。
5. `phase_z_scenealign.py` 只对 Gaussian world 施加单一约束 similarity transform，不改 camera 和 image warp。

## 2. 清理并构建 adjusted dynamic condition

顺序如下：

1. `p48_1_tripo_cleanup.py`：清理高风险 Gaussian。
2. `p48_2_dynamic_suppression.py`：识别动态或近相机干扰。
3. `p48_3_dynamic_gaussian_cleanup.py`：逐帧 attribution 和全渲染 opacity attenuation 权重。
4. `phase_a_clean_adjusted_gaussian.py`：生成固定 Gaussian adjusted teacher。
5. `phase_b_dynamic_adjusted.py`：在精确 P48 相机下应用 causal EMA/hysteresis 权重，输出权威训练输入：
   `/fs1/private/user/baitongyuan/projects/liuzh/outputs/phase_b_adjusted_dynamic/condition_adjusted_rgb/F00.png` 到 `F71.png`。

Clean RGB 在该阶段只用于 comparison 或 QA，禁止作为 condition render 输入。详细 contract 在 `reference_contracts/phase_b_adjusted_dynamic/`。

## 3. P51A 数据构建与官方预处理

- Target：`outputs/p48_tripo_clean_aligned/clean_rgb/F00.png` 到 `F71.png`
- Input：`outputs/phase_b_adjusted_dynamic/condition_adjusted_rgb/F00.png` 到 `F71.png`
- `p51a_build_dataset.py` 生成 72 个 33-frame cyclic MKV clip 和 `training_source.jsonl`。
- `p51a_write_parquet.py` 调用官方 preprocessing 路径并生成 `preprocessed/training_source.parquet`。
- `manifests/p51a_v2v_sources.yaml` 声明 official weighted multisource 输入。
- `p51a_data_contract.py` 与 `scripts/tests/` 负责配对、clip 长度和序列化检查。

## 4. 训练、DCP 恢复和验证

- 配置：`p51a_runtime/configs/bernini_r13b_single_scene_4gpu_batch2.yaml`
- Trainer：`p51a_runtime/scripts/train_bernini_renderer_p51a.py`
- 每个训练 unit：`p51a_train_validate_cycles.sh`，正式 DCP 每 20 step 原子发布。
- 持续监督器：`p51a_persistent_guard.sh`。若系统杀进程，锁释放；cron 仅从最新包含 `P51A_DCP_COMPLETE.json` 的目录恢复。
- 每个 100 step：`p51a_validate_shard.py` 分别验证 start00、17、31、50；所有 shard marker 就绪后再由 finalizer 发布全局验证 marker。
- `clean_used_as_inference_input` 必须始终为 false。

## 5. 完训交付

`p51a_posttrain_controller.sh` 依次执行 checkpoint 选择、Full72 输入准备、adjusted/raw chunk 推理、72帧组装、同步对比与 `p51a_finalize.py` 报告生成。所有 final 文件输出到 P51A `final/`。
