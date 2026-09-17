# P49A Topdown-to-VACE 代码快照

这是 `P49A_TRIPO_VACE13B_FRESH` 的只读、可审计源码快照。它覆盖从 Topdown 相机合同和 TripoSplat 条件渲染，经 P48 clean 相机对齐与动态 Gaussian 清理，直到 VACE-1.3B fresh LoRA 四卡训练、同 run 断点续训、固定验证和后处理的实际代码。

本包只含源码、启动脚本、配置、manifest、环境/模型审计及轻量 QA 合同；不含权重、PNG 帧、视频、LoRA checkpoint、训练日志、虚拟环境、缓存或 Git 元数据。权威资产位置和 clean 使用边界见 `ASSET_MAP.json`。

## 阅读顺序

1. `ARCHITECTURE.md`：端到端数据流及真实产物路径。
2. `RUNBOOK.md`：运行入口与依赖关系。
3. `DECISIONS.md`：数据配对、clean 边界、fresh LoRA/续训和本地 DiffSynth 修改。
4. `local/topdown_condition/` 与 `upstream/TripoSplat/`：Topdown 到 72 帧 condition。
5. `local/condition_pipeline/scripts/`：P48 对齐、清理、动态抑制。
6. `p49a_runtime/scripts/` 与 `p49a_runtime/launchers/`：P49A VACE 训练、恢复、验证、Full72/QA。
7. `upstream/DiffSynth-Studio/`：训练真实使用的固定 working tree；本地改动补丁在该目录内。

## 完整性

在本目录执行：`sha256sum -c CODE_MANIFEST.sha256`。
