# CONDI：Topdown 条件生成与视频微调代码快照

本仓库汇集三套可复现、可审计的单场景视频生成训练流水线代码快照。它们共同围绕固定的 Topdown 相机合同，将 Gaussian 场景渲染为 72 帧 condition，再以严格配对的 clean target 训练或验证视频模型。仓库刻意不包含模型权重、训练 checkpoint、帧序列、视频、运行日志和虚拟环境。

## 仓库组成

| 目录 | 作用 |
| --- | --- |
| `inif_vace/` | InfiniSplat 条件生成、33 帧 cyclic paired dataset、Wan2.1-VACE-1.3B LoRA 训练与 Full72 QA 的 P39 代码整理包。 |
| `tripo_vace/` | 从 TripoSplat 条件渲染和动态清理，到 VACE-1.3B fresh LoRA 训练、续训、验证和后处理的 P49A 快照。 |
| `tripo_bernini/` | 使用同一类 Topdown/Tripo 条件链路，对 Bernini-R 1.3B 进行四卡训练、断点恢复、验证及 Full72 后处理的 P51A 快照。 |

每个目录都自带 `README.md`、架构说明、运行手册、决策记录、资产映射与校验清单；请从相应子项目的说明开始阅读。

## 共通数据流

```text
Topdown RGB + camera contract
  -> 72-frame condition rendering
  -> condition / clean 同相机轨迹配对
  -> 33-frame cyclic clips
  -> LoRA training and validation
  -> Full72 generation, metrics, and QA
```

关键边界是：condition 仅从 Topdown/Tripo 路径产生；clean RGB 只可作为训练目标、验证指标或对比输入，不能作为条件生成输入。相机及帧序号合同必须保持不变。

## 阅读与运行

1. 先阅读目标子项目的 `README.md`、`ARCHITECTURE.md` 和 `RUNBOOK.md`（若提供）。
2. 核对 `ASSET_MAP.json`、`PIPELINE_MANIFEST.json` 或对应合同，准备仓库外保存的原始数据、模型与输出路径。
3. 使用相应上游项目的 Python 环境与依赖声明，例如 `pyproject.toml`、`requirements.txt`，再运行各子项目提供的 shell wrapper 或 Python 入口。
4. 训练和 Full72 推理会占用 GPU 并生成大型产物；执行前请先检查脚本中的输入、输出与环境变量路径。

常用完整性检查：

```bash
cd tripo_vace && sha256sum -c CODE_MANIFEST.sha256
cd ../tripo_bernini && sha256sum -c CODE_MANIFEST.sha256
```

`inif_vace/audit_pipeline.py` 可基于 `PIPELINE_MANIFEST.json` 执行只读流水线审计。

## 仓库约定

- 本仓库追踪源码、配置、合同、轻量 manifest 与文档。
- 不追踪本地 `.env` 配置（`.env.example` 例外）、Python 缓存、虚拟环境或构建产物。
- 不将权重、checkpoint、数据集帧、推理视频和实验日志作为本快照的一部分；以各子项目的资产映射为准。

## 许可证与上游代码

仓库含固定版本的上游代码快照（如 Bernini、VeOmni、DiffSynth-Studio 与 TripoSplat）。使用或再分发前，请分别遵循各上游目录中的许可证和使用条款。
