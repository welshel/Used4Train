# P51A Topdown-to-Bernini 代码快照

这是在 P51A 四卡连续训练期间创建的只读代码快照。它把从 Topdown 与 TripoSplat Gaussian 场景、经 clean 相机对齐和动态抑制得到 adjusted condition，到 Bernini-R 1.3B 单场景 v2v 微调、周期验证和 Full72 后处理的实际代码集中到一个目录。

快照只含代码、配置、manifest 和轻量级合同。权重、PNG 序列、视频、DCP checkpoint、训练日志和虚拟环境不在此包内；其权威位置见 [ASSET_MAP.json](ASSET_MAP.json)。

## 阅读顺序

1. [ARCHITECTURE.md](ARCHITECTURE.md)：端到端数据流和代码边界。
2. [RUNBOOK.md](RUNBOOK.md)：真实入口、输入输出和运行顺序。
3. [DECISIONS.md](DECISIONS.md)：不可变相机、Clean 使用边界、恢复与 SP 决策。
4. `local/condition_pipeline/scripts/`：TripoSplat 对齐、清理、动态 adjusted condition。
5. `p51a_runtime/scripts/`：paired 数据集、Bernini 训练、验证、后处理。
6. `upstream/Bernini/` 与 `upstream/VeOmni-v0.1.11/`：固定 revision 的官方训练实现。

## 目录

| 路径 | 内容 |
| --- | --- |
| `local/topdown_condition/` | Topdown 合同与 condition 相机、72帧渲染入口 |
| `upstream/TripoSplat/` | TripoSplat 源码快照 |
| `local/condition_pipeline/scripts/` | 对齐、P48 清理、P49 adjusted dynamic condition |
| `p51a_runtime/` | P51A 自定义数据、训练、验证、恢复、后处理代码与配置 |
| `upstream/Bernini/` | Bernini 官方源码，排除 `.venv` 和 Git 元数据 |
| `upstream/VeOmni-v0.1.11/` | VeOmni 官方 runtime 源码，排除 build 缓存 |
| `reference_contracts/` | P48、Phase B、P51A 的轻量 handoff 和 QA 合同 |

## 快照完整性

`CODE_MANIFEST.sha256` 列出所有快照文件的 SHA256。用 `sha256sum -c CODE_MANIFEST.sha256` 验证。
