# 运行入口索引（历史真实路径）

| 阶段 | 入口 |
| --- | --- |
| Topdown/condition | `local/topdown_condition/01_infinisplat/run_topdown_to_condition.sh` |
| P48 对齐与清理 | `local/condition_pipeline/scripts/render_tripo_clean_path.py`、`phase_z_*.py`、`p48_{1,2,3}_*.py` |
| P49A dataset manifest | `p49a_runtime/scripts/p49a_prepare_dataset.py` |
| 4×A40 fresh train | `p49a_runtime/launchers/training/launch_train_4gpu.sh` |
| same-run LoRA segmented resume | `p49a_runtime/launchers/training/run_segmented_resume_supervisor.sh` |
| selective-GC benchmarks | `p49a_runtime/scripts/p49a_benchmark_selective_gc*.sh` 与 `launchers/benchmark/` |
| four-start validation | `p49a_runtime/launchers/validation/07_start31_validation_run_parallel.sh` |
| metrics, ranking, Full72, QA | `p49a_runtime/scripts/p49a_postprocess.py` |

项目脚本在原工作区以 `repos/DiffSynth-Studio` 插入 `sys.path`；本快照以 `upstream/DiffSynth-Studio` 保存该精确源树及 `P49A_LOCAL_WORKTREE.patch`。要重放历史运行，应先恢复同一工作区布局或将该目录挂载/链接为 `repos/DiffSynth-Studio`，再运行原启动器。
