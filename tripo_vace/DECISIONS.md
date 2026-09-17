# P49A 不可变决策

- 模型：Wan2.1-VACE-1.3B，fresh rank-16 LoRA；不加载旧任务 LoRA。
- 数据：RGB→RGB；672×672；33 帧 cyclic；输入是 P48 raw dynamic condition，target 是同 stem clean RGB。
- 条件：训练和推理时 `vace_video` 与 `vace_reference_image` 均来自 condition；clean 仅用于 target/验证/指标/对比。
- 续训：使用同一 run 最近完整 `step-*.safetensors` LoRA；AdamW 重新初始化，scheduler 定位到全局 step；不把它表述为旧任务 warm start。
- 训练实现：VACE 使用 `repos/DiffSynth-Studio` commit `c458cb42ab1ee838bff85c6546e14bb01c3571e9` 的工作树；4 个本地已修改文件同时被源树快照和补丁保存。
- 上游 TripoSplat 固定为 commit `d8db9e018b413dd9c4a9fe22463781bf98e8e68d`。
