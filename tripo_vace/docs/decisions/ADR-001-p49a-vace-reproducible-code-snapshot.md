# ADR-001：P49A 可重放源码快照

将 P49A 的 condition 上游、项目自定义训练代码和实际使用的 DiffSynth-Studio 工作树放入同一只读快照。保留 working-tree diff 是必要的：P49A 使用的 gradient checkpoint、VACE 模型/管线修正并非纯 upstream commit 内容。数据和模型参数不复制，以避免权重/帧重复和训练干扰；资产映射而非数据负载成为可审计接口。
