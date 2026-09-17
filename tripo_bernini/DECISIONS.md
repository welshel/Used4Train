# 关键决策与已知限制

## 相机与 Clean 约束

P48 clean camera trajectory 是不可变合同。对齐只允许作用在 Gaussian world transform。Clean 不得成为 condition render 或 Bernini 推理输入。

## Adjusted dynamic 而非 raw condition

当前训练使用 `phase_b_adjusted_dynamic/condition_adjusted_rgb`。该路线通过保持 Gaussian identity 和因果时间权重减轻白色近景干扰；raw condition 仅用于训练后泛化评估。

## 33-frame cyclic paired clips

训练不是随机的单帧配对。每一个 start 使用相同 frame index 同时从 condition 与 clean 取 33 帧，跨越结尾时 cyclic wrap。固定验证 starts 是 0、17、31、50。

## Four-GPU SP attention

P51A 使用 `ulysses_size=4`。GPU0 Xid 13 后，恢复 smoke 验证官方 VeOmni path，将 FA2 解析为 `veomni_flash_attention_2_with_sp`。正式训练从完整 DCP 恢复，不使用隔离 smoke checkpoint。

## 连续监督与 fail-closed

持续监督器连续执行训练和验证 unit，不主动暂停。DCP 与验证 shard 均以原子 marker 作为恢复依据。语义错误写入 fail-closed state 后，cron 不可盲目重试；必须先完成独立 smoke 与人工审阅。

## 已知视觉限制

TripoSplat 为生成 Gaussian 场景，植物、窗帘、细小家具和窗口高光不必像素级匹配 Clean。训练和人工 QA 应重点比较 ghosting、double object、translucency、时间 flicker 与场景身份保持。
