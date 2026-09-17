# P49A 实际链路

```text
Topdown assets + condition camera contract
  -> local/topdown_condition/01_infinisplat/{infer_topdown,build_condition_cameras,render_condition_72}.py
  -> upstream/TripoSplat/ render source
  -> P48 clean-camera alignment (render_tripo_clean_path.py + phase_z_*.py)
  -> P48 dynamic cleanup (p48_1 -> p48_2 -> p48_3)
  -> raw dynamic RGB condition F00..F71
  -> p49a_prepare_dataset.py: 33-frame cyclic PNG-list manifest
  -> p49a_train.py: VACE-1.3B fresh LoRA
  -> p49a_train_resume.py + segmented supervisor: same-run LoRA continuation
  -> p49a_validate.py: condition-only generation + clean target-only metrics
  -> p49a_postprocess.py: checkpoint ranking, Full72 overlap assembly, boards/QA
```

P49A 训练实际条件输入：
`/fs1/private/user/baitongyuan/projects/liuzh/outputs/p48_3_tripo_dynamic_cleanup/06_full72/condition_rgb_dynamic/F00.png` 至 `F71.png`。

P49A clean target/QA：
`/fs1/private/user/baitongyuan/projects/liuzh/outputs/p48_tripo_clean_aligned/clean_rgb/F00.png` 至 `F71.png`。

训练 clip 为 33-frame cyclic clips；每个 PNG 列表保留 F00–F71 的严格 stem 映射。Clean 只允许 target 和 QA，禁止作为 VACE 的 `vace_video`、`vace_reference_image` 或生成输入。
