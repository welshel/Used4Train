# HANDOFF P3.9

## Final State

- Status: `P39_SINGLE_SCENE_QUALITY_PUSH_PASS`
- Scene: `327431980_4_normalized`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Environment: `/root/autodl-tmp/envs/p37_blackwell/bin/python`
- Model: DiffSynth Wan2.1-VACE-1.3B

## Best Model

- Checkpoint: `/root/autodl-tmp/outputs/p39_single_scene_quality_push/checkpoints/P39_H672_STEP0925_BEST.safetensors`
- SHA256: `936b16703099c232c28cf5738ac323b5ee31f1a11e377619349aaa542d99f86a`
- Warm start: `/root/autodl-tmp/outputs/p38_high_quality_roomtour/checkpoints/P38_H640_STEP0900_BEST.safetensors`
- Training: 672×672, 33f, batch1, accumulation4, bf16, no GC/offload
- Best point: optimizer step925, 25 added steps from P3.8
- Inference: 704×704
- Structural supervision: NO

## Best Outputs

- 33f: `/root/autodl-tmp/outputs/p39_single_scene_quality_push/08_final_videos/P39_BEST_RAW_33F_START31_INFER704.mp4`
- Full72: `/root/autodl-tmp/outputs/p39_single_scene_quality_push/08_final_videos/P39_BEST_FULL72_RAW_ROT08.mp4`
- Full comparison: `/root/autodl-tmp/outputs/p39_single_scene_quality_push/08_final_videos/P39_P38_VS_P39_FULL72_COMPARISON.mp4`
- Slow loop comparison: `/root/autodl-tmp/outputs/p39_single_scene_quality_push/08_final_videos/P39_P38_VS_P39_LOOP_PREVIEW_SLOW.mp4`

## Full72 Contract

- Method: Direct73
- Rotated start: 8
- Output reorder: canonical F00–F71
- Seed: 20260911
- Inference steps: 20
- CFG: 5.0
- VACE scale: 1.0
- Preview FPS: 12, operational only
- Inference inputs: Native InfiniSplat Gaussian RGB condition, camera encoded in condition frames, text prompt, model parameters
- Clean inference inputs: none

## QA Conclusion

- Four-start 33f aggregate: all primary metrics improve versus P3.8 at 704 inference.
- Full72: +10.15% Laplacian variance; small ~3% temporal/flicker trade-off; human playback PASS.
- Camera/layout/scene identity: PASS.
- Model boundary and F71→F00 loop: automated BORDERLINE, human slow-playback PASS.
- Plant/curtain detail: improved; furniture identity: same.
- Clean depth/normal/lines: QA_ONLY.
- Tests: 40 unittest PASS; py_compile PASS.

## Rollback

Do not modify or overwrite:

- `/root/autodl-tmp/outputs/p38_high_quality_roomtour/checkpoints/P38_H640_STEP0900_BEST.safetensors`
- `/root/autodl-tmp/outputs/p38_high_quality_roomtour/11_final_videos/P38_BEST_FULL72_RAW_ROT16.mp4`

If P3.9 is unsuitable for a downstream scene, roll back to the P3.8 paths above.

## Next

Proceed to multi-scene generalization only from a new output root. Keep P3.9 frozen as the single-scene quality reference and retain the same no-clean-inference contract.
