# HANDOFF — P51A Bernini-R 1.3B

## State

Fail-closed before training. The official repository and the official 1.3B single-expert configuration were audited, and the paired data contract was verified. No checkpoint, inference output, metric, or visual claim is valid yet.

## What is ready

- Repository: `/fs1/private/user/baitongyuan/projects/liuzh/repos/Bernini`, commit `e6c2cf11843470b08491745947abcba9c3c6b079`.
- Data scripts and tests: `/fs1/private/user/baitongyuan/projects/liuzh/outputs/p51a_bernini_r13b_single_scene/scripts/`.
- Paired source manifest: `manifests/training_source.jsonl`.
- Official training config draft: `configs/bernini_r13b_single_scene.yaml`.
- Lossless 33-frame cyclic clips: `clips/adjusted/` and `clips/clean/`.
- Six data-contract tests pass.

## Blocker

Complete the fixed-revision `ByteDance/Bernini-R-1.3B-Diffusers` snapshot and install the repository's Python 3.11/Torch 2.7.1+cu126/VeOmni v0.1.11 environment. Verify every file against the Hub manifest before running official preprocessing. Do not substitute VACE weights or an unofficial LoRA route.

## Resume rule

Resume only from a complete checkpoint in the same `training/run_700` directory after a SIGTERM. Never warm-start from VACE/P49. After a trusted checkpoint exists, run the fixed starts `0/17/31/50` at every step `0/100/.../700`, then produce adjusted and raw Full72 outputs and metrics.
