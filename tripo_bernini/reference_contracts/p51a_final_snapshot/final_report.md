# P51A_BERNINI_R13B_SINGLE_SCENE_V2V — final report

## Decision

**FAIL-CLOSED: training was not started.** No Bernini-R 1.3B result is claimed.

The official Bernini repository does contain a dedicated 1.3B Wan2.1 single-expert config (`configs/bernini_renderer_wan21_1p3b/config.json`), so the architecture route is auditable. However, the complete fixed-revision model snapshot and the official Python 3.11/Torch/VeOmni runtime were not available in a verified state. Starting training or fabricating checkpoint metrics would violate the task's fail-closed rule.

## Completed checks

- SSH host: `liuzh-icl`.
- Repository commit: `e6c2cf11843470b08491745947abcba9c3c6b079`.
- Official model: `ByteDance/Bernini-R-1.3B-Diffusers`, revision `ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce`.
- Four NVIDIA A40 GPUs were visible.
- Adjusted, clean, and raw sequences each contain exact F00–F71 RGB frames.
- 72 cyclic 33-frame adjusted clips and 72 cyclic 33-frame clean clips were created losslessly.
- Pairing and inference-safety tests passed: 6/6.

## Required questions

- Did Bernini-R 1.3B training succeed? **No; it was not launched.**
- Did every-100-step validation improve or remain stable? **Not applicable; no checkpoint exists.**
- Best checkpoint? **None.**
- Is Bernini better than adjusted input? **Cannot assess.**
- Did it reduce ghosting/translucency/double objects? **Cannot assess.**
- Is raw-route generalization effective? **Not tested.**
- Is it visually better than VACE-1.3B? **Cannot assess.**
- Largest residual problem? **Runtime gate: complete official weights and a usable official training environment were not verified.**
- Continue Bernini-R? **Yes, conditionally: finish the fixed-revision download and environment installation, run a one-batch official smoke test, and only then resume the 700-step plan.**

## Outputs intentionally absent

There are no checkpoint directories, prediction videos, comparison videos, or validation metrics. The files `P51A_CHECKPOINT_SUMMARY.json`, `P51A_VALIDATION_SUMMARY.json`, and `P51A_BEST_CHECKPOINT.txt` explicitly record this absence rather than inventing values.

See `P51A_ENVIRONMENT_AUDIT.md` for the complete audit and `HANDOFF_P51A.md` for the exact resume procedure.
