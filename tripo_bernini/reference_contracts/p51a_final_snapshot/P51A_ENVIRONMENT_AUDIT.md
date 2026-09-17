# P51A Environment Audit

- Task: `P51A_BERNINI_R13B_SINGLE_SCENE_V2V`
- Status: **preflight passed; training is in progress**.
- Host: `liuzh-icl`; allowed workspace: `/fs1/private/user/baitongyuan/projects/liuzh/`.
- Python: `3.11.16`; PyTorch: `2.7.1+cu126`; CUDA runtime: `12.6`.
- GPUs: `4 × NVIDIA A40`, `46068 MiB` per GPU; world size `4`, Ulysses size `4`, FSDP2.
- Bernini repo commit: `e6c2cf11843470b08491745947abcba9c3c6b079`.
- Base snapshot: `ByteDance/Bernini-R-1.3B-Diffusers` at fixed revision `ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce`; manifest/SHA verification passed.
- Qwen2.5-VL-7B auxiliary weights: SHA verification passed.
- FlashAttention: `2.8.3`, source-built and CUDA-forward smoke-tested. `torchcodec==0.3.0` installed.
- Model smoke: high-noise transformer present; `transformer_2 is None`; T5 frozen.
- Data smoke: 72/72 official preprocessed paired rows; exact F00–F71 mapping; each 672² VAE latent has shape `[1, 32, 9, 84, 84]`.
- Four-GPU training smoke: forward/backward/DCP export passed (loss `0.1338`, grad norm `1.5368`), with all four ranks participating.
- Checkpoint inference smoke: in progress at the time this audit was updated; it uses adjusted condition only and clean exclusively for post-generation QA.

## Durability policy

The site terminates long four-GPU jobs at roughly 30 minutes. P51A therefore runs one isolated 20-step four-GPU training process at a time, publishes an atomic `P51A_DCP_COMPLETE.json`, and resumes only from that marker. A five-minute user cron guard re-launches the next unit. Partial checkpoints and partial validations are moved to timestamped audit paths rather than resumed or overwritten.

Before the fresh 700-step run, an isolated four-GPU batch-2 smoke determines whether global/micro batch 2 fits; on a true CUDA OOM the already validated batch-1 configuration is selected. No VACE/P49 checkpoint is loaded.
