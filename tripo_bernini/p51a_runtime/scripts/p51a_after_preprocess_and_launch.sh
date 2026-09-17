#!/usr/bin/env bash
# Fail-closed handoff from full preprocessing to 4xA40 smoke and P51A training.
set -euo pipefail

ROOT=/fs1/private/user/baitongyuan/projects/liuzh
OUT=$ROOT/outputs/p51a_bernini_r13b_single_scene
REPO=$ROOT/repos/Bernini
VENV=$REPO/.venv/bin/python
PREPROCESS_PID=${1:?usage: p51a_after_preprocess_and_launch.sh <preprocess-pid>}
LOGDIR=$OUT/logs
SMOKE_RUN=$OUT/smoke/training_4gpu_step1
FINAL=$OUT/final

export PYTHONPATH="$REPO:$ROOT/repos/VeOmni-v0.1.11${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
mkdir -p "$LOGDIR" "$FINAL"

fail() {
  printf '# P51A fail-closed\n\n%s\n' "$1" > "$FINAL/P51A_FAIL_CLOSED.md"
  exit 1
}

while kill -0 "$PREPROCESS_PID" 2>/dev/null; do
  sleep 30
done

PREPROCESSED=$OUT/preprocessed/training_source.parquet
[[ -f "$PREPROCESSED" ]] || fail "Full72 official preprocessing process ended without $PREPROCESSED. See logs/preprocess_full72.log."
"$VENV" - <<'PY' || fail "Full72 preprocessing contract check failed; no training was started."
import io
import json
import pandas as pd
import torch

p = "/fs1/private/user/baitongyuan/projects/liuzh/outputs/p51a_bernini_r13b_single_scene/preprocessed/training_source.parquet"
df = pd.read_parquet(p)
expected = {f"cyclic_start{i:02d}" for i in range(72)}
assert len(df) == 72, len(df)
assert set(df.sample_id) == expected
for _, row in df.iterrows():
    assert len(row.videos) == len(row.video_embeds) == len(row.video_grid_thw) == len(row.video_vae_latents) == 2
    assert row.input_frame_indices.tolist() == row.target_frame_indices.tolist()
for idx in (0, len(df) - 1):
    row = df.iloc[idx]
    latents = [torch.load(io.BytesIO(x), map_location="cpu", weights_only=True) for x in row.video_vae_latents]
    assert all(tuple(x.shape) == (1, 32, 9, 84, 84) for x in latents)
report = {"status": "pass", "rows": 72, "paired_videos_per_row": 2, "processed_resolution": [672, 672], "latent_shape": [1, 32, 9, 84, 84]}
open("/fs1/private/user/baitongyuan/projects/liuzh/outputs/p51a_bernini_r13b_single_scene/smoke/preprocess_full72_contract.json", "w").write(json.dumps(report, indent=2) + "\n")
PY

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | awk 'NF {print}')
[[ -z "$busy" ]] || fail "GPU allocation is no longer empty after preprocessing; refusing to start a four-GPU P51A run. Busy PIDs: $busy"
[[ ! -e "$SMOKE_RUN/checkpoints" ]] || fail "Refusing to overwrite existing four-GPU smoke checkpoint directory: $SMOKE_RUN/checkpoints"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$REPO/.venv/bin/torchrun" \
  --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29512 \
  "$OUT/scripts/train_bernini_renderer_p51a.py" "$OUT/configs/bernini_r13b_single_scene_4gpu.yaml" \
  --train.max_steps 1 \
  --train.checkpoint.output_dir "$SMOKE_RUN" \
  --train.checkpoint.save_steps 1 \
  --train.checkpoint.hf_save_steps 1 \
  >"$LOGDIR/train_smoke_4gpu_step1.log" 2>&1 \
  || fail "Four-GPU forward/backward/checkpoint smoke failed. See logs/train_smoke_4gpu_step1.log."

SMOKE_HF=$SMOKE_RUN/checkpoints/global_step_1/hf_ckpt
[[ -f "$SMOKE_HF/P51A_HF_EXPORT_COMPLETE.json" ]] || fail "Four-GPU smoke did not create a complete transformer safetensors export."
CUDA_VISIBLE_DEVICES=0 "$VENV" "$OUT/scripts/p51a_validate_checkpoint.py" \
  --base "$ROOT/models/Bernini-R-1.3B-Diffusers" \
  --checkpoint "$SMOKE_HF" \
  --output-dir "$OUT/smoke/validation_step0001" \
  --adjusted-clips "$OUT/clips/adjusted" \
  --clean-clips "$OUT/clips/clean" \
  --comparison-script "$OUT/scripts/p51a_make_comparisons.sh" \
  >"$LOGDIR/validate_smoke_step0001.log" 2>&1 \
  || fail "Checkpoint inference/metric smoke failed. See logs/validate_smoke_step0001.log."
[[ -f "$OUT/smoke/validation_step0001/P51A_VALIDATION_COMPLETE.json" ]] || fail "Checkpoint inference smoke ended without a completion marker."

[[ ! -e "$OUT/training/run_700/checkpoints" ]] || fail "Refusing to overwrite an existing P51A 700-step run."
# The durable controller handles exactly one <=20-step GPU unit per process;
# cron re-invokes its persistent guard every five minutes after normal exits or
# the site's periodic SIGKILL.  Launch the first unit now, then release this
# smoke supervisor instead of leaving a long-lived training process behind.
nohup "$OUT/scripts/p51a_persistent_guard.sh" >"$LOGDIR/train_validate_cycles_supervisor.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$OUT/training/P51A_TRAINING_ORCHESTRATOR.pid"
printf 'status=launched\npid=%s\n' "$!" > "$OUT/training/P51A_TRAINING_ORCHESTRATOR.status"
