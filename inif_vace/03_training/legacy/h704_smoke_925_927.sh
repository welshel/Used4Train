#!/usr/bin/env bash
set -euo pipefail

P39=/root/autodl-tmp/outputs/p39_single_scene_quality_push
OUT="$P39/02_704_runtime/B1_ACC4_NO_GC_NO_OFFLOAD"
PY=/root/autodl-tmp/envs/p37_blackwell/bin/python
TRAIN="$P39/01_code/scripts/p38_train.py"
MANIFEST="$P39/01_704_dataset/dataset_33f_704/training_manifest.jsonl"
MODEL=/root/autodl-tmp/VACE/models/Wan2.1-VACE-1.3B
REPO=/root/autodl-tmp/DiffSynth-Studio
LORA="$P39/03_672_training/H672_step900_to950/run/checkpoints/optimizer_step-0925.safetensors"
OPT="$P39/03_672_training/H672_step900_to950/run/checkpoints/optimizer_state_step-0925.pt"

rm -rf "$OUT"
mkdir -p "$OUT"

(
  while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
    sleep 1
  done
) > "$OUT/nvidia_smi_samples.csv" 2>&1 &
MONITOR_PID=$!

cd "$P39/01_code"
export PYTHONPATH="$P39/01_code/scripts"
set +e
"$PY" "$TRAIN" \
  --diffsynth-repo "$REPO" \
  --model-dir "$MODEL" \
  --manifest "$MANIFEST" \
  --output-dir "$OUT/run" \
  --stage P39_H704_SMOKE \
  --start-step 925 \
  --end-step 927 \
  --checkpoint-steps 927 \
  --resume-lora "$LORA" \
  --resume-optimizer "$OPT" \
  --rank 16 \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --num-workers 0 \
  --width 704 \
  --height 704 \
  --num-frames 33 \
  --seed 20260911 \
  --no-gradient-checkpointing \
  > "$OUT/train.log" 2>&1
RC=$?
set -e

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
printf '%s\n' "$RC" > "$OUT/exit_code.txt"
exit "$RC"
