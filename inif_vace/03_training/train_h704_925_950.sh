#!/usr/bin/env bash
set -euo pipefail
ROOT="${P39_REPRO_ROOT:-/root/autodl-tmp/outputs/p39_single_scene_quality_push/09_reproducible_code}"
PY="${PYTHON:-/root/autodl-tmp/envs/p37_blackwell/bin/python}"
REPO="${DIFFSYNTH_REPO:-/root/autodl-tmp/DiffSynth-Studio}"
MODEL="${BASE_MODEL:-/root/autodl-tmp/VACE/models/Wan2.1-VACE-1.3B}"
MANIFEST="${TRAIN_MANIFEST:-/root/autodl-tmp/outputs/p39_single_scene_quality_push/01_704_dataset/dataset_33f_704/training_manifest.jsonl}"
LORA="${SOURCE_LORA:-/root/autodl-tmp/outputs/p39_single_scene_quality_push/03_672_training/H672_step900_to950/run/checkpoints/optimizer_step-0925.safetensors}"
OPT="${SOURCE_OPTIMIZER:-/root/autodl-tmp/outputs/p39_single_scene_quality_push/03_672_training/H672_step900_to950/run/checkpoints/optimizer_state_step-0925.pt}"
OUT="${TRAIN_OUT:-$ROOT/runs/H704_step925_to950}"
[[ "${ALLOW_OVERWRITE:-0}" == 1 || ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT; set ALLOW_OVERWRITE=1" >&2; exit 3; }
[[ -f "$LORA" && -f "$OPT" && -f "$MANIFEST" ]] || { echo "missing source checkpoint/optimizer/manifest" >&2; exit 2; }
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT/03_training:$ROOT/02_pair_dataset"
exec "$PY" "$ROOT/03_training/p38_train.py" \
  --diffsynth-repo "$REPO" --model-dir "$MODEL" --manifest "$MANIFEST" \
  --output-dir "$OUT/run" --stage P39_H704_925_TO950 \
  --start-step 925 --end-step 950 --checkpoint-steps 937,950 \
  --resume-lora "$LORA" --resume-optimizer "$OPT" \
  --rank 16 --learning-rate 1e-4 --weight-decay 1e-2 \
  --batch-size 1 --gradient-accumulation 4 --num-workers 0 \
  --width 704 --height 704 --num-frames 33 --seed 20260911 \
  --no-gradient-checkpointing
