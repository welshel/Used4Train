#!/usr/bin/env bash
set -euo pipefail
ROOT="${P39_REPRO_ROOT:-/root/autodl-tmp/outputs/p39_single_scene_quality_push/09_reproducible_code}"
PY="${PYTHON:-/root/autodl-tmp/envs/p37_blackwell/bin/python}"
REPO="${DIFFSYNTH_REPO:-/root/autodl-tmp/DiffSynth-Studio}"
MODEL="${BASE_MODEL:-/root/autodl-tmp/VACE/models/Wan2.1-VACE-1.3B}"
MANIFEST="${VALIDATION_MANIFEST:-/root/autodl-tmp/outputs/p39_single_scene_quality_push/01_672_dataset/dataset_33f_672/validation_manifest.jsonl}"
LORA="${VALIDATION_LORA:-/root/autodl-tmp/outputs/p39_single_scene_quality_push/03_672_training/H672_step900_to950/run/checkpoints/optimizer_step-0925.safetensors}"
OUT="${VALIDATION_OUT:-$ROOT/validation/full4_672}"
[[ -f "$MANIFEST" && -f "$LORA" ]] || { echo "missing validation manifest or LoRA" >&2; exit 2; }
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT/04_validation"
exec "$PY" "$ROOT/04_validation/p38_validate.py" \
  --diffsynth-repo "$REPO" --model-dir "$MODEL" --manifest "$MANIFEST" \
  --output-dir "$OUT" --label h672_925 --lora "$LORA" \
  --only-starts 0,17,31,50 --base-seed 1337 --num-inference-steps 20 \
  --cfg-scale 5.0 --vace-scale 1.0 --fps 12
