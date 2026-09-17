#!/usr/bin/env bash
set -euo pipefail

P38=/root/autodl-tmp/outputs/p38_high_quality_roomtour
P39=/root/autodl-tmp/outputs/p39_single_scene_quality_push
PY=/root/autodl-tmp/envs/p37_blackwell/bin/python
VALIDATE="$P39/01_code/scripts/p38_validate.py"
REPO=/root/autodl-tmp/DiffSynth-Studio
MODEL=/root/autodl-tmp/VACE/models/Wan2.1-VACE-1.3B
MANIFEST="$P39/01_672_dataset/dataset_33f_672/validation_manifest.jsonl"
ROOT_OUT="$P39/04_quality_metrics/full4_start_validation"

mkdir -p "$ROOT_OUT"
cd "$P39/01_code"
export PYTHONPATH="$P39/01_code/scripts"

run_validation() {
  local label=$1
  local lora=$2
  local out=$3

  if [[ -f "$out/validation_${label}.json" ]]; then
    echo "SKIP completed: $label"
    return
  fi

  mkdir -p "$out"
  "$PY" "$VALIDATE" \
    --diffsynth-repo "$REPO" \
    --model-dir "$MODEL" \
    --manifest "$MANIFEST" \
    --output-dir "$out" \
    --label "$label" \
    --lora "$lora" \
    --only-starts 0,17,31,50 \
    --base-seed 1337 \
    --num-inference-steps 20 \
    --cfg-scale 5.0 \
    --vace-scale 1.0 \
    --fps 12 \
    2>&1 | tee "$out/inference.log"
}

run_validation \
  p38_h640_900_infer672 \
  "$P38/checkpoints/P38_H640_STEP0900_BEST.safetensors" \
  "$ROOT_OUT/p38_infer672"

run_validation \
  h672_925 \
  "$P39/03_672_training/H672_step900_to950/run/checkpoints/optimizer_step-0925.safetensors" \
  "$ROOT_OUT/h672_925"

echo "P39 full-four-start validation complete."
