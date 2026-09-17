#!/usr/bin/env bash
set -euo pipefail
ROOT="${P39_REPRO_ROOT:-/root/autodl-tmp/outputs/p39_single_scene_quality_push/09_reproducible_code}"
PY="${PYTHON:-/root/autodl-tmp/envs/p37_blackwell/bin/python}"
REPO="${DIFFSYNTH_REPO:-/root/autodl-tmp/DiffSynth-Studio}"
MODEL="${BASE_MODEL:-/root/autodl-tmp/VACE/models/Wan2.1-VACE-1.3B}"
LORA="${FULL72_LORA:-/root/autodl-tmp/outputs/p39_single_scene_quality_push/checkpoints/P39_H672_STEP0925_BEST.safetensors}"
PAIR="${PAIR_MANIFEST:-/root/autodl-tmp/outputs/p32_newscene_condition_pair_audit/final/metadata/pair_manifest.jsonl}"
COND_MANIFEST="${CONDITION_MANIFEST:-$ROOT/full72/condition_manifest.jsonl}"
OUT="${FULL72_OUT:-$ROOT/full72/direct73}"
WIDTH="${WIDTH:-704}"
HEIGHT="${HEIGHT:-704}"
if [[ "${ALLOW_OVERWRITE:-0}" != 1 && -e "$OUT" ]]; then echo "refusing to overwrite $OUT; set ALLOW_OVERWRITE=1" >&2; exit 3; fi
[[ -f "$LORA" && -f "$PAIR" ]] || { echo "missing LoRA or pair manifest" >&2; exit 2; }
mkdir -p "$(dirname "$COND_MANIFEST")" "$OUT"
if [[ ! -f "$COND_MANIFEST" ]]; then
  "$PY" "$ROOT/01_infinisplat/make_condition_only_manifest.py" --pair-manifest "$PAIR" --output "$COND_MANIFEST"
fi
cd "$ROOT"
export PYTHONPATH="$ROOT/05_metrics:$ROOT/02_pair_dataset"
exec "$PY" "$ROOT/05_metrics/p39_full72.py" \
  --diffsynth-repo "$REPO" --model-dir "$MODEL" --condition-manifest "$COND_MANIFEST" \
  --lora "$LORA" --output-dir "$OUT" --method direct73 \
  --width "$WIDTH" --height "$HEIGHT" --seed 20260911 \
  --num-inference-steps 20 --cfg-scale 5.0 --vace-scale 1.0 --fps 12
