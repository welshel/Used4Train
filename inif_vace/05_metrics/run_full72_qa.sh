#!/usr/bin/env bash
set -euo pipefail
ROOT="${P39_REPRO_ROOT:-/root/autodl-tmp/outputs/p39_single_scene_quality_push/09_reproducible_code}"
PY="${PYTHON:-/root/autodl-tmp/envs/p37_blackwell/bin/python}"
PAIR="${PAIR_MANIFEST:-/root/autodl-tmp/outputs/p32_newscene_condition_pair_audit/final/metadata/pair_manifest.jsonl}"
OUT="${FULL72_OUT:-$ROOT/full72/direct73}"
QA_OUT="${QA_OUT:-$ROOT/full72/direct73_qa}"
WIDTH="${WIDTH:-704}"
HEIGHT="${HEIGHT:-704}"
VIDEO="${OUTPUT_VIDEO:-$OUT/P39_FULL72_DIRECT_RAW.mp4}"
CONDITION="${CONDITION_VIDEO:-$OUT/P39_FULL72_CONDITION.mp4}"
[[ -f "$VIDEO" && -f "$CONDITION" && -f "$PAIR" ]] || { echo "missing full72 video, condition video, or pair manifest" >&2; exit 2; }
if [[ "${ALLOW_OVERWRITE:-0}" != 1 && -e "$QA_OUT" ]]; then echo "refusing to overwrite $QA_OUT; set ALLOW_OVERWRITE=1" >&2; exit 3; fi
mkdir -p "$QA_OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT/05_metrics"
exec "$PY" "$ROOT/05_metrics/p39_full72_qa.py" \
  --output-video "$VIDEO" --condition-video "$CONDITION" \
  --pair-manifest "$PAIR" --output-dir "$QA_OUT" \
  --method direct73 --width "$WIDTH" --height "$HEIGHT" --fps 12 \
  --model-boundary 71 0
