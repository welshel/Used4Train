#!/usr/bin/env bash
set -euo pipefail
ROOT="${P39_REPRO_ROOT:-/root/autodl-tmp/outputs/p39_single_scene_quality_push/09_reproducible_code}"
PY="${PYTHON:-/root/autodl-tmp/envs/p37_blackwell/bin/python}"
PAIR="${PAIR_MANIFEST:-$ROOT/00_topdown_assets/reproduced/pair_manifest.jsonl}"
OUT="${DATASET_OUT:-$ROOT/datasets/dataset_33f_704}"
WIDTH="${WIDTH:-704}"
HEIGHT="${HEIGHT:-704}"
FPS="${ASSUMED_FPS:-12}"

if [[ ! -f "$PAIR" ]]; then
  echo "missing pair manifest: $PAIR" >&2
  exit 2
fi
mkdir -p "$OUT"
export PYTHONPATH="$ROOT/02_pair_dataset"
"$PY" "$ROOT/02_pair_dataset/p38_dataset_builder.py" \
  --pair-manifest "$PAIR" --output-dir "$OUT" \
  --clip-length 33 --width "$WIDTH" --height "$HEIGHT" --assumed-fps "$FPS"
