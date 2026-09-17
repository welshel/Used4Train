#!/usr/bin/env bash
set -euo pipefail

# Set P39_REPRO_ROOT to the directory containing this package.  The defaults
# point at the audited server assets and never overwrite the original P32/P39.
ROOT="${P39_REPRO_ROOT:-/root/autodl-tmp/outputs/p39_single_scene_quality_push/09_reproducible_code}"
PY="${PYTHON:-/root/autodl-tmp/envs/p37_blackwell/bin/python}"
INF="${INFINISPLAT_REPO:-/root/autodl-tmp/InfiniSplat}"
TOP="${TOPDOWN:-/root/autodl-tmp/clean_dataset_327431980_4/327431980_4_normalized/topdown_normalized/topdown.png}"
TOP_CAM="${TOPDOWN_CAMERA:-/root/autodl-tmp/clean_dataset_327431980_4/327431980_4_normalized/topdown_normalized/camera_para.json}"
CLEAN="${CLEAN_TRAJECTORY:-/root/autodl-tmp/clean_dataset_327431980_4/327431980_4_normalized/video_center_72}"
CKPT="${INFINISPLAT_CHECKPOINT:-/root/autodl-tmp/InfiniSplat/checkpoints/infinisplat_rgb.ckpt}"
PAIR_AUDIT="${PAIR_AUDIT_ROOT:-/root/autodl-tmp/outputs/p32_newscene_condition_pair_audit}"
SCALE_AUDIT="${SCALE_AUDIT:-$PAIR_AUDIT/05_source_canonical_transfer/scale_audit.json}"
OUT="$ROOT/00_topdown_assets/reproduced"
ART="$OUT/gaussians.pt"
PLY="$OUT/scene.ply"
PAYLOAD="$OUT/condition_camera_payload.json"
PAIR="$OUT/pair_manifest.jsonl"
CONDITION_MANIFEST="$OUT/condition_manifest.jsonl"

mkdir -p "$OUT"
"$PY" "$ROOT/01_infinisplat/infer_topdown.py" \
  --infinisplat-repo "$INF" --checkpoint "$CKPT" --topdown "$TOP" \
  --topdown-camera "$TOP_CAM" \
  --artifact "$ART" --ply "$PLY" --meta "$OUT/inference_meta.json"
"$PY" "$ROOT/01_infinisplat/build_condition_cameras.py" \
  --pair-manifest "$PAIR_AUDIT/final/metadata/pair_manifest.jsonl" \
  --source-camera "$TOP_CAM" --scale-audit "$SCALE_AUDIT" --output "$PAYLOAD"
"$PY" "$ROOT/01_infinisplat/render_condition_72.py" \
  --infinisplat-repo "$INF" --artifact "$ART" --camera-payload "$PAYLOAD" \
  --output-dir "$OUT/rendered" --video "$OUT/condition_72.mp4"
"$PY" "$ROOT/01_infinisplat/make_pair_manifest.py" \
  --condition-dir "$OUT/rendered/rgb" --clean-dir "$CLEAN" \
  --camera-payload "$PAYLOAD" --output "$PAIR" \
  --condition-only-output "$CONDITION_MANIFEST"
"$PY" "$ROOT/01_infinisplat/validate_condition_pair.py" \
  --pair-manifest "$PAIR" --topdown "$TOP" --condition-dir "$OUT/rendered/rgb" \
  --clean-dir "$CLEAN" --report "$OUT/pair_validation.json"
