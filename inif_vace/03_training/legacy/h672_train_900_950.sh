#!/usr/bin/env bash
set -euo pipefail

P38=/root/autodl-tmp/outputs/p38_high_quality_roomtour
P39=/root/autodl-tmp/outputs/p39_single_scene_quality_push
OUT="$P39/03_672_training/H672_step900_to950"
PY=/root/autodl-tmp/envs/p37_blackwell/bin/python
TRAIN="$P39/01_code/scripts/p38_train.py"
MANIFEST="$P39/01_672_dataset/dataset_33f_672/training_manifest.jsonl"
MODEL=/root/autodl-tmp/VACE/models/Wan2.1-VACE-1.3B
REPO=/root/autodl-tmp/DiffSynth-Studio
LORA="$P38/checkpoints/P38_H640_STEP0900_BEST.safetensors"
OPT="$P38/checkpoints/P38_H640_STEP0900_OPTIMIZER.pt"

rm -rf "$OUT"
mkdir -p "$OUT"

"$PY" - "$OUT" "$LORA" "$OPT" <<'PY'
import datetime
import hashlib
import json
import sys
from pathlib import Path

out, lora, optimizer = map(Path, sys.argv[1:])
sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
provenance = {
    "timestamp": datetime.datetime.now().astimezone().isoformat(),
    "stage": "P39_H672_900_TO950",
    "source_checkpoint": str(lora),
    "source_checkpoint_sha256": sha256(lora),
    "source_optimizer": str(optimizer),
    "source_optimizer_sha256": sha256(optimizer),
    "warm_start": True,
    "resume_mode": "WEIGHT_AND_OPTIMIZER_RESUME",
    "source_optimizer_step": 900,
    "target_optimizer_step": 950,
    "additional_steps": 50,
    "resolution": [672, 672],
    "frames": 33,
    "batch_size": 1,
    "gradient_accumulation": 4,
    "effective_batch": 4,
    "gradient_checkpointing": False,
    "offload": False,
    "precision": "bf16",
    "learning_rate": 0.0001,
    "structural_supervision": False,
    "inference_clean_inputs": False,
    "smoke_weights_reused": False,
}
(out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
PY

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
  --diffsynth-repo "$REPO" --model-dir "$MODEL" --manifest "$MANIFEST" \
  --output-dir "$OUT/run" --stage P39_H672_900_TO950 \
  --start-step 900 --end-step 950 --checkpoint-steps 925,950 \
  --resume-lora "$LORA" --resume-optimizer "$OPT" \
  --rank 16 --learning-rate 1e-4 --weight-decay 1e-2 \
  --batch-size 1 --gradient-accumulation 4 --num-workers 0 \
  --width 672 --height 672 --num-frames 33 --seed 20260911 \
  --no-gradient-checkpointing > "$OUT/train.log" 2>&1
RC=$?
set -e

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
printf '%s\n' "$RC" > "$OUT/exit_code.txt"
exit "$RC"
