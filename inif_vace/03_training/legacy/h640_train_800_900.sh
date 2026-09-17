#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/outputs/p38_high_quality_roomtour
OUT="$ROOT/02_640_training/H640_step800_to900"
PY=/root/autodl-tmp/envs/p37_blackwell/bin/python
TRAIN="$ROOT/01_code/scripts/p38_train.py"
MANIFEST="$ROOT/00_handoff/dataset_33f_640/training_manifest.jsonl"
MODEL=/root/autodl-tmp/VACE/models/Wan2.1-VACE-1.3B
REPO=/root/autodl-tmp/DiffSynth-Studio
LORA=/root/autodl-tmp/outputs/p37_wan_33f_temporal_persistence/pro6000_resume/11_highspeed_production/run_step600_to800/checkpoints/optimizer_step-0800.safetensors
OPT=/root/autodl-tmp/outputs/p37_wan_33f_temporal_persistence/pro6000_resume/11_highspeed_production/run_step600_to800/checkpoints/optimizer_state_step-0800.pt

mkdir -p "$OUT"
cd "$ROOT/01_code"
export PYTHONPATH="$ROOT/01_code/scripts"

python3 - "$OUT" "$LORA" "$OPT" <<'PY'
import datetime
import hashlib
import json
import sys
from pathlib import Path

out, lora, optimizer = map(Path, sys.argv[1:])
sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
provenance = {
    "timestamp": datetime.datetime.now().astimezone().isoformat(),
    "stage": "P38_H640_800_TO900",
    "source_checkpoint": str(lora),
    "source_checkpoint_sha256": sha256(lora),
    "source_optimizer": str(optimizer),
    "source_optimizer_sha256": sha256(optimizer),
    "warm_start": True,
    "resume_mode": "WEIGHT_AND_OPTIMIZER_RESUME",
    "source_optimizer_step": 800,
    "target_optimizer_step": 900,
    "additional_steps": 100,
    "resolution": [640, 640],
    "frames": 33,
    "batch_size": 1,
    "gradient_accumulation": 4,
    "effective_batch": 4,
    "gradient_checkpointing": False,
    "offload": False,
    "learning_rate": 0.0001,
    "structural_supervision": False,
    "inference_clean_inputs": False,
    "benchmark_weights_reused": False,
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

set +e
"$PY" "$TRAIN" \
  --diffsynth-repo "$REPO" --model-dir "$MODEL" --manifest "$MANIFEST" \
  --output-dir "$OUT/run" --stage P38_H640_800_TO900 \
  --start-step 800 --end-step 900 --checkpoint-steps 900 \
  --resume-lora "$LORA" --resume-optimizer "$OPT" \
  --rank 16 --learning-rate 1e-4 --weight-decay 1e-2 \
  --batch-size 1 --gradient-accumulation 4 --num-workers 0 \
  --width 640 --height 640 --num-frames 33 --seed 20260911 \
  --no-gradient-checkpointing > "$OUT/train.log" 2>&1
RC=$?
set -e

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
printf '%s\n' "$RC" > "$OUT/exit_code.txt"
exit "$RC"
