#!/usr/bin/env bash
# Lightweight progress telemetry.  It never touches model/checkpoint state.
set -euo pipefail

ROOT=/fs1/private/user/baitongyuan/projects/liuzh
OUT=$ROOT/outputs/p51a_bernini_r13b_single_scene
RUN=$OUT/training/run_700
VENV=$ROOT/repos/Bernini/.venv/bin/python
mkdir -p "$RUN" "$OUT/logs"

"$VENV" - "$RUN" "$OUT" <<'PY'
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

run, out = map(Path, sys.argv[1:3])
steps = []
for marker in sorted((run / "checkpoints").glob("global_step_*/P51A_DCP_COMPLETE.json")):
    try:
        payload = json.loads(marker.read_text())
        if payload.get("status") == "complete":
            steps.append(int(payload["global_step"]))
    except (OSError, ValueError, KeyError):
        pass
validations = []
for marker in sorted((out / "validation").glob("step*/P51A_VALIDATION_COMPLETE.json")):
    try:
        metrics_path = marker.parent / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        validations.append({
            "step": int(marker.parent.name.removeprefix("step")),
            "metrics": {
                "prediction_vs_clean": metrics["mean_prediction_vs_clean"],
                "condition_vs_clean": metrics["mean_condition_vs_clean"],
            },
            "prediction_videos": sorted(path.name for path in marker.parent.glob("start*.mp4")),
            "comparison_videos": sorted(path.name for path in marker.parent.glob("comparison_start*.mp4")),
            "clean_used_as_inference_input": metrics["inference"]["clean_used_as_inference_input"],
        })
    except (OSError, ValueError, KeyError):
        pass
status_path = run / "P51A_ORCHESTRATOR_STATUS.json"
try:
    orchestrator = json.loads(status_path.read_text())
except (OSError, ValueError):
    orchestrator = {"state": "unavailable_or_invalid"}
try:
    gpu_lines = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
except subprocess.CalledProcessError as error:
    gpu_lines = [f"nvidia-smi-error: {error}"]
try:
    processes = subprocess.check_output(
        ["pgrep", "-af", "train_bernini_renderer_p51a.py|p51a_validate_checkpoint.py"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).splitlines()
except subprocess.CalledProcessError:
    processes = []
latest_log = None
logs = sorted((out / "logs").glob("train_step*.log"), key=lambda path: path.stat().st_mtime)
if logs:
    latest_log = logs[-1]
    try:
        text = latest_log.read_text(errors="replace")[-12000:]
        progress = re.findall(r"(\d+)%\|.*?loss:\s*([0-9.]+)", text)
    except OSError:
        progress = []
else:
    progress = []
payload = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "latest_complete_step": max(steps) if steps else 0,
    "complete_steps": steps,
    "latest_validation": validations[-1] if validations else None,
    "completed_validations": validations,
    "orchestrator": orchestrator,
    "latest_train_log": str(latest_log) if latest_log else None,
    "latest_progress": {"percent": int(progress[-1][0]), "loss": float(progress[-1][1])} if progress else None,
    "gpu_csv_nounits": gpu_lines,
    "active_processes": processes,
}
target = run / "P51A_HEARTBEAT.json"
temporary = target.with_suffix(".json.tmp")
with open(temporary, "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(temporary, target)
with open(out / "logs" / "p51a_heartbeat.jsonl", "a") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")
PY
