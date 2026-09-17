#!/usr/bin/env bash
set -euo pipefail

ROOT=/fs1/private/user/baitongyuan/projects/liuzh
OUT="$ROOT/outputs/p51a_bernini_r13b_single_scene"
UV_PID_FILE="$OUT/logs/uv_sync_without_veomni.pid"

uv_pid=$(cat "$UV_PID_FILE")
while kill -0 "$uv_pid" 2>/dev/null; do
  sleep 30
done

cd "$ROOT/repos/Bernini"
"$ROOT/tools/uv/uv" pip install --no-deps "$ROOT/repos/VeOmni-v0.1.11"

.venv/bin/python - <<'PY'
import torch
import veomni

print("torch", torch.__version__, torch.version.cuda)
print("veomni", getattr(veomni, "__version__", "unknown"))
PY
