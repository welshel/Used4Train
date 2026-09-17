#!/usr/bin/env bash
# Cron entrypoint for P51A.  It is deliberately short-lived: the controller
# owns one bounded unit and exits.  Any SIGKILL releases its flock, so the next
# five-minute cron tick resumes only from a verified checkpoint.
set -euo pipefail

ROOT=/fs1/private/user/baitongyuan/projects/liuzh
OUT=$ROOT/outputs/p51a_bernini_r13b_single_scene
RUN=$OUT/training/run_700
LOGDIR=$OUT/logs
mkdir -p "$RUN" "$LOGDIR"

# Do not contend with the required single-GPU inference smoke.  The smoke
# supervisor starts the first durable unit itself immediately after completion.
[[ -f "$OUT/smoke/validation_step0001/P51A_VALIDATION_COMPLETE.json" ]] || exit 0
[[ -f "$OUT/final/P51A_FAIL_CLOSED.md" ]] && exit 0
# A controller records every semantic failure atomically in this state file.
# Treat it as a durable fail-closed latch so cron never retries a suspect unit.
if [[ -f "$RUN/P51A_ORCHESTRATOR_STATUS.json" ]] && grep -qE "\"state\"[[:space:]]*:[[:space:]]*\"fail_closed\"" "$RUN/P51A_ORCHESTRATOR_STATUS.json"; then
  echo "[$(date -Is)] P51A guard: fail-closed orchestrator state"
  exit 0
fi
[[ -f "$OUT/final/P51A_FINAL_COMPLETE.json" ]] && exit 0
# The smoke supervisor launches the first controller unit itself.  Waiting for
# it to exit avoids a narrow race where cron could create ``checkpoints/`` just
# before the supervisor performs its no-overwrite preflight.
pgrep -f "p51a_after_preprocess_and_launch.sh" >/dev/null && exit 0

# One durable supervisor owns the whole remaining run.  The inner controller
# still publishes every 20-step DCP atomically; if the scheduler kills this
# supervisor, the next cron invocation acquires the released flock and resumes
# solely from the most recent verified checkpoint / validation shard.
exec 9>"$RUN/P51A_CONTINUOUS_RUNNER.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] another P51A continuous supervisor is active; nothing to do"
  exit 0
fi

while true; do
  [[ -f "$OUT/final/P51A_FINAL_COMPLETE.json" ]] && exit 0
  if [[ -f "$RUN/P51A_TRAINING_COMPLETE" ]]; then
    exec "$OUT/scripts/p51a_posttrain_controller.sh"
  fi
  echo "[$(date -Is)] P51A continuous supervisor: running next atomic unit"
  "$OUT/scripts/p51a_train_validate_cycles.sh"
done
