#!/usr/bin/env bash
# P51A durable one-unit controller.
#
# Each invocation performs exactly one bounded unit: a 20-step four-GPU
# training segment *or* one four-start validation.  This keeps every GPU
# process comfortably below the cluster's ~30-minute termination window.
# A five-minute cron guard invokes this file again after a normal exit or an
# interruption.  Resume is accepted only from an atomically published DCP
# completion marker; partial checkpoints/validations are archived, never
# overwritten or resumed.
set -euo pipefail

ROOT=/fs1/private/user/baitongyuan/projects/liuzh
OUT=$ROOT/outputs/p51a_bernini_r13b_single_scene
REPO=$ROOT/repos/Bernini
VENV=$REPO/.venv/bin/python
DEFAULT_CONFIG=$OUT/configs/bernini_r13b_single_scene_4gpu.yaml
BATCH2_CONFIG=$OUT/configs/bernini_r13b_single_scene_4gpu_batch2.yaml
TRAIN=$OUT/scripts/train_bernini_renderer_p51a.py
VALIDATE_SHARD=$OUT/scripts/p51a_validate_shard.py
RUN=$OUT/training/run_700
LOGDIR=$OUT/logs
TOTAL_STEPS=700
SEGMENT_STEPS=20

export PYTHONPATH="$REPO:$ROOT/repos/VeOmni-v0.1.11${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# With ulysses_size=4, let the official VeOmni argument parser select the
# sequence-parallel FA2 wrapper (veomni_flash_attention_2_with_sp).  The
# resume smoke from the verified step500 DCP passed on all four A40s.
export MODELING_BACKEND=veomni
mkdir -p "$LOGDIR" "$RUN/checkpoints" "$RUN/interrupted_checkpoints" "$OUT/validation/interrupted"

# ``flock`` is intentionally held for the whole bounded unit.  If the system
# kills a process, the kernel releases the lock and cron can safely recover.
exec 9>"$RUN/P51A_CONTROLLER.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] another P51A controller unit is active; nothing to do"
  exit 0
fi

atomic_status() {
  local state=$1 detail=$2
  local tmp="$RUN/P51A_ORCHESTRATOR_STATUS.json.tmp"
  "$VENV" - "$state" "$detail" <<'PY' >"$tmp"
import json, sys
from datetime import datetime, timezone
print(json.dumps({"state": sys.argv[1], "detail": sys.argv[2], "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True))
PY
  mv "$tmp" "$RUN/P51A_ORCHESTRATOR_STATUS.json"
}

# The initial 1-step smoke established correctness at batch 1.  Before the
# fresh 700-step run, make one isolated official batch-2 attempt to use the
# remaining A40 memory for useful throughput.  The attempt has its own output
# root and cannot contaminate/resume the training run.  An OOM deterministically
# selects the already-smoked batch-1 configuration; any other failure stops
# fail-closed rather than silently changing the training route.
THROUGHPUT_DECISION=$OUT/smoke/P51A_THROUGHPUT_DECISION.json
THROUGHPUT_SMOKE=$OUT/smoke/training_4gpu_batch2_step1
if [[ ! -f "$THROUGHPUT_DECISION" && -e "$THROUGHPUT_SMOKE/checkpoints" ]]; then
  if [[ -f "$THROUGHPUT_SMOKE/checkpoints/global_step_1/P51A_DCP_COMPLETE.json" ]]; then
    "$VENV" - "$THROUGHPUT_DECISION" <<'PY'
import json, os, sys
path = sys.argv[1]
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump({"status": "pass", "selected_config": "batch2", "global_batch_size": 2, "micro_batch_size": 2, "world_size": 4, "recovered_after_interruption": True}, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(tmp, path)
PY
  else
    archived="$OUT/smoke/interrupted_training_4gpu_batch2_step1_$(date +%Y%m%dT%H%M%S%z)"
    echo "[$(date -Is)] archiving interrupted batch-2 throughput smoke -> $archived"
    mv "$THROUGHPUT_SMOKE" "$archived"
  fi
fi
if [[ ! -f "$THROUGHPUT_DECISION" ]]; then
  atomic_status "throughput_smoke" "isolated 4xA40 batch-2 one-step test"
  echo "[$(date -Is)] throughput smoke: four GPUs, global/micro batch 2"
  set +e
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$REPO/.venv/bin/torchrun" --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29513 \
    "$TRAIN" "$BATCH2_CONFIG" \
    --train.max_steps 1 \
    --train.checkpoint.output_dir "$THROUGHPUT_SMOKE" \
    --train.checkpoint.save_steps 1 \
    --train.checkpoint.hf_save_steps 100 \
    >"$LOGDIR/train_throughput_smoke_batch2.log" 2>&1
  throughput_status=$?
  set -e
  if (( throughput_status == 0 )) && [[ -f "$THROUGHPUT_SMOKE/checkpoints/global_step_1/P51A_DCP_COMPLETE.json" ]]; then
    "$VENV" - "$THROUGHPUT_DECISION" <<'PY'
import json, os, sys
path = sys.argv[1]
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump({"status": "pass", "selected_config": "batch2", "global_batch_size": 2, "micro_batch_size": 2, "world_size": 4}, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(tmp, path)
PY
    atomic_status "throughput_smoke_pass" "selected batch2"
    exit 0
  fi
  if grep -qiE 'CUDA out of memory|OutOfMemoryError' "$LOGDIR/train_throughput_smoke_batch2.log"; then
    "$VENV" - "$THROUGHPUT_DECISION" <<'PY'
import json, os, sys
path = sys.argv[1]
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump({"status": "fallback", "selected_config": "batch1", "reason": "batch2_cuda_oom", "global_batch_size": 1, "micro_batch_size": 1, "world_size": 4}, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(tmp, path)
PY
    atomic_status "throughput_smoke_oom" "batch2 OOM; selected the already verified batch1 configuration"
    exit 0
  fi
  atomic_status "fail_closed" "batch2 throughput smoke failed for a non-OOM reason; see train_throughput_smoke_batch2.log"
  exit "$throughput_status"
fi

selected_config=$("$VENV" - "$THROUGHPUT_DECISION" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print(json.load(f)["selected_config"])
PY
)
case "$selected_config" in
  batch2) CONFIG=$BATCH2_CONFIG ;;
  batch1) CONFIG=$DEFAULT_CONFIG ;;
  *) echo "invalid throughput decision: $selected_config" >&2; exit 2 ;;
esac

is_complete_checkpoint() {
  local step=$1 marker="$RUN/checkpoints/global_step_$1/P51A_DCP_COMPLETE.json"
  [[ -f "$marker" ]] || return 1
  "$VENV" - "$marker" "$step" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    payload = json.load(f)
assert payload["status"] == "complete"
assert int(payload["global_step"]) == int(sys.argv[2])
PY
}

archive_incomplete_checkpoint() {
  local step=$1 source="$RUN/checkpoints/global_step_$1"
  [[ -e "$source" ]] || return 0
  is_complete_checkpoint "$step" && return 0
  local destination="$RUN/interrupted_checkpoints/global_step_${step}_$(date +%Y%m%dT%H%M%S%z)"
  echo "[$(date -Is)] archiving incomplete checkpoint $source -> $destination"
  mv "$source" "$destination"
}

archive_incomplete_validation() {
  local step=$1 source="$OUT/validation/step$(printf '%04d' "$step")"
  [[ -e "$source" ]] || return 0
  [[ -f "$source/P51A_VALIDATION_COMPLETE.json" ]] && return 0
  local destination="$OUT/validation/interrupted/step$(printf '%04d' "$step")_$(date +%Y%m%dT%H%M%S%z)"
  echo "[$(date -Is)] archiving incomplete validation $source -> $destination"
  mv "$source" "$destination"
}

last_complete=0
shopt -s nullglob
for marker in "$RUN"/checkpoints/global_step_*/P51A_DCP_COMPLETE.json; do
  checkpoint_dir=${marker%/P51A_DCP_COMPLETE.json}
  step=${checkpoint_dir##*global_step_}
  [[ $step =~ ^[0-9]+$ ]] || continue
  if is_complete_checkpoint "$step" && (( step > last_complete )); then
    last_complete=$step
  fi
done
shopt -u nullglob

# Validation is a hard barrier. A 40-step denoise for each fixed start takes
# about five minutes; a monolithic four-start validator can straddle the
# cluster's half-hour kill boundary. Each start is an independent atomic GPU
# shard, and the CPU finalizer publishes the global marker only after all four.
if (( last_complete > 0 && last_complete % 100 == 0 )); then
  validation="$OUT/validation/step$(printf '%04d' "$last_complete")"
  hf_checkpoint="$RUN/checkpoints/global_step_$last_complete/hf_ckpt"
  if [[ ! -f "$validation/P51A_VALIDATION_COMPLETE.json" ]]; then
    [[ -f "$hf_checkpoint/P51A_HF_EXPORT_COMPLETE.json" ]] || {
      atomic_status "fail_closed" "DCP step $last_complete is complete but its required HF export marker is absent"
      echo "missing complete HF export for validation step $last_complete" >&2
      exit 3
    }

    shard_is_complete() {
      local shard_start=$1
      local name="start$(printf '%02d' "$shard_start")"
      local marker="$validation/P51A_START_COMPLETE_$name.json"
      local metrics="$validation/$name.metrics.json"
      [[ -f "$marker" && -f "$metrics" && -f "$validation/$name.mp4" && -f "$validation/comparison_$name.mp4" ]] || return 1
      "$VENV" - "$marker" "$metrics" "$shard_start" <<'PY'
import json, sys
marker = json.load(open(sys.argv[1]))
metrics = json.load(open(sys.argv[2]))
assert marker["status"] == "complete"
assert int(marker["start"]) == int(sys.argv[3])
assert marker["clean_used_as_inference_input"] is False
assert metrics["start"] == int(sys.argv[3])
assert metrics["clean_used_as_inference_input"] is False
PY
    }

    # Old monolithic output has no shard markers and is not resumable. A killed
    # sharded attempt retains only starts that publish an atomic marker.
    has_shard=0
    for shard_start in 0 17 31 50; do
      if [[ -f "$validation/P51A_START_COMPLETE_start$(printf '%02d' "$shard_start").json" ]]; then
        has_shard=1
        break
      fi
    done
    if (( has_shard == 0 )); then
      archive_incomplete_validation "$last_complete"
    fi

    missing=()
    for shard_start in 0 17 31 50; do
      if ! shard_is_complete "$shard_start"; then
        missing+=("$shard_start")
      fi
    done

    if (( ${#missing[@]} == 0 )); then
      atomic_status "finalizing_validation" "step=$last_complete all fixed-start shards complete"
      echo "[$(date -Is)] validation finalizer: step $last_complete"
      CUDA_VISIBLE_DEVICES= "$VENV" "$VALIDATE_SHARD" \
        --base "$ROOT/models/Bernini-R-1.3B-Diffusers" \
        --checkpoint "$hf_checkpoint" \
        --output-dir "$validation" \
        --adjusted-clips "$OUT/clips/adjusted" \
        --clean-clips "$OUT/clips/clean" \
        --comparison-script "$OUT/scripts/p51a_make_comparisons.sh" \
        --finalize-only \
        >"$LOGDIR/validate_step$(printf '%04d' "$last_complete")_finalize.log" 2>&1
      [[ -f "$validation/P51A_VALIDATION_COMPLETE.json" ]] || {
        atomic_status "fail_closed" "validation finalizer exited without global completion marker at step $last_complete"
        exit 4
      }
      atomic_status "validation_complete" "step=$last_complete"
      exit 0
    fi

    # Run validation immediately.  Each fixed-start shard publishes its own
    # atomic marker, so a scheduler kill leaves completed starts intact and the
    # continuous supervisor resumes only the missing shards on the next launch.

    atomic_status "validating" "step=$last_complete parallel_missing_starts=${missing[*]} clean_for_qa_only=true"
    echo "[$(date -Is)] validation shards: step $last_complete starts=${missing[*]}"
    pids=()
    starts=()
    gpu=0
    for shard_start in "${missing[@]}"; do
      CUDA_VISIBLE_DEVICES="$gpu" "$VENV" "$VALIDATE_SHARD" \
        --base "$ROOT/models/Bernini-R-1.3B-Diffusers" \
        --checkpoint "$hf_checkpoint" \
        --output-dir "$validation" \
        --adjusted-clips "$OUT/clips/adjusted" \
        --clean-clips "$OUT/clips/clean" \
        --comparison-script "$OUT/scripts/p51a_make_comparisons.sh" \
        --start "$shard_start" \
        >"$LOGDIR/validate_step$(printf '%04d' "$last_complete")_start$(printf '%02d' "$shard_start").log" 2>&1 &
      pids+=("$!")
      starts+=("$shard_start")
      ((gpu+=1))
    done

    shard_failed=0
    for index in "${!pids[@]}"; do
      if ! wait "${pids[$index]}"; then
        echo "[$(date -Is)] validation shard failed: start${starts[$index]}" >&2
        shard_failed=1
      fi
    done
    (( shard_failed == 0 )) || {
      atomic_status "fail_closed" "one or more validation shards exited unsuccessfully at step $last_complete"
      exit 4
    }
    for shard_start in "${starts[@]}"; do
      shard_is_complete "$shard_start" || {
        atomic_status "fail_closed" "validation shard exited without a verified start marker: step=$last_complete start=$shard_start"
        exit 4
      }
    done

    CUDA_VISIBLE_DEVICES= "$VENV" "$VALIDATE_SHARD" \
      --base "$ROOT/models/Bernini-R-1.3B-Diffusers" \
      --checkpoint "$hf_checkpoint" \
      --output-dir "$validation" \
      --adjusted-clips "$OUT/clips/adjusted" \
      --clean-clips "$OUT/clips/clean" \
      --comparison-script "$OUT/scripts/p51a_make_comparisons.sh" \
      --finalize-only \
      >"$LOGDIR/validate_step$(printf '%04d' "$last_complete")_finalize.log" 2>&1
    [[ -f "$validation/P51A_VALIDATION_COMPLETE.json" ]] || {
      atomic_status "fail_closed" "all validation shards completed but global finalizer marker is absent at step $last_complete"
      exit 4
    }
    atomic_status "validation_complete" "step=$last_complete"
    exit 0
  fi
fi

if (( last_complete == TOTAL_STEPS )); then
  atomic_status "training_complete" "all 20-step DCP segments and all 100-step validations are complete"
  touch "$RUN/P51A_TRAINING_COMPLETE"
  echo "[$(date -Is)] all P51A training/validation units complete"
  exit 0
fi

next_step=$((last_complete + SEGMENT_STEPS))
if (( next_step > TOTAL_STEPS )); then
  next_step=$TOTAL_STEPS
fi
# Do not step over a mandatory every-100-step validation checkpoint.
next_validation_step=$((((last_complete / 100) + 1) * 100))
if (( next_validation_step <= TOTAL_STEPS && next_step > next_validation_step )); then
  next_step=$next_validation_step
fi
archive_incomplete_checkpoint "$next_step"
checkpoint="$RUN/checkpoints/global_step_$next_step"
[[ ! -e "$checkpoint" ]] || {
  atomic_status "fail_closed" "refusing to overwrite existing checkpoint path $checkpoint"
  exit 5
}

args=(
  --train.max_steps "$next_step"
  --train.accelerator.ulysses_size 4
  --train.checkpoint.output_dir "$RUN"
  --train.checkpoint.save_steps "$next_step"
  --train.checkpoint.hf_save_steps 100
)
if (( last_complete > 0 )); then
  args+=(--train.checkpoint.load_path "$RUN/checkpoints/global_step_$last_complete")
fi

atomic_status "training" "segment=${last_complete}->${next_step} world_size=4 ulysses_size=4"
echo "[$(date -Is)] training unit: step $last_complete -> $next_step"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  "$REPO/.venv/bin/torchrun" --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29511 \
  "$TRAIN" "$CONFIG" "${args[@]}" \
  >"$LOGDIR/train_step$(printf '%04d' "$next_step").log" 2>&1

is_complete_checkpoint "$next_step" || {
  atomic_status "fail_closed" "training segment reached process exit without verified DCP completion at step $next_step"
  exit 6
}
atomic_status "checkpoint_complete" "step=$next_step"
echo "[$(date -Is)] complete DCP checkpoint: $checkpoint"
