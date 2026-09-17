#!/usr/bin/env bash
set -euo pipefail
cd /fs1/private/user/baitongyuan/projects/liuzh
export TRANSFORMERS_CACHE="$PWD/cache/huggingface"
export HF_HOME="$PWD/cache/huggingface"
export TORCH_HOME="$PWD/cache/torch"
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=True

root="$PWD/outputs/p49a_tripo_vace13b_fresh"
train_dir="$root/05_training"
model="$PWD/models/Wan2.1-VACE-1.3B"
condition="$PWD/outputs/p48_3_tripo_dynamic_cleanup/06_full72/condition_rgb_dynamic"
target="$PWD/outputs/p48_tripo_clean_aligned/clean_rgb"
outroot="$root/07_start31_validation"
mkdir -p "$outroot"

# Wait for the segmented continuation supervisor, not stale segment statuses.
while [[ ! -f "$train_dir/training_supervisor_exit_status.txt" ]]; do sleep 20; done
status=$(tr -d '[:space:]' < "$train_dir/training_supervisor_exit_status.txt")
if [[ "$status" != "0" ]]; then
  echo "Training continuation failed with status $status; validation not started." >&2
  exit 20
fi
for step in 50 100 200 350 500 600 700; do
  test -s "$root/06_checkpoints/step-${step}.safetensors"
done

lines=(
  $'step0_base_start31\t31\t'
  $'step50_start31\t31\t'"$root/06_checkpoints/step-50.safetensors"
  $'step100_start31\t31\t'"$root/06_checkpoints/step-100.safetensors"
  $'step200_start31\t31\t'"$root/06_checkpoints/step-200.safetensors"
  $'step350_start31\t31\t'"$root/06_checkpoints/step-350.safetensors"
  $'step500_start31\t31\t'"$root/06_checkpoints/step-500.safetensors"
  $'step700_start31\t31\t'"$root/06_checkpoints/step-700.safetensors"
  $'step200_start0\t0\t'"$root/06_checkpoints/step-200.safetensors"
  $'step200_start17\t17\t'"$root/06_checkpoints/step-200.safetensors"
  $'step200_start50\t50\t'"$root/06_checkpoints/step-200.safetensors"
  $'step350_start0\t0\t'"$root/06_checkpoints/step-350.safetensors"
  $'step350_start17\t17\t'"$root/06_checkpoints/step-350.safetensors"
  $'step350_start50\t50\t'"$root/06_checkpoints/step-350.safetensors"
  $'step500_start0\t0\t'"$root/06_checkpoints/step-500.safetensors"
  $'step500_start17\t17\t'"$root/06_checkpoints/step-500.safetensors"
  $'step500_start50\t50\t'"$root/06_checkpoints/step-500.safetensors"
  $'step700_start0\t0\t'"$root/06_checkpoints/step-700.safetensors"
  $'step700_start17\t17\t'"$root/06_checkpoints/step-700.safetensors"
  $'step700_start50\t50\t'"$root/06_checkpoints/step-700.safetensors"
)

# Avoid stealing a heavily occupied shared GPU. Training has already satisfied
# the 4-GPU requirement; validation can safely use the remaining devices.
gpu0_foreign_mib=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i 0 2>/dev/null | awk '{s+=$2} END{print s+0}')
if (( gpu0_foreign_mib > 1000 )); then gpu_list=(1 2 3); else gpu_list=(0 1 2 3); fi
printf '%s\n' "${gpu_list[@]}" > "$outroot/P49A_VALIDATION_GPUS.txt"

run_one() {
  local gpu="$1" name="$2" start="$3" checkpoint="$4"
  local out="$outroot/$name"
  mkdir -p "$out"
  [[ -s "$out/metrics.json" && -s "$out/validation.mp4" ]] && return 0
  local args=(--model-dir "$model" --condition-dir "$condition" --target-dir "$target" --output-dir "$out" --start "$start" --num-frames 33 --resolution 672 --inference-steps 20 --cfg-scale 5 --seed 20260915)
  [[ -n "$checkpoint" ]] && args+=(--checkpoint "$checkpoint")
  CUDA_VISIBLE_DEVICES="$gpu" envs/p49a_vace13b/bin/python scripts/p49a_validate.py "${args[@]}" > "$out/run.log" 2>&1
}
export -f run_one
export model condition target outroot

pids=()
for worker in "${!gpu_list[@]}"; do
  gpu=${gpu_list[$worker]}
  (
    for ((i=worker; i<${#lines[@]}; i+=${#gpu_list[@]})); do
      IFS=$'\t' read -r name start checkpoint <<< "${lines[$i]}"
      run_one "$gpu" "$name" "$start" "${checkpoint:-}"
    done
  ) > "$outroot/worker_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
date -Is > "$outroot/P49A_VALIDATIONS_FINISHED_AT.txt"
printf '%s\n' "$rc" > "$outroot/P49A_VALIDATIONS_EXIT_STATUS.txt"
exit "$rc"
