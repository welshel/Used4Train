#!/usr/bin/env bash
set -u -o pipefail
cd /fs1/private/user/baitongyuan/projects/liuzh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRANSFORMERS_CACHE="$PWD/cache/huggingface"
export HF_HOME="$PWD/cache/huggingface"
export TORCH_HOME="$PWD/cache/torch"
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

root="$PWD/outputs/p49a_tripo_vace13b_fresh"
train_dir="$root/05_training"
out="$root/06_checkpoints"
audit="$train_dir/P49A_SEGMENTED_RESUME_AUDIT.csv"
final_status="$train_dir/training_supervisor_exit_status.txt"
rm -f "$final_status"
if [[ ! -s "$audit" ]]; then
  printf 'started_at,start_step,finished_at,exit_status,latest_checkpoint,termination\n' > "$audit"
fi

latest_step() {
  find "$out" -maxdepth 1 -type f -name 'step-*.safetensors' -printf '%f\n' \
    | sed -E 's/step-([0-9]+)\.safetensors/\1/' | sort -n | tail -n 1
}

validate_checkpoint() {
  local checkpoint="$1"
  envs/p49a_vace13b/bin/python - "$checkpoint" <<'PY'
import sys
from safetensors import safe_open
path=sys.argv[1]
with safe_open(path, framework="pt", device="cpu") as f:
    keys=list(f.keys())
    assert len(keys)==300, len(keys)
    assert all(f.get_tensor(k).numel() for k in keys)
PY
}

wait_for_free_gpus() {
  while true; do
    mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    free=1
    for mib in "${used[@]}"; do
      (( mib >= 1000 )) && free=0
    done
    (( free == 1 )) && return 0
    printf '%s waiting_for_exclusive_gpus memory_used_mib=%s\n' "$(date -Is)" "${used[*]}" \
      >> "$train_dir/P49A_SEGMENTED_RESUME_WAIT.log"
    sleep 30
  done
}

(while true; do date -Is; nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader; sleep 5; done) \
  > "$train_dir/nvidia_smi_4gpu_segmented_resume.log" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

attempt=0
while true; do
  start=$(latest_step)
  if [[ -z "${start:-}" ]]; then
    printf '40\n' > "$final_status"
    exit 40
  fi
  if (( start >= 700 )); then
    printf '0\n' > "$final_status"
    exit 0
  fi
  attempt=$((attempt + 1))
  if (( attempt > 8 )); then
    printf '41\n' > "$final_status"
    exit 41
  fi
  checkpoint="$out/step-${start}.safetensors"
  validate_checkpoint "$checkpoint"
  wait_for_free_gpus
  if [[ -s "$out/loss.csv" ]]; then
    mv "$out/loss.csv" "$out/loss_before_resume_from_${start}_$(date +%Y%m%d_%H%M%S).csv"
  fi
  started=$(date -Is)
  log="$train_dir/train_4gpu_resume${start}.log"
  set +e
  "$PWD/envs/p49a_vace13b/bin/accelerate" launch --multi_gpu --num_processes 4 --mixed_precision no --main_process_port 0 scripts/p49a_train_resume.py \
    --dataset_base_path "$PWD" \
    --dataset_metadata_path "$root/03_data_contract/train_manifest.json" \
    --data_file_keys "video,vace_video,vace_reference_image" \
    --height 672 --width 672 --num_frames 33 --dataset_num_workers 0 \
    --model_paths "[\"$PWD/models/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors\",\"$PWD/models/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth\",\"$PWD/models/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth\"]" \
    --tokenizer_path "$PWD/models/Wan2.1-VACE-1.3B/google/umt5-xxl" \
    --learning_rate 1e-4 --lr_decay_start_step 400 --lr_after_decay 5e-5 \
    --num_epochs 1 --max_train_steps 700 --save_steps 50 \
    --output_path "$out" \
    --lora_base_model "vace" --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 16 \
    --extra_inputs "vace_video,vace_reference_image" --enable_csv_log \
    --lora_checkpoint "$checkpoint" --start_step "$start" \
    --resume_note "Same P49A fresh LoRA continuation from complete global step ${start}; optimizer reinitialized; scheduler restored to global step ${start}; no old LoRA." \
    > "$log" 2>&1
  status=$?
  set -e
  finished=$(date -Is)
  latest=$(latest_step)
  termination=normal
  grep -q 'Signal 15 (SIGTERM)' "$log" && termination=external_sigterm
  grep -q 'illegal memory access' "$log" && termination=illegal_cuda_memory_access
  printf '%s,%s,%s,%s,%s,%s\n' "$started" "$start" "$finished" "$status" "$latest" "$termination" >> "$audit"
  if (( latest >= 700 )); then
    printf '0\n' > "$final_status"
    exit 0
  fi
  if (( latest <= start )); then
    if [[ "$termination" == "external_sigterm" || "$termination" == "illegal_cuda_memory_access" ]]; then
      sleep 60
      continue
    fi
    printf '%s\n' "$status" > "$final_status"
    exit "$status"
  fi
  sleep 15
done
