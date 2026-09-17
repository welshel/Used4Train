#!/usr/bin/env bash
set -u -o pipefail

period="${1:?checkpoint period required}"
gpu="${2:?physical GPU index required}"
start="${3:-450}"
end="${4:-456}"
warmup_end="${5:-452}"
tail_blocks="${6:-0}"

cd /fs1/private/user/baitongyuan/projects/liuzh
root="$PWD/outputs/p49a_tripo_vace13b_fresh"
out="$root/04_multigpu_benchmark/selective_gc_period${period}_tail${tail_blocks}_gpu${gpu}_from${start}"
checkpoint="$root/06_checkpoints/step-${start}.safetensors"
mkdir -p "$out"

used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used >= 1000 )); then
  echo "GPU $gpu is not free: ${used} MiB used" >&2
  exit 42
fi
if [[ ! -s "$checkpoint" ]]; then
  echo "Missing checkpoint: $checkpoint" >&2
  exit 43
fi

export CUDA_VISIBLE_DEVICES="$gpu"
export TRANSFORMERS_CACHE="$PWD/cache/huggingface"
export HF_HOME="$PWD/cache/huggingface"
export TORCH_HOME="$PWD/cache/torch"
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

(while true; do
  date -Is
  nvidia-smi --id="$gpu" --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader
  sleep 1
done) > "$out/nvidia_smi.log" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

command=(
  envs/p49a_vace13b/bin/python scripts/p49a_train_resume.py
  --dataset_base_path "$PWD"
  --dataset_metadata_path "$root/03_data_contract/train_manifest.json"
  --data_file_keys "video,vace_video,vace_reference_image"
  --height 672 --width 672 --num_frames 33 --dataset_num_workers 0
  --model_paths "[\"$PWD/models/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors\",\"$PWD/models/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth\",\"$PWD/models/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth\"]"
  --tokenizer_path "$PWD/models/Wan2.1-VACE-1.3B/google/umt5-xxl"
  --learning_rate 1e-4 --lr_decay_start_step 400 --lr_after_decay 5e-5
  --num_epochs 1 --max_train_steps "$end" --benchmark_warmup_steps "$warmup_end" --save_steps 1000
  --output_path "$out"
  --lora_base_model "vace" --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 16
  --extra_inputs "vace_video,vace_reference_image" --enable_csv_log
  --gradient_checkpointing_period "$period"
  --gradient_checkpointing_uncheckpointed_tail_blocks "$tail_blocks"
  --lora_checkpoint "$checkpoint" --start_step "$start"
  --resume_note "P49A selective-GC benchmark only; same-run LoRA step $start; no formal checkpoint promotion."
)

printf '%q ' "${command[@]}" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
set +e
"${command[@]}" > "$out/train.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$out/exit_status.txt"
exit "$status"
