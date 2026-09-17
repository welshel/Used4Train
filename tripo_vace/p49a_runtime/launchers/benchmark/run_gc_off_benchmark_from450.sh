#!/usr/bin/env bash
set -euo pipefail
cd /fs1/private/user/baitongyuan/projects/liuzh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRANSFORMERS_CACHE="$PWD/cache/huggingface"
export HF_HOME="$PWD/cache/huggingface"
export TORCH_HOME="$PWD/cache/torch"
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
root="$PWD/outputs/p49a_tripo_vace13b_fresh"
out="$root/04_multigpu_benchmark/run_gc_off_from450"
rm -rf "$out"
mkdir -p "$out"
(while true; do date -Is; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader; sleep 2; done) > "$root/04_multigpu_benchmark/nvidia_smi_gc_off_from450.log" 2>&1 &
monitor=$!
set +e
envs/p49a_vace13b/bin/accelerate launch --multi_gpu --num_processes 4 --mixed_precision no --main_process_port 0 scripts/p49a_train_resume.py \
  --dataset_base_path "$PWD" --dataset_metadata_path "$root/03_data_contract/train_manifest.json" \
  --data_file_keys "video,vace_video,vace_reference_image" --height 672 --width 672 --num_frames 33 --dataset_num_workers 0 \
  --model_paths "[\"$PWD/models/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors\",\"$PWD/models/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth\",\"$PWD/models/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth\"]" \
  --tokenizer_path "$PWD/models/Wan2.1-VACE-1.3B/google/umt5-xxl" \
  --learning_rate 1e-4 --lr_decay_start_step 400 --lr_after_decay 5e-5 \
  --num_epochs 1 --max_train_steps 455 --save_steps 1000 --output_path "$out" \
  --lora_base_model vace --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 16 \
  --extra_inputs "vace_video,vace_reference_image" --enable_csv_log \
  --lora_checkpoint "$root/06_checkpoints/step-450.safetensors" --start_step 450 \
  --force_gradient_checkpointing_off --resume_note "GC-OFF speed/VRAM benchmark only from P49A step450" \
  > "$root/04_multigpu_benchmark/benchmark_gc_off_from450.log" 2>&1
status=$?
kill "$monitor" 2>/dev/null || true
wait "$monitor" 2>/dev/null || true
printf '%s\n' "$status" > "$root/04_multigpu_benchmark/gc_off_from450_exit_status.txt"
exit "$status"
