#!/usr/bin/env bash
set -euo pipefail
cd /fs1/private/user/baitongyuan/projects/liuzh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRANSFORMERS_CACHE="$PWD/cache/huggingface"
export HF_HOME="$PWD/cache/huggingface"
export TORCH_HOME="$PWD/cache/torch"
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=True
train_dir="$PWD/outputs/p49a_tripo_vace13b_fresh/05_training"
mkdir -p "$PWD/outputs/p49a_tripo_vace13b_fresh/06_checkpoints"
(while true; do date -Is; nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader; sleep 5; done) > "$train_dir/nvidia_smi_4gpu_training_final2.log" 2>&1 &
monitor_pid=$!
date -Is > "$train_dir/training_final2_started_at.txt"
set +e
"$PWD/envs/p49a_vace13b/bin/accelerate" launch --multi_gpu --num_processes 4 --mixed_precision no scripts/p49a_train.py \
  --dataset_base_path "$PWD" \
  --dataset_metadata_path "$PWD/outputs/p49a_tripo_vace13b_fresh/03_data_contract/train_manifest.json" \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 672 --width 672 --num_frames 33 --dataset_num_workers 0 \
  --model_paths "[\"$PWD/models/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors\",\"$PWD/models/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth\",\"$PWD/models/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth\"]" \
  --tokenizer_path "$PWD/models/Wan2.1-VACE-1.3B/google/umt5-xxl" \
  --learning_rate 1e-4 --lr_decay_start_step 400 --lr_after_decay 5e-5 \
  --num_epochs 1 --max_train_steps 700 --save_steps 50 \
  --output_path "$PWD/outputs/p49a_tripo_vace13b_fresh/06_checkpoints" \
  --lora_base_model "vace" --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 16 \
  --extra_inputs "vace_video,vace_reference_image" --enable_csv_log \
  > "$train_dir/train_4gpu_final2.log" 2>&1
status=$?
date -Is > "$train_dir/training_final2_finished_at.txt"
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
printf "%s\n" "$status" > "$train_dir/training_final2_exit_status.txt"
exit "$status"
