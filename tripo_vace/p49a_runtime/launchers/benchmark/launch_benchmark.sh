#!/usr/bin/env bash
set -euo pipefail
cd /fs1/private/user/baitongyuan/projects/liuzh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRANSFORMERS_CACHE="$PWD/cache/huggingface" HF_HOME="$PWD/cache/huggingface" TORCH_HOME="$PWD/cache/torch" TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/repos/DiffSynth-Studio:${PYTHONPATH:-}"
exec "$PWD/envs/p49a_vace13b/bin/accelerate" launch --multi_gpu --num_processes 4 \
  scripts/p49a_train.py \
  --dataset_base_path "$PWD" \
  --dataset_metadata_path "$PWD/outputs/p49a_tripo_vace13b_fresh/04_multigpu_benchmark/benchmark_manifest.json" \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 672 --width 672 --num_frames 33 --dataset_repeat 1 --dataset_num_workers 0 \
  --model_paths "[\"$PWD/models/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors\",\"$PWD/models/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth\",\"$PWD/models/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth\"]" \
  --tokenizer_path "$PWD/models/Wan2.1-VACE-1.3B/google/umt5-xxl" \
  --learning_rate 1e-4 --num_epochs 1 --save_steps 5 \
  --output_path "$PWD/outputs/p49a_tripo_vace13b_fresh/04_multigpu_benchmark/run" \
  --lora_base_model "vace" --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 16 \
  --extra_inputs "vace_video,vace_reference_image" \
  --enable_csv_log
