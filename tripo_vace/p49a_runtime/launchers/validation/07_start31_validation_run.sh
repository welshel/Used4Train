#!/usr/bin/env bash
set -euo pipefail
cd /fs1/private/user/baitongyuan/projects/liuzh
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE="$PWD/cache/huggingface"
export HF_HOME="$PWD/cache/huggingface"
export TORCH_HOME="$PWD/cache/torch"
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=True
root="$PWD/outputs/p49a_tripo_vace13b_fresh"
model="$PWD/models/Wan2.1-VACE-1.3B"
condition="$PWD/outputs/p48_3_tripo_dynamic_cleanup/06_full72/condition_rgb_dynamic"
target="$PWD/outputs/p48_tripo_clean_aligned/clean_rgb"
run_one() {
  local name="$1"; local start="$2"; local checkpoint="$3"
  local out="$root/07_start31_validation/$name"
  mkdir -p "$out"
  if [[ -n "$checkpoint" ]]; then
    envs/p49a_vace13b/bin/python scripts/p49a_validate.py --model-dir "$model" --condition-dir "$condition" --target-dir "$target" --output-dir "$out" --checkpoint "$checkpoint" --start "$start" --num-frames 33 --resolution 672 --inference-steps 20 --cfg-scale 5 --seed 20260915
  else
    envs/p49a_vace13b/bin/python scripts/p49a_validate.py --model-dir "$model" --condition-dir "$condition" --target-dir "$target" --output-dir "$out" --start "$start" --num-frames 33 --resolution 672 --inference-steps 20 --cfg-scale 5 --seed 20260915
  fi
}
run_one step0_base_start31 31 ""
for step in 50 100 200 350 500 700; do
  ckpt="$root/06_checkpoints/step-${step}.safetensors"
  if [[ -f "$ckpt" ]]; then run_one "step${step}_start31" 31 "$ckpt"; fi
done
for step in 200 350 500 700; do
  ckpt="$root/06_checkpoints/step-${step}.safetensors"
  if [[ -f "$ckpt" ]]; then
    for start in 0 17 31 50; do run_one "step${step}_start${start}" "$start" "$ckpt"; done
  fi
done
