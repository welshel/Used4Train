#!/usr/bin/env bash
set -euo pipefail
cd /fs1/private/user/baitongyuan/projects/liuzh
root="$PWD/outputs/p49a_tripo_vace13b_fresh"
model="$PWD/models/Wan2.1-VACE-1.3B"
condition="$PWD/outputs/p48_3_tripo_dynamic_cleanup/06_full72/condition_rgb_dynamic"
target="$PWD/outputs/p48_tripo_clean_aligned/clean_rgb"
pids=()
for spec in 1:0 2:17 3:50; do
  gpu="${spec%%:*}"
  start="${spec##*:}"
  out="$root/07_start31_validation/step400_start${start}_prelim"
  rm -rf "$out"
  mkdir -p "$out"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export TRANSFORMERS_CACHE="$PWD/cache/huggingface"
    export HF_HOME="$PWD/cache/huggingface"
    export TORCH_HOME="$PWD/cache/torch"
    export TOKENIZERS_PARALLELISM=false
    export DIFFSYNTH_SKIP_DOWNLOAD=True
    envs/p49a_vace13b/bin/python scripts/p49a_validate.py \
      --model-dir "$model" --condition-dir "$condition" --target-dir "$target" \
      --output-dir "$out" --checkpoint "$root/06_checkpoints/step-400.safetensors" \
      --start "$start" --num-frames 33 --resolution 672 --inference-steps 20 \
      --cfg-scale 5 --seed 20260915 > "$out/run.log" 2>&1
    printf '%s\n' "$?" > "$out/exit_status.txt"
  ) &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
exit "$rc"
