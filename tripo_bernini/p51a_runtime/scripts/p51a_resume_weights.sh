#!/usr/bin/env bash
set -euo pipefail

ROOT=/fs1/private/user/baitongyuan/projects/liuzh
MODEL="$ROOT/models/Bernini-R-1.3B-Diffusers"
OUT="$ROOT/outputs/p51a_bernini_r13b_single_scene"
REV=ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce
BASE="https://hf-mirror.com/ByteDance/Bernini-R-1.3B-Diffusers/resolve/$REV"

download() {
  local rel="$1"
  local expected="$2"
  local target="$MODEL/$rel"
  local log="$OUT/logs/resume_${rel//\//_}.log"
  mkdir -p "$(dirname "$target")"
  local current=0
  if [[ -f "$target" ]]; then current=$(stat -c%s "$target"); fi
  if [[ "$current" -eq "$expected" ]]; then
    printf '[%s] already complete: %s (%s bytes)\n' "$(date -Is)" "$rel" "$current" | tee -a "$log"
    return 0
  fi
  if [[ "$current" -gt "$expected" ]]; then
    printf '[%s] invalid oversized target: %s local=%s expected=%s\n' "$(date -Is)" "$rel" "$current" "$expected" | tee -a "$log"
    return 2
  fi
  printf '[%s] resume: %s local=%s expected=%s\n' "$(date -Is)" "$rel" "$current" "$expected" | tee -a "$log"
  curl -4 --fail --location --continue-at - --retry 30 --retry-delay 10 --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
    --output "$target" "$BASE/$rel" >>"$log" 2>&1
  local actual
  actual=$(stat -c%s "$target")
  if [[ "$actual" -ne "$expected" ]]; then
    printf '[%s] incomplete after curl: %s local=%s expected=%s\n' "$(date -Is)" "$rel" "$actual" "$expected" | tee -a "$log"
    return 3
  fi
  printf '[%s] complete: %s (%s bytes)\n' "$(date -Is)" "$rel" "$actual" | tee -a "$log"
}

download text_encoder/model-00003-of-00005.safetensors 4966309504 & p1=$!
download text_encoder/model-00004-of-00005.safetensors 4999880704 & p2=$!
download text_encoder/model-00005-of-00005.safetensors 2885866152 & p3=$!
download transformer/diffusion_pytorch_model-00001-of-00002.safetensors 5387151416 & p4=$!

status=0
for p in "$p1" "$p2" "$p3" "$p4"; do
  wait "$p" || status=1
done
exit "$status"
