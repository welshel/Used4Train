#!/usr/bin/env bash
set -u -o pipefail

ROOT=/fs1/private/user/baitongyuan/projects/liuzh
MODEL="$ROOT/models/Bernini-R-1.3B-Diffusers"
OUT="$ROOT/outputs/p51a_bernini_r13b_single_scene"
REV=ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce
BASE="https://hf-mirror.com/ByteDance/Bernini-R-1.3B-Diffusers/resolve/$REV"

download() {
  local rel="$1" expected="$2" target="$MODEL/$1"
  local log="$OUT/logs/sequential_${rel//\//_}.log" attempt=0 current=0
  mkdir -p "$(dirname "$target")"
  while (( attempt < 40 )); do
    current=0
    [[ -f "$target" ]] && current=$(stat -c%s "$target")
    if (( current == expected )); then
      printf '[%s] complete %s bytes=%s\n' "$(date -Is)" "$rel" "$current" | tee -a "$log"
      return 0
    fi
    if (( current > expected )); then
      printf '[%s] fail oversized %s local=%s expected=%s\n' "$(date -Is)" "$rel" "$current" "$expected" | tee -a "$log"
      return 2
    fi
    attempt=$((attempt + 1))
    printf '[%s] attempt=%s resume %s local=%s expected=%s\n' "$(date -Is)" "$attempt" "$rel" "$current" "$expected" | tee -a "$log"
    if curl -4 --fail --location --continue-at - --retry 8 --retry-delay 10 \
      --connect-timeout 30 --speed-time 180 --speed-limit 1024 --output "$target" "$BASE/$rel" >>"$log" 2>&1; then
      :
    else
      printf '[%s] curl failed; retaining partial data for next attempt\n' "$(date -Is)" | tee -a "$log"
    fi
    sleep 5
  done
  printf '[%s] exhausted retries for %s\n' "$(date -Is)" "$rel" | tee -a "$log"
  return 3
}

download text_encoder/model-00005-of-00005.safetensors 2885866152 || exit $?
download text_encoder/model-00004-of-00005.safetensors 4999880704 || exit $?
