#!/usr/bin/env bash
set -u -o pipefail

ROOT=/fs1/private/user/baitongyuan/projects/liuzh
MODEL="$ROOT/models/Qwen2.5-VL-7B-Instruct"
OUT="$ROOT/outputs/p51a_bernini_r13b_single_scene"
REV=cc594898137f460bfe9f0759e9844b3ce807cfb5
BASE="https://hf-mirror.com/Qwen/Qwen2.5-VL-7B-Instruct/resolve/$REV"

download() {
  local rel="$1" expected="$2" target="$MODEL/$1"
  local log="$OUT/logs/qwen_resume_${rel//\//_}.log" attempt=0 current=0
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

download model-00001-of-00005.safetensors 3900233256 || exit $?
download model-00002-of-00005.safetensors 3864726320 || exit $?
download model-00003-of-00005.safetensors 3864726424 || exit $?
download model-00004-of-00005.safetensors 3864733680 || exit $?
download model-00005-of-00005.safetensors 1089994880 || exit $?
