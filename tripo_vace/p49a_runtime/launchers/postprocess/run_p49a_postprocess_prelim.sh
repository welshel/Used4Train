#!/usr/bin/env bash
set -euo pipefail
cd /fs1/private/user/baitongyuan/projects/liuzh
root="$PWD/outputs/p49a_tripo_vace13b_fresh"
mkdir -p "$root/09_object_qa"
status_file="$root/07_start31_validation/P49A_VALIDATIONS_EXIT_STATUS.txt"
while [[ ! -f "$status_file" ]]; do sleep 20; done
status=$(tr -d '[:space:]' < "$status_file")
if [[ "$status" != "0" ]]; then
  echo "Validation failed with status $status; postprocess not started." >&2
  exit 30
fi
envs/p49a_vace13b/bin/python scripts/p49a_postprocess.py \
  --root "$root" \
  --condition-dir "$PWD/outputs/p48_3_tripo_dynamic_cleanup/06_full72/condition_rgb_dynamic" \
  --target-dir "$PWD/outputs/p48_tripo_clean_aligned/clean_rgb" \
  > "$root/09_object_qa/P49A_POSTPROCESS_PRELIM.log" 2>&1
date -Is > "$root/09_object_qa/P49A_POSTPROCESS_PRELIM_FINISHED_AT.txt"
