#!/usr/bin/env bash
# One bounded post-training P51A finalization unit per invocation.
set -euo pipefail

ROOT=/fs1/private/user/baitongyuan/projects/liuzh
OUT=$ROOT/outputs/p51a_bernini_r13b_single_scene
REPO=$ROOT/repos/Bernini
VENV=$REPO/.venv/bin/python
RUN=$OUT/training/run_700
FINAL=$OUT/final
LOGDIR=$OUT/logs

export PYTHONPATH="$REPO:$ROOT/repos/VeOmni-v0.1.11${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
mkdir -p "$FINAL" "$LOGDIR" "$OUT/full72/adjusted/chunks" "$OUT/full72/raw/chunks"

exec 9>"$OUT/full72/P51A_POSTTRAIN_CONTROLLER.lock"
flock -n 9 || exit 0
[[ -f "$RUN/P51A_TRAINING_COMPLETE" ]] || exit 0
[[ -f "$FINAL/P51A_FINAL_COMPLETE.json" ]] && exit 0

if [[ ! -f "$FINAL/P51A_CHECKPOINT_SELECTION.json" ]]; then
  "$VENV" "$OUT/scripts/p51a_select_checkpoint.py" >"$LOGDIR/select_best_checkpoint.log" 2>&1
  exit 0
fi

if [[ ! -f "$OUT/full72/inputs/P51A_INPUTS_COMPLETE.json" ]]; then
  "$VENV" "$OUT/scripts/p51a_prepare_full72_inputs.py" >"$LOGDIR/prepare_full72_inputs.log" 2>&1
  exit 0
fi

hf_checkpoint=$(awk -F= '$1 == "hf_checkpoint" {print $2}' "$FINAL/P51A_BEST_CHECKPOINT.txt")
[[ -n "$hf_checkpoint" && -f "$hf_checkpoint/P51A_HF_EXPORT_COMPLETE.json" ]] || { echo "invalid selected HF checkpoint" >&2; exit 2; }

for route in adjusted raw; do
  for start in 0 31 50; do
    output="$OUT/full72/$route/chunks/start$(printf '%02d' "$start").mp4"
    marker="${output%.mp4}.P51A_GENERATION_COMPLETE.json"
    if [[ ! -f "$marker" ]]; then
      if [[ "$route" == adjusted ]]; then
        condition="$OUT/clips/adjusted/cyclic_start$(printf '%02d' "$start").mkv"
      else
        condition="$OUT/full72/inputs/raw_cyclic_start$(printf '%02d' "$start").mkv"
      fi
      CUDA_VISIBLE_DEVICES=0 "$VENV" "$OUT/scripts/p51a_generate_v2v_clip.py" \
        --base "$ROOT/models/Bernini-R-1.3B-Diffusers" \
        --checkpoint "$hf_checkpoint" \
        --condition-video "$condition" \
        --output-video "$output" \
        --route "$route" --start "$start" \
        >"$LOGDIR/full72_${route}_start$(printf '%02d' "$start").log" 2>&1
      exit 0
    fi
  done
done

for route in adjusted raw; do
  output="$OUT/full72/$route/output_full72.mp4"
  marker="${output%.mp4}.P51A_ASSEMBLY_COMPLETE.json"
  if [[ ! -f "$marker" ]]; then
    "$VENV" "$OUT/scripts/p51a_assemble_full72.py" --route "$route" --chunks-dir "$OUT/full72/$route/chunks" --output-video "$output" \
      >"$LOGDIR/assemble_full72_${route}.log" 2>&1
    exit 0
  fi
done

adjusted_cmp=$FINAL/P51A_ADJUSTED_VS_OUTPUT_VS_CLEAN_FULL72.mp4
raw_cmp=$FINAL/P51A_RAW_VS_OUTPUT_VS_CLEAN_FULL72.mp4
summary_cmp=$FINAL/P51A_ADJUSTED_RAW_CLEAN_SUMMARY.mp4
if [[ ! -f "$adjusted_cmp" || ! -f "$raw_cmp" || ! -f "$summary_cmp" ]]; then
  for partial in "$adjusted_cmp" "$raw_cmp" "$summary_cmp"; do
    if [[ -e "$partial" ]]; then
      mv "$partial" "${partial}.interrupted_$(date +%Y%m%dT%H%M%S%z)"
    fi
  done
  "$OUT/scripts/p51a_make_comparisons.sh" "$OUT/full72/inputs/condition_adjusted_full72.mp4" "$OUT/full72/adjusted/output_full72.mp4" "$OUT/full72/inputs/clean_full72.mp4" "$adjusted_cmp"
  "$OUT/scripts/p51a_make_comparisons.sh" "$OUT/full72/inputs/condition_raw_full72.mp4" "$OUT/full72/raw/output_full72.mp4" "$OUT/full72/inputs/clean_full72.mp4" "$raw_cmp"
  "$OUT/scripts/p51a_make_summary.sh" "$OUT/full72/inputs/condition_adjusted_full72.mp4" "$OUT/full72/adjusted/output_full72.mp4" "$OUT/full72/inputs/condition_raw_full72.mp4" "$OUT/full72/raw/output_full72.mp4" "$OUT/full72/inputs/clean_full72.mp4" "$summary_cmp"
  exit 0
fi

"$VENV" "$OUT/scripts/p51a_finalize.py" >"$LOGDIR/finalize_p51a.log" 2>&1
