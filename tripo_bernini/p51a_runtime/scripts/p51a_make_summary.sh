#!/usr/bin/env bash
set -euo pipefail
# Five-way synchronized QA only; clean is never passed to Bernini inference.
ffmpeg -hide_banner -loglevel error -y \
  -i "$1" -i "$2" -i "$3" -i "$4" -i "$5" \
  -filter_complex "[0:v]scale=672:672[a];[1:v]scale=672:672[b];[2:v]scale=672:672[c];[3:v]scale=672:672[d];[4:v]scale=672:672[e];[a][b][c][d][e]hstack=inputs=5[v]" \
  -map "[v]" -c:v libx264 -crf 18 -pix_fmt yuv420p "$6"
