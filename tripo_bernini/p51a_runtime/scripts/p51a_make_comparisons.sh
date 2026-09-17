#!/usr/bin/env bash
set -euo pipefail
# hstack inputs only; clean is never supplied to inference, only to visual QA.
ffmpeg -hide_banner -loglevel error -y -i "$1" -i "$2" -i "$3" -filter_complex "[0:v]scale=672:672[a];[1:v]scale=672:672[b];[2:v]scale=672:672[c];[a][b][c]hstack=inputs=3[v]" -map "[v]" -map 0:a? -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac "$4"
