#!/bin/bash
# Text to video with FastH3. Usage: [STEPS=4|8|20] [RES=480p|720p] [UPSCALE=1] bash scripts/fast.sh "a prompt" out.mp4
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${STEPS:=8}" in 4|8|20) ;; *) echo "STEPS is 4, 8 or 20, got $STEPS" >&2; exit 2;; esac
# 720 is not a multiple of 32, which the model needs: 720p is generated 736 rows tall and cropped by 8 top and bottom
case "${RES:-480p}:${UPSCALE:-0}" in
  480p:0) export WIDTH=864 HEIGHT=480;;
  480p:1) export X2=1 WIDTH=1728 HEIGHT=960;;
  720p:0) export WIDTH=1280 HEIGHT=736 CROP=1280x720;;
  720p:1) export X2=1 WIDTH=2560 HEIGHT=1472 CROP=2560x1440;;
  *) echo "RES is 480p or 720p and UPSCALE 0 or 1, got ${RES:-} and ${UPSCALE:-}" >&2; exit 2;;
esac
START=$(date +%s)
ENGINE=fasth3 STEPS="$STEPS" bash "$HERE/video.sh" "${1:-}" "${2:-outputs/fast.mp4}"
echo "[studio] FastH3 $STEPS passes, ${RES:-480p}$([ "${UPSCALE:-0}" = 1 ] && echo " + 2x decoder"), total $(( $(date +%s) - START )) s -> ${2:-outputs/fast.mp4}"
