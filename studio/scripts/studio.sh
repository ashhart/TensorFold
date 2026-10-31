#!/bin/bash
# Text to image to video (Qwen-Image-2.1, then MiniMax H3). Usage: scripts/studio.sh "picture" "what happens" out.mp4
set -euo pipefail
# WIDTH and HEIGHT suit both models: multiples of 32, at most 768x1344 pixels; X2=1 needs multiples of 64
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_PROMPT="${1:-}"; VIDEO_PROMPT="${2:-}"; OUT="${3:-outputs/studio.mp4}"
IMAGE="${OUT%.*}.png"
# QHD=1: 2560x1440 from 1280x736 at 2x, 16 rows cropped top and bottom; TWOK=1: 2048x1152 from 1024x576
if [ "${QHD:-0}" = 1 ]; then export WIDTH=2560 HEIGHT=1472 X2=1 CROP=2560x1440 IMAGE_WIDTH="${IMAGE_WIDTH:-1280}" IMAGE_HEIGHT="${IMAGE_HEIGHT:-736}"; fi
if [ "${TWOK:-0}" = 1 ]; then export WIDTH=2048 HEIGHT=1152 X2=1 IMAGE_WIDTH="${IMAGE_WIDTH:-1024}" IMAGE_HEIGHT="${IMAGE_HEIGHT:-576}"; fi
export WIDTH="${WIDTH:-1344}" HEIGHT="${HEIGHT:-768}"
[ -z "${VIDEO_PROMPT_FILE:-}" ] || VIDEO_PROMPT="$(cat "$VIDEO_PROMPT_FILE")"
[ -n "$VIDEO_PROMPT" ] || { echo "give a video prompt or VIDEO_PROMPT_FILE" >&2; exit 2; }
# MiniMax's own image-to-video prompts open by saying where the picture is used
LEAD="For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
case "$VIDEO_PROMPT" in "For the target video"*) ;; *) [ "${RAW_PROMPT:-0}" = 1 ] || VIDEO_PROMPT="$LEAD

$VIDEO_PROMPT";; esac
START=$(date +%s)
if [ -n "${IMAGE_FILE:-}" ]; then
  # an image chosen earlier (a scout, or any picture) takes the place of the image step
  [ -s "$IMAGE_FILE" ] || { echo "no image at $IMAGE_FILE" >&2; exit 2; }
  IMAGE="$IMAGE_FILE"
else
  SEED="${IMAGE_SEED:-${SEED:-0}}" PROMPT_FILE="${IMAGE_PROMPT_FILE:-}" WIDTH="${IMAGE_WIDTH:-$WIDTH}" \
    HEIGHT="${IMAGE_HEIGHT:-$HEIGHT}" bash "$HERE/image.sh" "$IMAGE_PROMPT" "$IMAGE"
fi
[ -s "$IMAGE" ] || { echo "no image was written" >&2; exit 1; }
MID=$(date +%s)
SEED="${VIDEO_SEED:-${SEED:-0}}" PROMPT_FILE="" FIRST_FRAME="$IMAGE" bash "$HERE/video.sh" "$VIDEO_PROMPT" "$OUT"
END=$(date +%s)
echo "[studio] image $((MID - START)) s, video $((END - MID)) s, total $((END - START)) s -> $IMAGE, $OUT"
