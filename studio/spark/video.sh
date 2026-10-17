#!/bin/bash
# One clip on a DGX Spark (CUDA): scripts/video.sh hands over here on Linux with the same interface (env: README).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
[ "${ENGINE:-fasth3}" = fasth3 ] || { echo "the Spark build renders with FastH3 only (ENGINE=fasth3)" >&2; exit 2; }
[ "${X2:-0}" != 1 ] || { echo "the 2x decoder is not part of the Spark build: use WIDTH and HEIGHT up to 1344x768" >&2; exit 2; }
[ -x "$PREFIX/venv/bin/python" ] || { echo "run oneshot-setup-spark.sh first: no environment at $PREFIX" >&2; exit 1; }
PROMPT="${1:-}"; OUT="${2:-outputs/h3.mp4}"
ARGS=(video -o "$OUT" --width "${WIDTH:-864}" --height "${HEIGHT:-480}" --frames "${FRAMES:-124}"
      --seed "${SEED:-0}")
# a list of files from the environment: separated by spaces, or one per line when a path has spaces
ref_list() {
  LIST=()
  [ -n "$1" ] || return 0
  case "$1" in *$'\n'*) while IFS= read -r REF; do [ -z "$REF" ] || LIST+=("$REF"); done <<<"$1";;
    *) read -r -a LIST <<<"$1";; esac
}
ref_list "${REF_IMAGES:-}"; REFS=(${LIST[@]+"${LIST[@]}"})
ref_list "${REF_VIDEOS:-}"; REF_CLIPS=(${LIST[@]+"${LIST[@]}"})
ref_list "${REF_AUDIOS:-}"; REF_SOUNDS=(${LIST[@]+"${LIST[@]}"})
REF_COUNT=$(( ${#REFS[@]} + ${#REF_CLIPS[@]} + ${#REF_SOUNDS[@]} ))
if [ "$REF_COUNT" -gt 0 ]; then
  [ -z "${FIRST_FRAME:-}${LAST_FRAME:-}" ] || { echo "references (REF_IMAGES, REF_VIDEOS, REF_AUDIOS) and FIRST_FRAME / LAST_FRAME do not combine: a reference is not a frame of the clip" >&2; exit 2; }
  # MiniMax's limits for one request: 9 pictures, 3 videos, 3 audio clips, 12 files in all
  [ "${#REFS[@]}" -le 9 ] || { echo "at most 9 reference pictures, got ${#REFS[@]}" >&2; exit 2; }
  [ "${#REF_CLIPS[@]}" -le 3 ] || { echo "at most 3 reference videos, got ${#REF_CLIPS[@]}" >&2; exit 2; }
  [ "${#REF_SOUNDS[@]}" -le 3 ] || { echo "at most 3 reference audio clips, got ${#REF_SOUNDS[@]}" >&2; exit 2; }
  [ "$REF_COUNT" -le 12 ] || { echo "at most 12 references in all, got $REF_COUNT" >&2; exit 2; }
  [ "${#REF_SOUNDS[@]}" = 0 ] || [ $(( ${#REFS[@]} + ${#REF_CLIPS[@]} )) -gt 0 ] \
    || { echo "REF_AUDIOS needs a reference picture or video beside it (REF_IMAGES or REF_VIDEOS): the model does not take sound alone" >&2; exit 2; }
  for REF in ${REFS[@]+"${REFS[@]}"}; do [ -s "$REF" ] || { echo "no reference picture at $REF" >&2; exit 2; }; ARGS+=(--ref "$REF"); done
  for REF in ${REF_CLIPS[@]+"${REF_CLIPS[@]}"}; do [ -s "$REF" ] || { echo "no reference video at $REF" >&2; exit 2; }; ARGS+=(--ref-video "$REF"); done
  for REF in ${REF_SOUNDS[@]+"${REF_SOUNDS[@]}"}; do [ -s "$REF" ] || { echo "no reference audio clip at $REF" >&2; exit 2; }; ARGS+=(--ref-audio "$REF"); done
  [ "${REF_VIDEO_SIZE:-match}" = match ] || { echo "REF_VIDEO_SIZE=$REF_VIDEO_SIZE is not offered on a DGX Spark: a reference video is always brought down to the clip's pixel area there (match), for memory" >&2; exit 2; }
  [ "${#REF_CLIPS[@]}" = 0 ] || [ "${REF_VIDEO_SOUND:-0}" != 1 ] || ARGS+=(--ref-video-sound)
  if [ -n "${STEPS:-}" ]; then
    case "$STEPS" in *[!0-9]*|0) echo "STEPS must be a positive number of passes, got $STEPS" >&2; exit 2;; esac
    ARGS+=(--steps "$STEPS")
  fi
  if [ "${REF_TURBO:-0}" = 1 ]; then ARGS+=(--ref-mode turbo); PASSES="${STEPS:-4}"; else ARGS+=(--ref-mode full); PASSES="${STEPS:-20}"; fi
  [ -z "${REF_SIZE:-}" ] || ARGS+=(--ref-size "$REF_SIZE")
  echo "[tensorfold] reference to video: $PASSES passes$([ "${REF_TURBO:-0}" != 1 ] || echo " with the Ref2VA Turbo adapter"), ${#REFS[@]} picture(s), ${#REF_CLIPS[@]} video(s), ${#REF_SOUNDS[@]} audio clip(s)"
  KIND="MiniMax H3 Ref2VA, ${#REFS[@]} picture(s), ${#REF_CLIPS[@]} video(s), ${#REF_SOUNDS[@]} audio clip(s), $PASSES passes"
else
  if [ "${SPARK_BASE:-fasth3}" = h3 ]; then ARGS+=(--base h3 --steps "${STEPS:-25}"); KIND="MiniMax H3 ${STEPS:-25} passes"
  else ARGS+=(--steps "${STEPS:-8}"); KIND="FastH3 ${STEPS:-8} passes"; fi
fi
[ -z "${FIRST_FRAME:-}" ] || ARGS+=(--first-frame "$FIRST_FRAME")
[ -z "${LAST_FRAME:-}" ] || { [ -s "$LAST_FRAME" ] || { echo "no last frame at $LAST_FRAME" >&2; exit 2; }; ARGS+=(--last-frame "$LAST_FRAME"); }
[ -z "${TEXT_ENCODER:-}" ] || ARGS+=(--text-encoder "$TEXT_ENCODER")
[ -z "${CROP:-}" ] || ARGS+=(--crop "$CROP")
[ -z "${SPARK_SPARSITY:-}" ] || ARGS+=(--sparsity "$SPARK_SPARSITY")
[ -z "${SPARK_ENGINE:-}" ] || ARGS+=(--engine "$SPARK_ENGINE")
# hub LoRAs, one PATH:STRENGTH a line, merged by ComfyUI into the bf16 weights (FastH3's, or Ref2VA's for references)
if [ -n "${LORA:-}" ]; then
  while IFS= read -r ONE; do [ -z "$ONE" ] || ARGS+=(--lora "$ONE"); done <<<"$LORA"
fi
# REF_PLAIN=1: a reference clip's prompt goes to the model as written (a LoRA trained on plain captions)
[ "$REF_COUNT" = 0 ] || [ "${REF_PLAIN:-0}" != 1 ] || ARGS+=(--ref-plain)
if [ -n "${PROMPT_FILE:-}" ]; then ARGS+=(--prompt-file "$PROMPT_FILE"); else ARGS+=(--prompt "$PROMPT"); fi
START=$(date +%s)
PREFIX="$PREFIX" "$PREFIX/venv/bin/python" "$HERE/spark_generate.py" "${ARGS[@]}"
echo "[studio] $KIND, ${WIDTH:-864}x${HEIGHT:-480}, total $(( $(date +%s) - START )) s -> $OUT"
