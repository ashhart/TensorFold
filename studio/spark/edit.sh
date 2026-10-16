#!/bin/bash
# Edit a picture on a DGX Spark (CUDA): scripts/edit.sh hands over here on Linux with the same interface (env: README).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
[ -x "$PREFIX/venv/bin/python" ] || { echo "run oneshot-setup-spark.sh first: no environment at $PREFIX" >&2; exit 1; }
IMAGE="${1:-}"; PROMPT="${2:-}"; OUT="${3:-}"
[ -f "$IMAGE" ] || { echo "give the picture to edit: bash scripts/edit.sh photo.png \"what changes\" out.png" >&2; exit 1; }
ARGS=(--image "$IMAGE" --seed "${SEED:-0}" --size "${SIZE:-source}" --resolution "${RESOLUTION:-1024}" --model "${IMAGE_MODEL:-qwen}")
if [ -n "${IMAGES:-}" ]; then
  while IFS= read -r ONE; do [ -z "$ONE" ] || ARGS+=(--image "$ONE"); done <<<"$IMAGES"
fi
[ -z "${WIDTH:-}" ] || [ -z "${HEIGHT:-}" ] || ARGS+=(--width "$WIDTH" --height "$HEIGHT")
[ -z "${MOST_PIXELS:-}" ] || ARGS+=(--most "$MOST_PIXELS")
[ "${BASE:-0}" != 1 ] || ARGS+=(--base --steps "${STEPS:-40}")
if [ -n "${SET_FILE:-}" ]; then ARGS+=(--set "@$SET_FILE"); else ARGS+=(--prompt "$PROMPT" -o "$OUT"); fi
if [ -n "${LORA:-}" ]; then
  while IFS= read -r ONE; do [ -z "$ONE" ] || ARGS+=(--lora "$ONE"); done <<<"$LORA"
fi
PREFIX="$PREFIX" exec "$PREFIX/venv/bin/python" "$HERE/spark_edit.py" "${ARGS[@]}"
