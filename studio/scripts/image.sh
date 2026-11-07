#!/bin/bash
# One image with Qwen-Image-2.1-Turbo. Usage: bash scripts/image.sh "a prompt" out.png
set -euo pipefail
# On a DGX Spark (Linux, CUDA) the same interface is served by spark/image.sh.
[ "$(uname -s)" != Linux ] || exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/spark/image.sh" "$@"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$HOME/qwen-models/Qwen-Image-2.1-Turbo}"
QWEN_BASE_DIR="${QWEN_BASE_DIR:-$HOME/qwen-models/Qwen-Image-2.1}"
[ -d "$QWEN_MODEL_DIR" ] || QWEN_MODEL_DIR="$QWEN_BASE_DIR"
TURBO_ADAPTER="${TURBO_ADAPTER:-$PREFIX/adapters/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors}"
PROMPT="${1:-}"; OUT="${2:-outputs/image.png}"
mkdir -p "$(dirname "$OUT")"
export PATH="/opt/homebrew/bin:$PATH"
case "${IMAGE_MODEL:-qwen}" in
  qwen) ;;
  *) echo "IMAGE_MODEL is qwen, not ${IMAGE_MODEL}" >&2; exit 2;;
esac
# the Turbo checkpoint carries its schedule in model_index.json; the base model does not
if grep -q sample_sigmas "$QWEN_MODEL_DIR/model_index.json" 2>/dev/null; then DISTILLED=1; else DISTILLED=0; fi
SIZE=(-o "$OUT" --width "${WIDTH:-1344}" --height "${HEIGHT:-768}" --seed "${SEED:-0}")
if [ "${TURBO:-1}" = 1 ] && [ "$DISTILLED" = 1 ]; then
  ARGS=("$QWEN_MODEL_DIR" "${SIZE[@]}" --nodes 1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568)
elif [ "${TURBO:-1}" = 1 ]; then
  ARGS=("$QWEN_MODEL_DIR" "${SIZE[@]}" --lora "$TURBO_ADAPTER" --nodes 1.0,0.9375,0.875,0.75,0.5,0.25)
else
  [ "$DISTILLED" = 0 ] || QWEN_MODEL_DIR="$QWEN_BASE_DIR"
  [ -d "$QWEN_MODEL_DIR/transformer" ] || { echo "TURBO=0 needs the base model: hf download Qwen/Qwen-Image-2.1 --local-dir $QWEN_BASE_DIR" >&2; exit 1; }
  ARGS=("$QWEN_MODEL_DIR" "${SIZE[@]}" --steps "${STEPS:-40}")
fi
if [ -n "${PROMPT_FILE:-}" ]; then ARGS+=(--prompt-file "$PROMPT_FILE"); else ARGS+=(--prompt "$PROMPT"); fi
# shellcheck disable=SC2086
exec "$PREFIX/venv/bin/python" "$PREFIX/qwen_image_generate.py" "${ARGS[@]}" ${EXTRA:-}
