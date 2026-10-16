#!/bin/bash
# One image with Qwen-Image-2.1-Turbo (or FLUX3-K: IMAGE_MODEL). Usage: bash scripts/image.sh "a prompt" out.png
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
# FLUX3-K via flux3-mlx in its own venv (--flux3): Fast 16 steps, Full 50; BFL's upsampler first unless UPSAMPLE=0
case "${IMAGE_MODEL:-qwen}" in
  flux3k-fast|flux3k-full)
    FLUX3_MODEL_DIR="${FLUX3_MODEL_DIR:-$HOME/flux3-models}"
    CKPT="$FLUX3_MODEL_DIR/$([ "$IMAGE_MODEL" = flux3k-fast ] && echo FLUX3-K-Fast || echo FLUX3-K-Full)"
    [ -x "$PREFIX/flux3-venv/bin/python" ] && [ -s "$CKPT/model.safetensors" ] \
      || { echo "FLUX3-K is not installed here (no $CKPT): bash oneshot-setup.sh --flux3" >&2; exit 1; }
    ARGS=("$CKPT" --model "$IMAGE_MODEL" -o "$OUT" --width "${WIDTH:-1344}" --height "${HEIGHT:-768}" --seed "${SEED:-0}")
    [ "${UPSAMPLE:-1}" = 0 ] || ARGS+=(--upsampler "$FLUX3_MODEL_DIR/FLUX3-K-pup-MLX-8bit")
    if [ -n "${PROMPT_FILE:-}" ]; then ARGS+=(--prompt-file "$PROMPT_FILE"); else ARGS+=(--prompt "$PROMPT"); fi
    # shellcheck disable=SC2086
    exec "$PREFIX/flux3-venv/bin/python" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/flux3_generate.py" "${ARGS[@]}" ${EXTRA:-};;
  qwen) ;;
  *) echo "IMAGE_MODEL is qwen, flux3k-fast or flux3k-full, not ${IMAGE_MODEL}" >&2; exit 2;;
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
