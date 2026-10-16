#!/bin/bash
# Five-view turnaround sheet with matlod's Turnaround LoRA on MiniMax H3 Ref2VA. Usage: turnaround.sh PICTURE OUT_PREFIX
set -euo pipefail
# after MATLOWAI's contact-sheet sampler: five stills ride the five latent steps of a 17-frame clip
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
HOME_DIR="${STUDIO_HOME:-$HOME/TensorFoldStudio}"
REF="${1:-}"; OUT="${2:-outputs/turn}"
[ -s "$REF" ] || { echo "usage: turnaround.sh PICTURE OUT_PREFIX (no picture at '$REF')" >&2; exit 2; }
SIZE="${SIZE:-1024}"; STRENGTH="${STRENGTH:-1.0}"; STEPS="${STEPS:-28}"
# "level" and "upright" are there because the LoRA's bare instruction is sometimes read as a roll of the picture
PROMPT="${PROMPT:-the camera orbits horizontally around the subject of <Picture 1> at eye level, ninety degrees clockwise, the subject stays upright on the ground}"
case "$SIZE" in 512|1024|2048) ;; *) echo "SIZE is 512, 1024 or 2048, got $SIZE" >&2; exit 2;; esac
case "$PROMPT" in *"<Picture 1>"*) ;; *) echo "the prompt names the picture as <Picture 1>" >&2; exit 2;; esac
if [ "$(uname -s)" = Linux ]; then
  LORA="${TURNAROUND_LORA:-$HOME_DIR/loras/turnaround.safetensors}"
  [ -s "$LORA" ] || { echo "no Turnaround LoRA at $LORA: install it in the LoRA hub" >&2; exit 2; }
  exec env PREFIX="$PREFIX" "$PREFIX/venv/bin/python" "$HERE/../spark/spark_workflows.py" turnaround --ref "$REF" \
    --lora "$LORA:$STRENGTH" --size "$SIZE" --seed "${SEED:-0}" --steps "$STEPS" --prompt "$PROMPT" -o "$OUT"
fi
H3_MODEL_DIR="${H3_MODEL_DIR:-$HOME/h3-models/MiniMax-H3}"
LORA="${TURNAROUND_LORA:-$HOME_DIR/loras/turnaround.mlx.safetensors}"   # the hub's copy in the MLX engine's layout
[ -s "$LORA" ] || { echo "no Turnaround LoRA at $LORA: install it in the LoRA hub (it rewrites the file for the MLX engine)" >&2; exit 2; }
[ -d "$H3_MODEL_DIR/Ref2VA/transformer" ] || { echo "MiniMax H3 Ref2VA is not installed (oneshot-setup.sh --ref2va)" >&2; exit 2; }
for PART in text_encoder tokenizer video_vae audio_vae processor; do
  [ -e "$H3_MODEL_DIR/Ref2VA/$PART" ] || [ ! -e "$H3_MODEL_DIR/FL2VA/$PART" ] || ln -s "../FL2VA/$PART" "$H3_MODEL_DIR/Ref2VA/$PART"
done
export PATH="/opt/homebrew/bin:$PATH" PYTHONPATH="$PREFIX/minimax-h3-mlx"
mkdir -p "$(dirname "$OUT")"
echo "[tensorfold] engine: mlx (MiniMax H3 Ref2VA with the Turnaround LoRA at $STRENGTH), five views of ${SIZE}x${SIZE}, $STEPS passes"
START=$(date +%s)
# the passes in bfloat16: an adapted projection does not run on an int8 kernel
"$PREFIX/venv/bin/python" "${H3_GENERATE:-$HERE/../h3_generate.py}" "$H3_MODEL_DIR" -o /dev/null --sheet "$OUT" --ref-image "$REF" \
  --width "$SIZE" --height "$SIZE" --frames 22 --seed "${SEED:-0}" --points $((STEPS + 1)) --lora "$LORA:$STRENGTH" --prompt "$PROMPT" ${EXTRA:-}
echo "[studio] turnaround sheet, five views of ${SIZE}x${SIZE}, total $(( $(date +%s) - START )) s -> ${OUT}_sheet.png"
