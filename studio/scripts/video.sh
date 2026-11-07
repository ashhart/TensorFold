#!/bin/bash
# Render one clip with MiniMax H3 or FastH3 on TensorFold (Mac). Usage: bash scripts/video.sh "a prompt" out.mp4
set -euo pipefail
# On a DGX Spark (Linux, CUDA) the same interface is served by spark/video.sh.
[ "$(uname -s)" != Linux ] || exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/spark/video.sh" "$@"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
H3_MODEL_DIR="${H3_MODEL_DIR:-$HOME/h3-models/MiniMax-H3}"
# QUALITY=high: 20 steps without the adapter, fast recipe (step cache, attention reuse); QUALITY=full: plain 20 steps
case "${QUALITY:-standard}" in high) ADAPTER=none; RECIPE="--step-cache 0.05 --attention-every 2";; full) ADAPTER=none; RECIPE="";; *) RECIPE="";; esac
FASTH3_DIR="${FASTH3_DIR:-$HOME/h3-models/FastH3-8-Step-V2}"
# TEXT_ENCODER: regular (MiniMax's own, in the model folder) or a file's path
case "${TEXT_ENCODER:-regular}" in
  regular) TEXT_ENCODER="";;
  *) [ -f "$TEXT_ENCODER" ] || { echo "TEXT_ENCODER is regular or a safetensors file, and there is no file at $TEXT_ENCODER" >&2; exit 2; };;
esac
# MiniMax H3's own transformer (Turbo, 20-step qualities) is optional (--turbo); without it the engine is FastH3
TURBO_OK=0; ls "$H3_MODEL_DIR"/FL2VA/transformer/*.safetensors >/dev/null 2>&1 && TURBO_OK=1
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
  # reference to video runs on the Ref2VA transformer; the text encoder and the two decoders are FL2VA's own files
  [ -z "${FIRST_FRAME:-}${LAST_FRAME:-}" ] || { echo "references (REF_IMAGES, REF_VIDEOS, REF_AUDIOS) and FIRST_FRAME / LAST_FRAME do not combine: a reference is not a frame of the clip" >&2; exit 2; }
  [ "${ENGINE:-h3}" = h3 ] || { echo "references need MiniMax H3's Ref2VA transformer, not ENGINE=$ENGINE" >&2; exit 2; }
  # MiniMax's limits for one request: 9 pictures, 3 videos, 3 audio clips, 12 files in all
  [ "${#REFS[@]}" -le 9 ] || { echo "at most 9 reference pictures, got ${#REFS[@]}" >&2; exit 2; }
  [ "${#REF_CLIPS[@]}" -le 3 ] || { echo "at most 3 reference videos, got ${#REF_CLIPS[@]}" >&2; exit 2; }
  [ "${#REF_SOUNDS[@]}" -le 3 ] || { echo "at most 3 reference audio clips, got ${#REF_SOUNDS[@]}" >&2; exit 2; }
  [ "$REF_COUNT" -le 12 ] || { echo "at most 12 references in all, got $REF_COUNT" >&2; exit 2; }
  [ "${#REF_SOUNDS[@]}" = 0 ] || [ $(( ${#REFS[@]} + ${#REF_CLIPS[@]} )) -gt 0 ] \
    || { echo "REF_AUDIOS needs a reference picture or video beside it (REF_IMAGES or REF_VIDEOS): the model does not take sound alone" >&2; exit 2; }
  for REF in ${REFS[@]+"${REFS[@]}"}; do [ -s "$REF" ] || { echo "no reference picture at $REF" >&2; exit 2; }; done
  for REF in ${REF_CLIPS[@]+"${REF_CLIPS[@]}"}; do [ -s "$REF" ] || { echo "no reference video at $REF" >&2; exit 2; }; done
  for REF in ${REF_SOUNDS[@]+"${REF_SOUNDS[@]}"}; do [ -s "$REF" ] || { echo "no reference audio clip at $REF" >&2; exit 2; }; done
  case "${REF_VIDEO_SIZE:-match}" in match|full) ;; *) echo "REF_VIDEO_SIZE is match or full, got $REF_VIDEO_SIZE" >&2; exit 2;; esac
  # each reference video and audio clip is 2-15 s, 15 s per kind: the renderer checks before loading any model
  ls "$H3_MODEL_DIR"/Ref2VA/transformer/*.safetensors >/dev/null 2>&1 \
    || { echo "MiniMax H3's Ref2VA transformer is not installed (oneshot-setup.sh --ref2va adds it: 62 GB, 128 GB of memory)" >&2; exit 2; }
  for PART in text_encoder tokenizer processor video_vae audio_vae; do
    [ -e "$H3_MODEL_DIR/Ref2VA/$PART" ] || [ ! -e "$H3_MODEL_DIR/FL2VA/$PART" ] || ln -s "../FL2VA/$PART" "$H3_MODEL_DIR/Ref2VA/$PART"
  done
  ENGINE=h3; TURBO_OK=1
  # a number of passes set by hand: STEPS, as for FastH3 and on a Spark (POINTS, one more, still works and wins)
  if [ -n "${STEPS:-}" ] && [ -z "${POINTS:-}" ]; then
    case "$STEPS" in *[!0-9]*|0) echo "STEPS must be a positive number of passes, got $STEPS" >&2; exit 2;; esac
    POINTS=$((STEPS + 1))
  fi
  # the FL2VA Turbo adapter is not for this transformer: plain 20 passes, or the Ref2VA adapter with REF_TURBO=1
  ADAPTER=none
  if [ "${REF_TURBO:-0}" = 1 ]; then
    ADAPTER="${REF_ADAPTER:-$PREFIX/adapters/lightx2v_ref2v_4step_v0.1_ourlayout.safetensors}"; RECIPE=""
    [ -f "$ADAPTER" ] || { echo "no Ref2VA Turbo adapter at $ADAPTER (oneshot-setup.sh --ref2va adds it)" >&2; exit 2; }
    POINTS="${POINTS:-5}"  # four passes, what the adapter was trained for
    [ "$POINTS" = 5 ] || echo "[tensorfold] the Ref2VA Turbo adapter was trained for 4 passes; running $((POINTS - 1)) as asked"
  fi
  case "${POINTS:-21}" in *[!0-9]*|0|1) echo "POINTS is one more than the passes, 2 at least, got ${POINTS}" >&2; exit 2;; esac
  echo "[tensorfold] reference to video: $(( ${POINTS:-21} - 1 )) passes$([ "$ADAPTER" = none ] || echo " with the Ref2VA Turbo adapter"), ${#REFS[@]} picture(s), ${#REF_CLIPS[@]} video(s), ${#REF_SOUNDS[@]} audio clip(s)"
fi
ENGINE="${ENGINE:-$([ "$TURBO_OK" = 1 ] && echo h3 || echo fasth3)}"; STEPS="${STEPS:-8}"
[ "$ENGINE" = fasth3 ] || [ "$TURBO_OK" = 1 ] || { echo "MiniMax H3's transformer is not installed (oneshot-setup.sh --turbo adds it, 128 GB of memory): use ENGINE=fasth3" >&2; exit 2; }
if [ "$ENGINE" = fasth3 ]; then
  ADAPTER=none; RECIPE=""
  [ -f "$FASTH3_DIR/fastvideo_inference.json" ] || { echo "no FastH3 checkpoint at $FASTH3_DIR (run oneshot-setup.sh, or set FASTH3_DIR)" >&2; exit 2; }
  case "$STEPS" in ''|*[!0-9]*|0) echo "STEPS must be a positive number of passes, got $STEPS" >&2; exit 2;; esac
fi
ADAPTER="${ADAPTER:-$PREFIX/adapters/lightx2v_v1.0_768p_ourlayout.safetensors}"
# sigma points are one more than the passes: 5 passes with the adapter, 20 without
if [ "$ADAPTER" = none ]; then POINTS="${POINTS:-21}"; REVOICE="${REVOICE:-0}"; else POINTS="${POINTS:-6}"; REVOICE="${REVOICE:-20}"; fi
X2_VAE="${X2_VAE:-$PREFIX/adapters/MiniMax-H3-X2-Detail-v1.safetensors}"
PROMPT="${1:-}"; OUT="${2:-outputs/h3.mp4}"
mkdir -p "$(dirname "$OUT")"
export PATH="/opt/homebrew/bin:$PATH" PYTHONPATH="$PREFIX/minimax-h3-mlx"
if [ "${X2:-0}" = 1 ]; then
  W="${WIDTH:-1344}"; H="${HEIGHT:-768}"
  [ $((W % 64)) = 0 ] && [ $((H % 64)) = 0 ] || { echo "with X2=1, WIDTH and HEIGHT must be multiples of 64, got ${W}x${H}" >&2; exit 2; }
  [ -f "$X2_VAE" ] || { echo "no 2x decoder at $X2_VAE (run oneshot-setup.sh, or set X2_VAE)" >&2; exit 2; }
  W=$((W / 2)); H=$((H / 2))
else
  W="${WIDTH:-864}"; H="${HEIGHT:-480}"
fi
ARGS=("$H3_MODEL_DIR" -o "$OUT" --width "$W" --height "$H" --frames "${FRAMES:-124}"
      --seed "${SEED:-0}" --points "$POINTS")
# int8 kernels, except Ref2VA Turbo with a reference video: there they lose its fast motion (REF_INT8=1 forces them)
if [ "$ADAPTER" != none ] && [ "${#REF_CLIPS[@]}" -gt 0 ] && [ "${REF_INT8:-0}" != 1 ]; then
  echo "[tensorfold] reference video with the Turbo adapter: passes in bfloat16, without the int8 kernels"
elif [ "${INT8:-1}" = 0 ]; then
  echo "[tensorfold] INT8=0: passes in bfloat16, without the int8 kernels"
else
  ARGS+=(--int8-mlp --int8-qkv --int8-out)
fi
[ "$ENGINE" != fasth3 ] || ARGS+=(--fasth3 "$FASTH3_DIR" --fasth3-steps "$STEPS")
[ "$REVOICE" = 0 ] || ARGS+=(--revoice "$REVOICE")
[ "${X2:-0}" != 1 ] || ARGS+=(--upscale-vae "$X2_VAE")
if [ "$ENGINE" = fasth3 ]; then
  # which engine runs the passes, and why
  ZIG_ENGINE="${ZIG_ENGINE:-$PREFIX/zig-engine}"; WHY=""
  [ -x "$ZIG_ENGINE/tf-h3-dit" ] && [ -f "$PREFIX/zig_fasth3.py" ] || WHY="the native engine is not installed"
  [ -n "$WHY" ] || sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -Eq 'Apple M([5-9]|[1-9][0-9])' || WHY="this chip has no tensor units"
  [ -n "$WHY" ] || [ -z "${AUDIO_SHIFT:-}${EXTRA:-}" ] || WHY="AUDIO_SHIFT and EXTRA are MLX options"
  [ -n "$WHY" ] || [ -z "$TEXT_ENCODER" ] || WHY="TEXT_ENCODER is an MLX option"
  [ -n "$WHY" ] || [ -z "${LORA:-}" ] || WHY="a LoRA is merged by the MLX engine"
  [ -n "$WHY" ] || [ -z "${LAST_FRAME:-}" ] || WHY="a last frame is held by the MLX engine"
  case "${FASTH3_ENGINE:-auto}" in
    mlx) WHY="FASTH3_ENGINE=mlx";;
    zig) [ -z "$WHY" ] || { echo "FASTH3_ENGINE=zig, but $WHY" >&2; exit 2; };;
    auto) ;;
    *) echo "FASTH3_ENGINE is zig or mlx, got $FASTH3_ENGINE" >&2; exit 2;;
  esac
  if [ -z "$WHY" ]; then
    ZARGS=("$H3_MODEL_DIR" --fasth3 "$FASTH3_DIR" --engine "$ZIG_ENGINE" -o "$OUT" --width "$W" --height "$H"
           --frames "${FRAMES:-124}" --seed "${SEED:-0}" --steps "$STEPS")
    [ "${X2:-0}" != 1 ] || ZARGS+=(--upscale-vae "$X2_VAE")
    [ -z "${CROP:-}" ] || ZARGS+=(--crop "$CROP")
    [ -z "${FIRST_FRAME:-}" ] || ZARGS+=(--first-frame "$FIRST_FRAME")
    if [ -n "${PROMPT_FILE:-}" ]; then ZARGS+=(--prompt-file "$PROMPT_FILE"); else ZARGS+=(--prompt "$PROMPT"); fi
    exec "$PREFIX/venv/bin/python" "$PREFIX/zig_fasth3.py" "${ZARGS[@]}"
  fi
  # no tensor units: MLX would hold bf16 (64 GiB); fasth3_q8.py's 8-bit weights halve it (FASTH3_Q8=0|1 forces)
  Q8="${FASTH3_Q8:-auto}"
  if [ "$Q8" = auto ]; then Q8=0; [ -f "$FASTH3_DIR/transformer-q8/q8.json" ] && [ "$WHY" = "this chip has no tensor units" ] && Q8=1; fi
  if [ "$Q8" = 1 ]; then
    [ -f "$FASTH3_DIR/transformer-q8/q8.json" ] || { echo "no 8-bit weights at $FASTH3_DIR/transformer-q8 (run oneshot-setup.sh, or $PREFIX/venv/bin/python $PREFIX/fasth3_q8.py convert $FASTH3_DIR)" >&2; exit 2; }
    ARGS+=(--fasth3-q8 --vsa-impl simd); WHY="$WHY, 8-bit weights"
  fi
  echo "[tensorfold] engine: mlx ($WHY)"
fi
[ "$ADAPTER" = none ] || ARGS+=(--lora "$ADAPTER")
# style LoRAs from the hub, one PATH:STRENGTH per line, already in the layout --lora reads (scripts/convert-lora.py)
if [ -n "${LORA:-}" ]; then while IFS= read -r ONE; do [ -z "$ONE" ] || ARGS+=(--lora "$ONE"); done <<<"$LORA"; fi
[ -z "${AUDIO_SHIFT:-}" ] || ARGS+=(--audio-shift "$AUDIO_SHIFT")
[ -z "${FIRST_FRAME:-}" ] || ARGS+=(--first-frame "$FIRST_FRAME")
[ -z "${LAST_FRAME:-}" ] || { [ -s "$LAST_FRAME" ] || { echo "no last frame at $LAST_FRAME" >&2; exit 2; }; ARGS+=(--last-frame "$LAST_FRAME"); }
[ -z "$TEXT_ENCODER" ] || ARGS+=(--text-encoder "$TEXT_ENCODER")
for REF in ${REFS[@]+"${REFS[@]}"}; do ARGS+=(--ref-image "$REF"); done
for REF in ${REF_CLIPS[@]+"${REF_CLIPS[@]}"}; do ARGS+=(--ref-video "$REF"); done
for REF in ${REF_SOUNDS[@]+"${REF_SOUNDS[@]}"}; do ARGS+=(--ref-audio "$REF"); done
[ "${#REF_CLIPS[@]}" = 0 ] || [ "${REF_VIDEO_SOUND:-0}" != 1 ] || ARGS+=(--ref-video-sound)
[ "${#REF_CLIPS[@]}" = 0 ] || [ "${REF_VIDEO_SIZE:-match}" = match ] || ARGS+=(--ref-video-size "$REF_VIDEO_SIZE")
if [ -n "${PROMPT_FILE:-}" ]; then ARGS+=(--prompt-file "$PROMPT_FILE"); else ARGS+=(--prompt "$PROMPT"); fi
[ -z "${CROP:-}" ] || ARGS+=(--crop "$CROP")
# shellcheck disable=SC2086
"$PREFIX/venv/bin/python" "${H3_GENERATE:-$PREFIX/h3_generate.py}" "${ARGS[@]}" $RECIPE ${EXTRA:-}
# thin few-step takes: measured bass lifted up to 6 dB + limiter, video not re-encoded (AUDIO_EQ=off, AUDIO_BASS=dB)
WAV="${OUT%.*}.wav"
if [ "$ADAPTER" != none ] && [ "${AUDIO_EQ:-on}" != off ] && [ -s "$WAV" ]; then
  GAIN="${AUDIO_BASS:-$("$PREFIX/venv/bin/python" - "$WAV" "${AUDIO_BASS_TARGET:-0.5}" <<'PY'
import sys, wave
import numpy as np
with wave.open(sys.argv[1]) as handle:
    rate = handle.getframerate()
    audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32)
    audio = audio.reshape(-1, handle.getnchannels()).mean(axis=1)
power = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2
freq = np.fft.rfftfreq(len(audio), 1 / rate)
bass = power[(freq >= 120) & (freq < 300)].sum()
mid = power[(freq >= 300) & (freq < 4000)].sum()
ratio = bass / max(mid, 1e-20)
gain = float(np.clip(10 * np.log10(float(sys.argv[2]) / max(ratio, 1e-6)), 0.0, 6.0))
print(f"{gain:.1f}")
PY
)}"
  if [ "$(printf '%.0f' "$(echo "$GAIN * 10" | bc)")" -ge 10 ]; then
    FILTER="bass=g=$GAIN:f=250:w=0.6,alimiter=limit=0.95:level=disabled"
    ffmpeg -v error -y -i "$OUT" -i "$WAV" -map 0:v -map 1:a -c:v copy -af "$FILTER" -c:a aac -b:a 256k "${OUT%.*}.eq.mp4"
    mv "${OUT%.*}.eq.mp4" "$OUT"
    echo "[tensorfold] thin take: bass lifted by $GAIN dB"
  else
    echo "[tensorfold] audio left as rendered (bass is full)"
  fi
fi
