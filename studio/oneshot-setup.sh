#!/usr/bin/env bash
# One-shot Mac install: Qwen-Image-2.1-Turbo + FastH3 into its own venv ($PREFIX). Options and sizes: README.
set -euo pipefail
# Engine payload order: ./payload -> GHCR carrier image -> release download -> git at the pinned commit.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# On Linux hand over to the DGX Spark installer with the options it knows; Mac-only ones are dropped with a note.
if [ "$(uname -s)" = Linux ]; then
  SPARK_ARGS=()
  for arg in "$@"; do
    case "$arg" in
      --image-only|--no-render|--verify|--motion|--upgrade|--flux3|--loras|--loras=*|--no-loras|--lora-bundle=*) SPARK_ARGS+=("$arg");;
      --fasth3) ;;
      --turbo|--app|--ref2va) echo "note: $arg is for the Mac build and does nothing on a DGX Spark";;
      *) echo "unknown option $arg" >&2; exit 2;;
    esac
  done
  echo "Linux detected: installing the DGX Spark (CUDA) build"
  exec bash "$HERE/oneshot-setup-spark.sh" ${SPARK_ARGS[@]+"${SPARK_ARGS[@]}"}
fi
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
H3_MODEL_DIR="${H3_MODEL_DIR:-$HOME/h3-models/MiniMax-H3}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$HOME/qwen-models/Qwen-Image-2.1-Turbo}"
FASTH3_DIR="${FASTH3_DIR:-$HOME/h3-models/FastH3-8-Step-V2}"
FASTH3_REPO="FastVideo/FastVideo-FastH3-8-Step-V2"
IMAGE="${IMAGE:-ghcr.io/drowzeys/keys-tensorfold-studio:2.8.0}"
# the carrier under its name up to 2.2, with the same Mac payload
IMAGE_BEFORE="ghcr.io/drowzeys/keys-mac-tensorfold-studio:2.2"
TF_REPO="https://github.com/drowzeys/TensorFold.git"
TF_COMMIT="a2b592fef5366bf16d4729651b6549094981d3fe"
REF_REPO="https://github.com/mrbizarro/minimax-h3-mlx.git"
REF_COMMIT="79190205258454b43e6c9e50e577de234222419c"
# the native engine for FastH3: TensorFold 1.0.4's Zig + Metal runtime with the H3 family, from the same fork
ZIG_COMMIT="b7e309e042e48b72e75e9fb87342df4e9bf22f37"
ZIG_ENGINE_URL="https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3/releases/download/v2.3/zig-engine-b7e309e0-macos-arm64.tar.gz"
ZIG_ENGINE_SHA256="d142b1d3c57d3c5bfc4d2a0d4f83ca5bf91e8091ac6357cda52759a250fd391c"
MFLUX_REPO="https://github.com/mflux-community/mflux.git"
MFLUX_COMMIT="add5164e62c07cbcc9aec7a95c2e19c75605880c"
H3_ADAPTER_NAME="lightx2v_v1.0_768p_ourlayout.safetensors"
H3_ADAPTER_URL="https://github.com/mrbizarro/Phosphene/releases/download/weights-ltx25-v1/$H3_ADAPTER_NAME"
H3_ADAPTER_SHA256="d51d626fe0845da7e5845a47c323cf3f29086d44d24cb1a4b980882488746197"
# LightX2V's 4-step adapter for the Ref2VA transformer, rewritten at install into the layout --lora reads
REF_ADAPTER_REPO="lightx2v/Minimax-h3-Turbo"
REF_ADAPTER_SOURCE="minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
REF_ADAPTER_SHA256="9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99"
REF_ADAPTER_NAME="lightx2v_ref2v_4step_v0.1_ourlayout.safetensors"
X2_REPO="speach1sdef178/MiniMax-H3-X2-Detail-VAE"
X2_NAME="MiniMax-H3-X2-Detail-v1.safetensors"
X2_SHA256="2296840f4acedcaa976688e7d7b97f7bf570b136e400385d3f46224011897aac"
MODE=""; VIDEO=1; TURBO=0; APP=0; REF2VA=0; MOTION=0; UPGRADE=0; LORAS=""; IMAGE_EDIT=0; FLUX3=0
for arg in "$@"; do
  case "$arg" in
    --image-only) VIDEO=0;;
    --fasth3) ;;  # FastH3 is the default video engine since 2.1; the flag is kept so older commands still run
    --turbo) TURBO=1;;
    --ref2va) REF2VA=1;;
    --motion) MOTION=1;;
    --flux3) FLUX3=1;;
    --image-edit) IMAGE_EDIT=1;;
    --app) APP=1;;
    --verify|--no-render) MODE="$arg";;
    --upgrade) UPGRADE=1; MODE="--no-render"; MOTION=1;;
    --loras) LORAS=all;; --loras=*) LORAS="${arg#--loras=}";; --no-loras) LORAS=none;;
    --lora-bundle=*) LORA_BUNDLE="${arg#--lora-bundle=}";;
    *) echo "unknown option $arg" >&2; exit 2;;
  esac
done

die() { echo "FATAL: $*" >&2; exit 1; }
ok()  { echo "  ✓ $*"; }
step(){ echo; echo "==> $*"; }
# a weight that is not on disk: --verify stops there; --upgrade says so and goes on without fetching it
absent() { [ "$MODE" != "--verify" ] || die "$1"; [ "$UPGRADE" = 1 ] || return 1; echo "  ! $1: --upgrade fetches no weights (run oneshot-setup.sh without it to add them)"; }

step "Preflight"
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "Apple silicon macOS only"
CHIP=$(sysctl -n machdep.cpu.brand_string); RAM=$(( $(sysctl -n hw.memsize) / 1073741824 ))
ok "$CHIP, macOS $(sw_vers -productVersion), ${RAM} GB"
# --upgrade moves an install to this version (engine, scripts, Motion Previs env); kind and studio folder are kept.
INSTALLED="$(cat "$PREFIX/VERSION" 2>/dev/null || echo "2.6 or earlier")"
if [ "$UPGRADE" = 1 ]; then
  [ -x "$PREFIX/venv/bin/python" ] || die "nothing to upgrade at $PREFIX: run oneshot-setup.sh without --upgrade"
  ls "$H3_MODEL_DIR"/FL2VA/text_encoder/*.safetensors >/dev/null 2>&1 || VIDEO=0
  ok "upgrade: installed $INSTALLED -> $(cat "$HERE/VERSION")$([ "$VIDEO" = 1 ] || echo " (image-only install)")"
else ok "version $(cat "$HERE/VERSION")$([ ! -f "$PREFIX/VERSION" ] || echo " (installed: $INSTALLED)")"; fi
# an install that already holds MiniMax H3's transformer and the Turbo adapter keeps them
[ "$VIDEO" = 0 ] || { ls "$H3_MODEL_DIR"/FL2VA/transformer/*.safetensors >/dev/null 2>&1 && [ -f "$PREFIX/adapters/$H3_ADAPTER_NAME" ] && TURBO=1; } || true
# Peak memory: FastH3 64 GiB on MLX bf16 (45 GiB 8-bit), 103 GiB while a Turbo adapter merges; keep Ref2VA if present.
[ "$VIDEO" = 0 ] || { ls "$H3_MODEL_DIR"/Ref2VA/transformer/*.safetensors >/dev/null 2>&1 && REF2VA=1; } || true
FAST_NEED=64
if [ "$VIDEO" = 0 ]; then NEED=48; elif [ "$TURBO" = 1 ] || [ "$REF2VA" = 1 ]; then NEED=128; else NEED=$FAST_NEED; fi
[ "$RAM" -ge "$NEED" ] || die "needs ${NEED} GB+ unified memory for this install$([ "$TURBO" = 1 ] && [ "$RAM" -ge "$FAST_NEED" ] && echo " (without --turbo it needs ${FAST_NEED} GB)") (measured on 256 GB only)"
case "$CHIP" in *M5*) ok "M5: int8 kernels on the tensor units";;
  *) echo "  ! $CHIP is not M5: the int8 kernels need Metal 4 tensor operations; without them the engine runs bfloat16 and the README numbers do not apply";; esac
export PATH="/opt/homebrew/bin:$PATH"
command -v uv >/dev/null || die "uv required: brew install uv"
command -v python3.11 >/dev/null || die "python3.11 required: brew install python@3.11"
command -v git >/dev/null || die "git required"
[ "$VIDEO" = 0 ] || { command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null; } || die "ffmpeg (with ffprobe) required: brew install ffmpeg"

fetch_ghcr() {
  command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 || return 1
  local image
  for image in "$IMAGE" "$IMAGE_BEFORE"; do
    docker pull -q "$image" >/dev/null 2>&1 || continue
    echo "  payload from GHCR $image"
    docker run --rm -v "$HERE":/out "$image" cp -a /payload/. /out/payload/
    return
  done
  return 1
}

step "TensorFold 0.6.5 + H3 and Qwen-Image families @ ${TF_COMMIT:0:8}, mflux @ ${MFLUX_COMMIT:0:8} (own venv at $PREFIX)"
HAS_ENGINE='import inspect, tensorfold.families.qwen_image.sampler, mflux.models.qwen21.qwen21_initializer; from tensorfold.families.h3.vae_video import load_video_decoder as d; assert "upscale_decoder" in inspect.signature(d).parameters; from tensorfold.families.h3.sampler import denoise as n; assert "audio_shift" in inspect.signature(n).parameters; from tensorfold.families.h3.sampler import revoice, default_gates; import tensorfold.kernels.minimax.h3.v1.attention_tiles_int8; from tensorfold.families.h3 import packing; assert hasattr(packing, "reference_layout")'
if [ ! -x "$PREFIX/venv/bin/python" ] || ! "$PREFIX/venv/bin/python" -c "$HAS_ENGINE" 2>/dev/null; then
  [ "$MODE" = "--verify" ] && die "TensorFold with the H3 and Qwen-Image families is not installed at $PREFIX"
  mkdir -p "$HERE/payload" "$PREFIX"
  ( cd "$HERE/payload" 2>/dev/null && shasum -a 256 -c SHA256SUMS >/dev/null 2>&1 ) \
    || { rm -f "$HERE"/payload/tensorfold-*.whl "$HERE"/payload/SHA256SUMS; fetch_ghcr || echo "  no carrier payload; installing from git"; }
  [ -x "$PREFIX/venv/bin/python" ] || uv venv -q -p "$(command -v python3.11)" "$PREFIX/venv"
  if ls "$HERE"/payload/tensorfold-*.whl >/dev/null 2>&1; then
    ( cd "$HERE/payload" && shasum -a 256 -c SHA256SUMS >/dev/null ) || die "carrier payload checksum mismatch"
    ok "carrier payload checksums verified"
    TF_SPEC=("$HERE"/payload/tensorfold-*.whl)
  else
    TF_SPEC=("tensorfold @ git+$TF_REPO@$TF_COMMIT")
  fi
  uv pip install -q --reinstall-package tensorfold -p "$PREFIX/venv/bin/python" -r "$HERE/requirements.lock" \
    "${TF_SPEC[@]}" "mflux @ git+$MFLUX_REPO@$MFLUX_COMMIT"
fi
cp "$HERE/h3_generate.py" "$HERE/qwen_image_generate.py" "$HERE/zig_fasth3.py" "$HERE/fasth3_q8.py" "$HERE/h3_text.py" "$HERE/h3_media.py" "$HERE/h3_condition.py" "$HERE/h3_decode.py" "$PREFIX/"
# speech-to-text for the QA gate of the Wasserman Sharp Spec (app/qa.py); without it that gate is shown as skipped
uv pip install -q -p "$PREFIX/venv/bin/python" faster-whisper || echo "  note: faster-whisper was not installed; the QA gate's speech check will be skipped"
"$PREFIX/venv/bin/python" - <<'EOF' || die "the installed TensorFold lacks the H3 or Qwen-Image family"
import importlib.metadata as m
import mlx.core as mx
from tensorfold.families import families
from tensorfold.kernels.minimax.h3.v1 import mlp_int8
assert {"minimax_h3", "qwen_image_21"} <= set(families())
print(f"  ✓ tensorfold {m.version('tensorfold')} with the H3 and Qwen-Image families, mflux {m.version('mflux')}, mlx {mx.__version__}")
print("  ✓ int8 tensor-unit kernels available" if mlp_int8.available() else "  ! int8 kernels unavailable on this GPU: bfloat16 only")
EOF

step "Qwen-Image-2.1-Turbo weights (33 GB) -> $QWEN_MODEL_DIR"
if [ ! -f "$QWEN_MODEL_DIR/transformer/diffusion_pytorch_model-00002-of-00002.safetensors" ] && ! absent "no Qwen-Image-2.1-Turbo weights at $QWEN_MODEL_DIR"; then
  echo "  Qwen-Image-2.1-Turbo is under the Qwen Research License Agreement: NON-COMMERCIAL use only. Read it first:"
  echo "  https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE"
  BASE="$(dirname "$QWEN_MODEL_DIR")/Qwen-Image-2.1"
  if [ -f "$BASE/text_encoder/model.safetensors" ] && [ -d "$BASE/vae" ]; then
    # an install from before 2.3 has the base model: Turbo shares its encoder and decoder, so fetch the transformer only
    mkdir -p "$QWEN_MODEL_DIR"
    for part in text_encoder vae processor scheduler; do [ -e "$QWEN_MODEL_DIR/$part" ] || ln -s "../$(basename "$BASE")/$part" "$QWEN_MODEL_DIR/$part"; done
    "$PREFIX/venv/bin/hf" download Qwen/Qwen-Image-2.1-Turbo model_index.json LICENSE README.md transformer/config.json \
      transformer/diffusion_pytorch_model-00001-of-00002.safetensors transformer/diffusion_pytorch_model-00002-of-00002.safetensors \
      transformer/diffusion_pytorch_model.safetensors.index.json --local-dir "$QWEN_MODEL_DIR"
  else
    "$PREFIX/venv/bin/hf" download Qwen/Qwen-Image-2.1-Turbo --local-dir "$QWEN_MODEL_DIR"
  fi
fi
grep -q sample_sigmas "$QWEN_MODEL_DIR/model_index.json" || die "$QWEN_MODEL_DIR is not Qwen-Image-2.1-Turbo (its model_index.json has no saved schedule)"
ok "Qwen-Image-2.1-Turbo at $QWEN_MODEL_DIR ($(du -shL "$QWEN_MODEL_DIR" | cut -f1))"

if [ "$IMAGE_EDIT" = 1 ]; then
  # the 40-step base model (what image LoRAs train on): fetch its transformer, link Turbo's encoder, coder, processor
  BASE="${QWEN_BASE_DIR:-$(dirname "$QWEN_MODEL_DIR")/Qwen-Image-2.1}"
  step "Qwen-Image-2.1 base transformer (13 GB) -> $BASE"
  if [ ! -f "$BASE/transformer/diffusion_pytorch_model-00002-of-00002.safetensors" ] && ! absent "no Qwen-Image-2.1 base transformer at $BASE"; then
    echo "  Qwen-Image-2.1 is under the Qwen Research License Agreement: NON-COMMERCIAL use only. Read it first:"
    echo "  https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE"
    FREE=$(df -g "$(dirname "$BASE")" | awk 'NR==2{print $4}')
    [ "${FREE:-0}" -ge 20 ] || die "only ${FREE} GB free where $BASE goes: the base transformer is 13 GB"
    mkdir -p "$BASE"
    for part in text_encoder vae processor; do [ -e "$BASE/$part" ] || ln -s "$QWEN_MODEL_DIR/$part" "$BASE/$part"; done
    "$PREFIX/venv/bin/hf" download Qwen/Qwen-Image-2.1 model_index.json LICENSE scheduler/scheduler_config.json transformer/config.json \
      transformer/diffusion_pytorch_model-00001-of-00002.safetensors transformer/diffusion_pytorch_model-00002-of-00002.safetensors \
      transformer/diffusion_pytorch_model.safetensors.index.json --local-dir "$BASE"
  fi
  [ ! -d "$BASE/transformer" ] || ok "Qwen-Image-2.1 base model at $BASE: BASE=1 bash scripts/edit.sh, and the 40 steps choice of Edit a picture"
fi

if [ "$VIDEO" = 1 ]; then
  step "Text encoder, audio decoder and MP4 writer for H3: minimax-h3-mlx @ ${REF_COMMIT:0:8}"
  # a prefix layered on another install can hold minimax-h3-mlx as a folder that is not a clone: kept when it imports
  if [ -d "$PREFIX/minimax-h3-mlx" ] && [ ! -d "$PREFIX/minimax-h3-mlx/.git" ] && [ -n "$(ls -A "$PREFIX/minimax-h3-mlx")" ]; then
    PYTHONPATH="$PREFIX/minimax-h3-mlx" "$PREFIX/venv/bin/python" -c "import minimax_h3_mlx.text_encoder, minimax_h3_mlx.media" \
      || die "$PREFIX/minimax-h3-mlx is not a clone and does not import: move it aside and run this again"
    ok "minimax-h3-mlx at $PREFIX/minimax-h3-mlx (not a clone of ours: kept as it is)"
  else
  if [ ! -d "$PREFIX/minimax-h3-mlx/.git" ]; then
    [ "$MODE" = "--verify" ] && die "minimax-h3-mlx is not at $PREFIX/minimax-h3-mlx"
    git clone -q "$REF_REPO" "$PREFIX/minimax-h3-mlx"
  fi
  git -C "$PREFIX/minimax-h3-mlx" fetch -q origin "$REF_COMMIT" 2>/dev/null || true
  git -C "$PREFIX/minimax-h3-mlx" checkout -q "$REF_COMMIT" || die "cannot check out minimax-h3-mlx $REF_COMMIT"
  PYTHONPATH="$PREFIX/minimax-h3-mlx" "$PREFIX/venv/bin/python" -c "import minimax_h3_mlx.text_encoder, minimax_h3_mlx.media" \
    || die "minimax-h3-mlx does not import"
  ok "minimax-h3-mlx $(git -C "$PREFIX/minimax-h3-mlx" rev-parse --short HEAD)"
  fi

  # FastH3 replaces MiniMax H3's transformer and keeps the rest of its pipeline: the text encoder and both decoders
  if [ "$TURBO" = 1 ]; then step "MiniMax H3 weights (FL2VA partition, 134 GB) -> $H3_MODEL_DIR"; SKIP=()
  else step "MiniMax H3 text encoder and decoders (FL2VA without its transformer, 72 GB) -> $H3_MODEL_DIR"; SKIP=(--exclude "FL2VA/transformer/*.safetensors"); fi
  h3_ok() { ls "$H3_MODEL_DIR"/FL2VA/text_encoder/model-00014-of-00014.safetensors "$H3_MODEL_DIR"/FL2VA/audio_vae/model.safetensors \
              "$H3_MODEL_DIR"/FL2VA/video_vae/config.json "$H3_MODEL_DIR"/FL2VA/transformer/config.json >/dev/null 2>&1 \
            && { [ "$TURBO" = 0 ] || ls "$H3_MODEL_DIR"/FL2VA/transformer/model-00013-of-00013.safetensors >/dev/null 2>&1; }; }
  if ! h3_ok && ! absent "no H3 weights at $H3_MODEL_DIR"; then
    echo "  MiniMax H3 is under the MiniMax H3 Community License, which limits where it may be used. Read it first:"
    echo "  https://huggingface.co/MiniMaxAI/MiniMax-H3"
    "$PREFIX/venv/bin/hf" download MiniMaxAI/MiniMax-H3 --include "FL2VA/*" --include model_index.json \
      --include LICENSE --include README.md ${SKIP[@]+"${SKIP[@]}"} --local-dir "$H3_MODEL_DIR"
    h3_ok || die "the MiniMax H3 download is incomplete at $H3_MODEL_DIR"
  fi
  ok "FL2VA at $H3_MODEL_DIR ($(du -sh "$H3_MODEL_DIR/FL2VA" | cut -f1))"

  if [ "$REF2VA" = 1 ]; then
    # Ref2VA is the same pipeline with another transformer: its text encoder and decoders are FL2VA's files, linked
    step "MiniMax H3 Ref2VA transformer (reference pictures to video, 62 GB) -> $H3_MODEL_DIR/Ref2VA"
    if ! ls "$H3_MODEL_DIR"/Ref2VA/transformer/model-00013-of-00013.safetensors "$H3_MODEL_DIR"/Ref2VA/model_index.json >/dev/null 2>&1 && ! absent "no Ref2VA transformer at $H3_MODEL_DIR/Ref2VA"; then
      "$PREFIX/venv/bin/hf" download MiniMaxAI/MiniMax-H3 --include "Ref2VA/transformer/*" --include "Ref2VA/model_index.json" \
        --local-dir "$H3_MODEL_DIR"
      ls "$H3_MODEL_DIR"/Ref2VA/transformer/model-00013-of-00013.safetensors >/dev/null 2>&1 || die "the Ref2VA download is incomplete at $H3_MODEL_DIR/Ref2VA"
    fi
    for part in text_encoder tokenizer processor video_vae audio_vae; do
      [ -e "$H3_MODEL_DIR/Ref2VA/$part" ] || [ "$MODE" = "--verify" ] || ln -s "../FL2VA/$part" "$H3_MODEL_DIR/Ref2VA/$part"
      [ -e "$H3_MODEL_DIR/Ref2VA/$part" ] || die "$H3_MODEL_DIR/Ref2VA/$part is missing (a link to ../FL2VA/$part)"
    done
    REF_ADAPTER="$PREFIX/adapters/$REF_ADAPTER_NAME"
    if [ ! -f "$REF_ADAPTER" ]; then
      if [ "$MODE" = "--verify" ] || [ "$UPGRADE" = 1 ]; then echo "  ! no Ref2VA Turbo adapter at $REF_ADAPTER: reference clips run their 20 passes"; else
        mkdir -p "$PREFIX/adapters/h3-ref2v"
        "$PREFIX/venv/bin/hf" download "$REF_ADAPTER_REPO" "$REF_ADAPTER_SOURCE" --local-dir "$PREFIX/adapters/h3-ref2v" >/dev/null
        echo "$REF_ADAPTER_SHA256  $PREFIX/adapters/h3-ref2v/$REF_ADAPTER_SOURCE" | shasum -a 256 -c - >/dev/null || die "Ref2VA Turbo adapter checksum mismatch"
        "$PREFIX/venv/bin/python" "$HERE/scripts/convert-h3-adapter.py" "$PREFIX/adapters/h3-ref2v/$REF_ADAPTER_SOURCE" "$REF_ADAPTER.partial.safetensors" | sed 's/^/  /'
        mv "$REF_ADAPTER.partial.safetensors" "$REF_ADAPTER"; rm -f "$PREFIX/adapters/h3-ref2v/$REF_ADAPTER_SOURCE"
      fi
    fi
    [ ! -f "$REF_ADAPTER" ] || ok "$REF_ADAPTER_NAME (Ref2VA Turbo, 4 passes: REF_TURBO=1)"
    ok "Ref2VA at $H3_MODEL_DIR/Ref2VA ($(du -sh "$H3_MODEL_DIR/Ref2VA/transformer" | cut -f1)): REF_IMAGES=face.jpg bash scripts/video.sh \"Use <Picture 1> as the person. ...\" out.mp4"
  fi

  if [ "$TURBO" = 1 ]; then
    step "Video Turbo adapter (lightx2v v1.0, runner layout, 1.96 GB)"
    H3_ADAPTER="$PREFIX/adapters/$H3_ADAPTER_NAME"
    if [ ! -f "$H3_ADAPTER" ] && ! absent "no video Turbo adapter at $H3_ADAPTER"; then
      curl -L --fail --retry 3 -C - -o "$H3_ADAPTER.partial" "$H3_ADAPTER_URL"
      echo "$H3_ADAPTER_SHA256  $H3_ADAPTER.partial" | shasum -a 256 -c - >/dev/null || die "video Turbo adapter checksum mismatch"
      mv "$H3_ADAPTER.partial" "$H3_ADAPTER"
    fi
    [ ! -f "$H3_ADAPTER" ] || ok "$H3_ADAPTER_NAME"
  fi

  step "2x video decoder (MiniMax-H3-X2-Detail-VAE, 5.2 GB)"
  X2_VAE="$PREFIX/adapters/$X2_NAME"
  if [ ! -f "$X2_VAE" ] && ! absent "no 2x video decoder at $X2_VAE"; then
    "$PREFIX/venv/bin/hf" download "$X2_REPO" "$X2_NAME" LICENSE NOTICE --local-dir "$PREFIX/adapters/h3-x2" >/dev/null
    echo "$X2_SHA256  $PREFIX/adapters/h3-x2/$X2_NAME" | shasum -a 256 -c - >/dev/null || die "2x video decoder checksum mismatch"
    ln -sf "h3-x2/$X2_NAME" "$X2_VAE"
  fi
  [ ! -e "$X2_VAE" ] || ok "$X2_NAME"

  if true; then
    step "FastH3 8-Step V2 transformer (70 GB) -> $FASTH3_DIR"
    if ! ls "$FASTH3_DIR"/transformer/diffusion_pytorch_model-00014-of-00014.safetensors >/dev/null 2>&1 \
       && [ ! -f "$FASTH3_DIR/transformer-q8/q8.json" ] && ! absent "no FastH3 transformer at $FASTH3_DIR"; then
      echo "  FastH3 is a derivative of MiniMax H3 under the MiniMax H3 Community License. Read it first:"
      echo "  https://huggingface.co/$FASTH3_REPO"
      "$PREFIX/venv/bin/hf" download "$FASTH3_REPO" --include "transformer/*" --include "fastvideo_inference.json" \
        --include "scheduler/*" --include "audio_scheduler/*" --include LICENSE --include NOTICE --include README.md \
        --local-dir "$FASTH3_DIR"
    fi
    ok "FastH3 at $FASTH3_DIR ($(du -shL "$FASTH3_DIR/transformer" | cut -f1)): bash scripts/fast.sh \"a prompt\" out.mp4"

    case "$CHIP" in *M[5-9]*) ;; *)
      step "8-bit FastH3 weights for a chip without tensor units (37 GB, written once) -> $FASTH3_DIR/transformer-q8"
      if [ ! -f "$FASTH3_DIR/transformer-q8/q8.json" ] && ! absent "no 8-bit FastH3 weights at $FASTH3_DIR/transformer-q8"; then
        PYTHONPATH="$PREFIX/minimax-h3-mlx" "$PREFIX/venv/bin/python" "$PREFIX/fasth3_q8.py" convert "$FASTH3_DIR" | grep -E "wrote|block [0-9]*0/" | sed 's/^/  /'
        [ -f "$FASTH3_DIR/transformer-q8/q8.json" ] || die "the 8-bit conversion did not finish"
      fi
      # Q8_ONLY=1: the 65 GB of bfloat16 weights are not read again on this chip
      [ "${Q8_ONLY:-0}" != 1 ] || [ "$MODE" = "--verify" ] || [ ! -f "$FASTH3_DIR/transformer-q8/q8.json" ] || rm -f "$FASTH3_DIR"/transformer/diffusion_pytorch_model-*.safetensors
      ok "transformer-q8 ($(du -sh "$FASTH3_DIR/transformer-q8" | cut -f1)): FastH3 runs on the MLX engine with 8-bit weights"
      ;; esac

    step "Native engine for FastH3: TensorFold 1.0.4 Zig + Metal runtime @ ${ZIG_COMMIT:0:8} -> $PREFIX/zig-engine"
    ZIG_ENGINE="$PREFIX/zig-engine"
    if [ "$(cat "$ZIG_ENGINE/COMMIT" 2>/dev/null)" != "$ZIG_COMMIT" ] || [ ! -x "$ZIG_ENGINE/tf-h3-dit" ]; then
      if [ "$MODE" = "--verify" ]; then
        echo "  ! the native engine is not installed: FastH3 runs on the MLX engine"
      else
        # carrier payload first, then the release download, then a build from source, else the MLX engine stays in use
        engine_ok() { [ "$(cat "$HERE/payload/zig-engine/COMMIT" 2>/dev/null)" = "$ZIG_COMMIT" ] && ( cd "$HERE/payload/zig-engine" && shasum -a 256 -c SHA256SUMS >/dev/null 2>&1 ); }
        engine_ok || { mkdir -p "$HERE/payload"; fetch_ghcr >/dev/null 2>&1 || true; }
        # no Docker on this Mac: the same engine as a checksummed release download
        if ! engine_ok; then
          T="$(mktemp -d)"
          if curl -fsSL -o "$T/engine.tgz" "$ZIG_ENGINE_URL" && [ "$(shasum -a 256 "$T/engine.tgz" | cut -d' ' -f1)" = "$ZIG_ENGINE_SHA256" ]; then
            echo "  prebuilt engine from $ZIG_ENGINE_URL"
            rm -rf "$HERE/payload/zig-engine"; tar -xzf "$T/engine.tgz" -C "$HERE/payload"
          fi
          rm -rf "$T"
        fi
        if ! engine_ok && command -v zig >/dev/null && xcrun -sdk macosx metal -v >/dev/null 2>&1; then
          echo "  no carrier payload for the engine; building from source"
          ZIG_COMMIT="$ZIG_COMMIT" TF_REPO="$TF_REPO" bash "$HERE/scripts/build-zig-engine.sh" | sed 's/^/  /' || true
        fi
        if engine_ok; then
          rm -rf "$ZIG_ENGINE"; cp -R "$HERE/payload/zig-engine" "$ZIG_ENGINE"; chmod +x "$ZIG_ENGINE/tf-h3-dit"
          ln -sfn ../h3_generate.py "$ZIG_ENGINE/h3_generate_dev.py"
        else
          echo "  ! no native engine (no carrier payload or release download, and a source build needs zig 0.17 and Xcode's Metal toolchain): FastH3 runs on the MLX engine"
        fi
      fi
    fi
    if [ -x "$ZIG_ENGINE/tf-h3-dit" ]; then
      case "$CHIP" in *M[5-9]*) ok "tf-h3-dit @ $(cut -c1-8 "$ZIG_ENGINE/COMMIT"): FastH3 runs on the native engine";;
        *) ok "tf-h3-dit installed, but $CHIP has no tensor units: FastH3 runs on the MLX engine";; esac
    fi
  fi
fi

if [ "$MOTION" = 1 ] || [ -x "$PREFIX/motion-venv/bin/python" ]; then
  # Motion Previs (port of Sam Wasserman's Motion Previs Studio analysis) in its own env; uninstallable layers skipped
  step "Motion Previs: MediaPipe, OpenCV, ONNX Runtime and the pose and depth models (0.5 GB) -> $PREFIX/motion-venv"
  PREFIX="$PREFIX" bash "$HERE/scripts/setup-motion.sh" $([ "$MODE" = "--verify" ] && echo --verify) | sed 's/^/  /' \
    || echo "  ! Motion Previs is not fully set up: its tab says which layers are not available on this machine"
fi

[ "$MODE" = "--verify" ] || cp "$HERE/VERSION" "$PREFIX/VERSION"
export PREFIX H3_MODEL_DIR QWEN_MODEL_DIR FASTH3_DIR
if [ "$APP" = 1 ] && [ "$MODE" != "--verify" ]; then
  step "One-click app"
  DESKTOP=1 bash "$HERE/scripts/make-app.sh" | sed 's/^/  /'
fi

if [ "$UPGRADE" = 1 ]; then
  step "Upgraded: $INSTALLED -> $(cat "$PREFIX/VERSION"). Checking the install (restart the app to serve the new pages)"
  exec bash "$HERE/oneshot-setup.sh" --verify $([ "$VIDEO" = 1 ] || echo --image-only)
fi
# Essential LoRA bundle: --loras=GROUPS takes some groups, --lora-bundle=PATH installs offline, --no-loras skips it.
if [ "$LORAS" != none ] && [ "$VIDEO" = 1 ] && [ "$MODE" != "--verify" ]; then
  echo; echo "== Essential LoRA bundle${LORA_BUNDLE:+ from $LORA_BUNDLE}"
  PREFIX="$PREFIX" H3_MODEL_DIR="${H3_MODEL_DIR:-$HOME/h3-models/MiniMax-H3}" bash "$HERE/oneshot-essential-bundle.sh" \
    $([ -z "$LORAS" ] || [ "$LORAS" = all ] || echo "--categories=$LORAS") $([ -z "${LORA_BUNDLE:-}" ] || echo "--source=$LORA_BUNDLE") 2>&1 | sed 's/^/  /' \
    || echo "  ! some LoRAs could not be fetched (listed above): the rest are installed, and bash oneshot-essential-bundle.sh tries again"
fi
if [ "$FLUX3" = 1 ] || [ -x "$PREFIX/flux3-venv/bin/python" ]; then  # FLUX3-K picture models: their own venv and weights
  PREFIX="$PREFIX" bash "$HERE/scripts/setup-flux3.sh" $([ "$MODE" = "--verify" ] || [ "$UPGRADE" = 1 ] && echo --verify) \
    || [ "$FLUX3" = 1 ] || echo "  ! FLUX3-K is not fully set up: bash scripts/setup-flux3.sh"
fi
if [ "$MODE" = "--verify" ] || [ "$MODE" = "--no-render" ]; then
  echo; echo "DONE. Image: bash $HERE/scripts/image.sh \"a prompt\" out.png"
  [ "$VIDEO" = 0 ] || echo "      Studio: bash $HERE/scripts/studio.sh \"the picture\" \"what happens\" out.mp4"
  exit 0
fi

export PREFIX H3_MODEL_DIR QWEN_MODEL_DIR
if [ "$VIDEO" = 0 ]; then
  step "Test image: 1344x768, turbo adapter, int8 kernels"
  PROMPT_FILE="$HERE/prompts/baker-image.txt" SEED=42 bash "$HERE/scripts/image.sh" "" "$HERE/outputs/test.png" 2>&1 \
    | grep -E "tensorfold\] \{|rror|Trace" | sed 's/^/  /'
  [ -s "$HERE/outputs/test.png" ] || die "the test render wrote no image"
  echo; echo "DONE. $HERE/outputs/test.png"
  exit 0
fi
step "Test clip: text -> image -> 5 s video with sound, 864x480, FastH3 8 passes"
ENGINE=fasth3 WIDTH=864 HEIGHT=480 FRAMES=124 SEED=42 IMAGE_PROMPT_FILE="$HERE/prompts/baker-image.txt" \
  VIDEO_PROMPT_FILE="$HERE/prompts/baker-video.txt" bash "$HERE/scripts/studio.sh" "" "" "$HERE/outputs/test.mp4" 2>&1 \
  | grep -E "tensorfold\] \{|studio\]|rounded|rror|Trace" | sed 's/^/  /'
[ -s "$HERE/outputs/test.mp4" ] || die "the test render wrote no file"
echo
echo "DONE. $HERE/outputs/test.png and $HERE/outputs/test.mp4"
echo "App:    double-click TensorFold Studio in ~/Applications   (bash $HERE/scripts/make-app.sh builds it; or bash $HERE/scripts/app.sh)"
echo "Studio: bash $HERE/scripts/studio.sh \"the picture\" \"what happens in the clip\" out.mp4"
echo "Image:  bash $HERE/scripts/image.sh \"a prompt\" out.png"
echo "Video:  bash $HERE/scripts/video.sh \"a prompt\" out.mp4      (FIRST_FRAME=photo.jpg for image to video)"
[ "$REF2VA" = 0 ] || echo "Ref:    REF_IMAGES=face.jpg bash $HERE/scripts/video.sh \"Use <Picture 1> as the person. The person ...\" out.mp4"
echo "Fast:   bash $HERE/scripts/fast.sh \"a prompt\" out.mp4       (FastH3, text to video; STEPS=4|8|20, RES=480p|720p, UPSCALE=1)"
