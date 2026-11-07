#!/usr/bin/env bash
# Fish Audio Drama 3 for the Voice tab (--voice): drama3-mlx on a Mac, only a check on a Spark. Built with Fish Audio.
set -euo pipefail
# Fish Audio Community License v1.0 weights; a hosted TTS or cloning service needs Fish's licence; no Fish API calls.

REPO="https://github.com/drowzeys/TensorFold.git"
# Env: DRAMA3_SRC, DRAMA3_VENV, DRAMA3_CHECKPOINT, DRAMA3_MODEL; the Spark's CUDA engine is built by hand (README).
MLX_BRANCH="drama3-mlx"; MLX_COMMIT="8f3d4b33"
CUDA_BRANCH="drama3-cuda"; CUDA_COMMIT="81173e6c"
VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --verify) VERIFY=1;;
    *) echo "unknown option $arg" >&2; exit 2;;
  esac
done
die() { echo "FATAL: $*" >&2; exit 1; }
ok()  { echo "  ✓ $*"; }
step(){ echo; echo "==> $*"; }
CHECKPOINT="${DRAMA3_CHECKPOINT:-$HOME/models/drama-3}"

licence() {
  echo "  Fish Audio Drama 3 (39 GB) is licensed under the Fish Audio Community License v1.0: research, non-commercial"
  echo "  and small-company commercial use; a hosted TTS or voice-cloning service needs Fish Audio's own licence."
  echo "  Read it: https://huggingface.co/fishaudio/drama-3"
}

if [ "$(uname -s)" = Linux ]; then
  PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
  BIN="${DRAMA3_CUDA_BIN:-$PREFIX/drama3/bin/drama3-generate}"
  LIB="${DRAMA3_CUDA_LIB:-$(dirname "$(dirname "$BIN")")/lib/libtensorfold_drama3.so}"
  CODEC="${DRAMA3_CUDA_CODEC:-$PREFIX/drama3/codec-v1.safetensors}"
  MODEL="${DRAMA3_CUDA_MODEL:-$CHECKPOINT}"
  step "Drama 3 on a DGX Spark (TensorFold $CUDA_BRANCH @ $CUDA_COMMIT)"
  MISSING=0
  [ -x "$BIN" ] && ok "engine $BIN" || { echo "  ! no engine at $BIN"; MISSING=1; }
  [ -f "$LIB" ] && ok "library $LIB" || { echo "  ! no library at $LIB"; MISSING=1; }
  [ -f "$MODEL/model.safetensors.index.json" ] && ok "checkpoint $MODEL" || { echo "  ! no checkpoint at $MODEL"; MISSING=1; }
  [ -f "$CODEC" ] && ok "codec $CODEC" || { echo "  ! no converted codec at $CODEC"; MISSING=1; }
  if [ "$MISSING" = 1 ]; then
    echo
    echo "  The CUDA engine is set up by hand (README, Voice on a DGX Spark):"
    echo "    git clone -b $CUDA_BRANCH $REPO ~/tensorfold-drama3-cuda && git -C ~/tensorfold-drama3-cuda checkout $CUDA_COMMIT"
    echo "    hf download fishaudio/drama-3 --local-dir $CHECKPOINT"
    echo "    build drama3-generate and drama3-lib (tools/drama3/build_local.sh shows how), copy them into $PREFIX/drama3/{bin,lib}"
    echo "    python3 tools/drama3/codec_convert.py $CHECKPOINT $CODEC     # once"
    licence
    [ "$VERIFY" = 1 ] && exit 1
    exit 0
  fi
  echo "  Voice tab ready: the app runs it as a persistent server, unloaded after five idle minutes."
  exit 0
fi

[ "$(uname -m)" = arm64 ] || die "the MLX port of Drama 3 needs Apple silicon"
MEM=$(sysctl -n hw.memsize)
[ "$MEM" -ge $((64 * 1024 * 1024 * 1024)) ] || die "Drama 3 peaks at about 52 GB: a Mac with 64 GB or more"
SRC="${DRAMA3_SRC:-$HOME/tensorfold-drama3-mlx}"
# a TensorFold clone at ~/TensorFold that has the port already (the README's earlier hand setup) is used as it is
if [ -z "${DRAMA3_SRC:-}" ] && [ ! -d "$SRC" ] && git -C "$HOME/TensorFold" merge-base --is-ancestor "$MLX_COMMIT" HEAD 2>/dev/null; then
  SRC="$HOME/TensorFold"
fi
VENV="${DRAMA3_VENV:-$HOME/drama3-venv}"
MODEL="${DRAMA3_MODEL:-$HOME/models/drama-3-mlx}"
PY="$VENV/bin/python"

step "TensorFold $MLX_BRANCH @ $MLX_COMMIT in $SRC"
if [ -d "$SRC/.git" ]; then
  if git -C "$SRC" merge-base --is-ancestor "$MLX_COMMIT" HEAD 2>/dev/null; then
    ok "$SRC is at $(git -C "$SRC" rev-parse --short=8 HEAD) (has $MLX_COMMIT)"
  elif [ "$VERIFY" = 1 ]; then
    die "$SRC does not have $MLX_COMMIT (run without --verify)"
  elif [ -n "$(git -C "$SRC" status --porcelain)" ]; then
    die "$SRC has local changes and lacks $MLX_COMMIT: commit or move them, or set DRAMA3_SRC to another folder"
  else
    git -C "$SRC" fetch -q "$REPO" "$MLX_BRANCH" && git -C "$SRC" checkout -q --detach "$MLX_COMMIT" && ok "checked out $MLX_COMMIT"
  fi
else
  [ "$VERIFY" = 0 ] || die "no TensorFold clone at $SRC"
  git clone -q -b "$MLX_BRANCH" "$REPO" "$SRC" && git -C "$SRC" checkout -q --detach "$MLX_COMMIT" && ok "cloned"
fi

step "Its own environment: $VENV"
if [ ! -x "$PY" ]; then
  [ "$VERIFY" = 0 ] || die "no environment at $VENV"
  PY311="$(command -v python3.11 || echo /opt/homebrew/bin/python3.11)"
  [ -x "$PY311" ] || die "python3.11 is needed (brew install python@3.11)"
  "$PY311" -m venv "$VENV"
fi
if ! "$PY" -I -c "import tensorfold.drama3, tensorfold.families.drama3.stream, mlx_whisper" 2>/dev/null; then
  [ "$VERIFY" = 0 ] || die "the environment lacks tensorfold.drama3 (with streaming) or mlx-whisper"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -e "$SRC" mlx mlx-lm mlx-whisper soundfile torch huggingface_hub
fi
"$PY" -I -c "import tensorfold.drama3, tensorfold.families.drama3.stream, mlx_whisper" || die "the environment does not import the engine"
ok "tensorfold.drama3 + mlx-whisper (the engine's own end-of-text check)"

step "Fish's checkpoint: $CHECKPOINT"
if [ -f "$CHECKPOINT/model.safetensors.index.json" ] && [ -f "$CHECKPOINT/codec.pth" ]; then
  ok "there"
elif [ -f "$MODEL/config.json" ] && [ -f "$MODEL/codec.safetensors" ]; then
  ok "not needed: the MLX conversion is there"
else
  [ "$VERIFY" = 0 ] || die "no checkpoint at $CHECKPOINT"
  licence
  "$VENV/bin/hf" download fishaudio/drama-3 --local-dir "$CHECKPOINT"
fi

step "The MLX conversion: $MODEL"
if [ -f "$MODEL/config.json" ] && [ -f "$MODEL/codec.safetensors" ]; then
  ok "there"
else
  [ "$VERIFY" = 0 ] || die "no converted model at $MODEL"
  "$PY" -I -m tensorfold.families.drama3.convert "$CHECKPOINT" "$MODEL"
  ok "converted (8-bit experts)"
fi

if [ "$VERIFY" = 0 ]; then
  step "A test line"
  OUT="$(mktemp -d)/drama3-test.wav"
  "$PY" -I -m tensorfold.drama3 --model "$MODEL" --text "Hello there. The Voice tab is ready." --out "$OUT" --seed 1 | tail -1
  [ -s "$OUT" ] || die "the test line wrote nothing"
  ok "$OUT"
fi
echo
echo "Voice tab ready (DRAMA3_VENV=$VENV, DRAMA3_MODEL=$MODEL if not the defaults). Built with Fish Audio."
