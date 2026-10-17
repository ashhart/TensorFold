#!/usr/bin/env bash
# Motion Previs analysis packages and models in $PREFIX/motion-venv and motion-models. Usage: [--verify]
set -euo pipefail
# A Python port of Motion Previs Studio by Sam Wasserman (Apache-2.0); models and hashes from its ASSET_MANIFEST.json.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
VENV="$PREFIX/motion-venv"; MODELS="${MOTION_MODELS:-$PREFIX/motion-models}"
VERIFY=0; [ "${1:-}" = "--verify" ] && VERIFY=1
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
ok() { echo "  ✓ $*"; }
note() { echo "  ! $*"; }
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# name, sha256 (Motion Previs Studio's ASSET_MANIFEST.json), url
POSE="pose_landmarker_lite.task 59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
pose_landmarker_full.task 5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1 https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
pose_landmarker_heavy.task 64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
# Depth-Anything-small ONNX as the original loads it (Xenova/depth-anything-small-hf at its pinned revision, 8-bit)
DEPTH_NAME="depth-anything-small/model_quantized.onnx"
DEPTH_SHA256="e4e2ffe9b0396ebd4a0c09b855762b186aab338e79f76531b09777aafc91db84"
DEPTH_URL="https://huggingface.co/Xenova/depth-anything-small-hf/resolve/2e942621ab9f2371c1df9eb223291b5ac31475e6/onnx/model_quantized.onnx"

fetch() {  # path, sha256, url: a file that is there with the right hash is kept; a wrong one is never left in place
  local path="$1" want="$2" url="$3"
  if [ -f "$path" ] && [ "$(sha "$path")" = "$want" ]; then return 0; fi
  [ "$VERIFY" = 0 ] || return 1
  mkdir -p "$(dirname "$path")"
  curl -fsSL --retry 3 -o "$path.partial" "$url" || { rm -f "$path.partial"; return 1; }
  [ "$(sha "$path.partial")" = "$want" ] || { rm -f "$path.partial"; echo "  checksum mismatch for $(basename "$path")" >&2; return 1; }
  mv "$path.partial" "$path"
}

echo "==> Motion Previs: analysis environment -> $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  [ "$VERIFY" = 0 ] || { note "no environment at $VENV: pose, the camera solve and AI depth are off"; exit 1; }
  command -v uv >/dev/null || { echo "uv required (brew install uv, or https://astral.sh/uv)" >&2; exit 1; }
  # the Studio's Python on each machine: 3.11 on a Mac, 3.12 on a DGX Spark
  if [ "$(uname -s)" = Darwin ]; then uv venv -q -p "$(command -v python3.11 || echo 3.11)" "$VENV"
  else uv venv -q --python 3.12 --python-preference only-managed "$VENV"; fi
fi
if [ "$VERIFY" = 0 ]; then
  # a package without a wheel here is left out: its layer reports itself unavailable, the others still work
  uv pip install -q -p "$VENV/bin/python" --only-binary :all: -r "$HERE/app/motion-requirements.txt" || {
    note "the full set did not install here; installing what has a wheel for this machine"
    for package in numpy opencv-contrib-python "onnxruntime==1.31.0" "mediapipe==1.1.0"; do
      uv pip install -q -p "$VENV/bin/python" --only-binary :all: "$package" || note "$package has no wheel for this machine: left out"
    done
    "$VENV/bin/python" -c "import cv2" 2>/dev/null || uv pip install -q -p "$VENV/bin/python" --only-binary :all: opencv-python-headless || true
  }
fi

echo "==> Motion Previs: models -> $MODELS"
while read -r name want url; do
  fetch "$MODELS/$name" "$want" "$url" && ok "$name" || note "$name is missing: that pose model is not offered"
done <<< "$POSE"
fetch "$MODELS/$DEPTH_NAME" "$DEPTH_SHA256" "$DEPTH_URL" && ok "$DEPTH_NAME (Depth-Anything-small, ONNX)" || note "no Depth-Anything-small: the ffmpeg depth proxy stands in"

MOTION_MODELS="$MODELS" "$VENV/bin/python" "$HERE/app/motion_worker.py" --probe | "$VENV/bin/python" -c '
import json, sys
p = json.load(sys.stdin)
for key, label in (("pose", "pose (MediaPipe " + str(p["mediapipe"]) + ", models: " + ", ".join(p["pose_models"]) + ")"),
                   ("camera", "camera solve (OpenCV " + str(p["cv2"]) + ")"), ("ai_depth", "AI depth (ONNX Runtime " + str(p["onnxruntime"]) + ")")):
    print(("  ✓ " if p[key] else "  ! not available on this machine: ") + label)
for name, why in p["why"].items():
    print("    " + name + ": " + why)
sys.exit(0 if p["ffmpeg"] else 1)' || { echo "ffmpeg is needed for every layer" >&2; exit 1; }
echo "Motion Previs is set up: open the Motion Previs tab of the app (bash scripts/app.sh)."
