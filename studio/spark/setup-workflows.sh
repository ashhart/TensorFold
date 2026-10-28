#!/bin/bash
# Node packs and models behind the Library actions on a DGX Spark (needs Ref2VA). Usage: [--verify] [--no-models]
set -euo pipefail
# Authors' nodes, pinned, never copied: ContactSheet by MATLOWAI (GPL-3.0), Video-Upsampler by dntpi/Sandpies (MIT),
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
# H3 Relay by Akatz Labs (GPL-3.0); models: SAM 3.1 (SAM License), LBH-123-AI's H3 latent upscaler (Apache-2.0).
MODELS="${SPARK_MODELS:-$HOME/spark-studio-models}"
VERIFY=0; NOMODELS=0
for arg in "$@"; do case "$arg" in --verify) VERIFY=1;; --no-models) NOMODELS=1;; *) echo "usage: setup-workflows.sh [--verify] [--no-models]" >&2; exit 2;; esac; done
COMFY="$PREFIX/ComfyUI"
[ -f "$COMFY/main.py" ] || { echo "no ComfyUI at $COMFY: run oneshot-setup-spark.sh first" >&2; exit 1; }
bad=0

# node <folder> <repository> <commit>
node() {
  local dir="$COMFY/custom_nodes/$1" repo="$2" commit="$3"
  if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" = "$commit" ]; then echo "  ✓ $1 @ ${commit:0:8}"; return 0; fi
  if [ "$VERIFY" = 1 ]; then echo "  ! $1 is missing or not at ${commit:0:8}"; bad=1; return 0; fi
  [ -d "$dir/.git" ] || git clone -q "$repo" "$dir"
  git -C "$dir" fetch -q origin "$commit" 2>/dev/null || git -C "$dir" fetch -q origin
  git -C "$dir" checkout -q "$commit"
  echo "  ✓ $1 @ ${commit:0:8} (new: restart ComfyUI to load it)"; touch "$PREFIX/workflows.restart"
}
node ComfyUI-H3-ContactSheet     https://github.com/matlowai/ComfyUI-H3-ContactSheet     3c8c8c66e69362e5abf9cf4c4e693baddab90ca3
node ComfyUI-H3-Video-Upsampler  https://github.com/dntpi/ComfyUI-H3-Video-Upsampler     36cb612ec3df30094eeb4fec66f1528925ebba73
node h3-relay                    https://github.com/akatz-ai/h3-relay                    cf489b7f2472fdf33866fae36de49aea03ef464f

# model <repository> <file in it> <folder under $MODELS/Workflows-Comfy> <name there> <sha256>
model() {
  local repo="$1" file="$2" dir="$MODELS/Workflows-Comfy/$3" name="$4" sum="$5"
  if [ -s "$dir/$name" ]; then echo "  ✓ $name"; return 0; fi
  if [ "$VERIFY" = 1 ]; then echo "  ! missing: $dir/$name"; bad=1; return 0; fi
  if [ "$NOMODELS" = 1 ]; then echo "  ! missing: $dir/$name (run spark/setup-workflows.sh to fetch it)"; return 0; fi
  mkdir -p "$dir"
  curl -fL --retry 5 -C - -sS -o "$dir/$name.part" "https://huggingface.co/$repo/resolve/main/$file"
  echo "$sum  $dir/$name.part" | sha256sum -c --quiet - || { echo "  ! $name: checksum mismatch" >&2; rm -f "$dir/$name.part"; exit 1; }
  mv "$dir/$name.part" "$dir/$name"; echo "  ✓ $name (downloaded)"; touch "$PREFIX/workflows.restart"
}
model Comfy-Org/sam3.1 checkpoints/sam3.1_multiplex_fp16.safetensors checkpoints sam3.1_multiplex_fp16.safetensors \
  9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03
model LBH-123-AI/Minimax_h3_latent_Upscaler minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors \
  latent_upscale_models minimax_h3_latent_upscaler_3d_bf16.safetensors 4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6

# ComfyUI reads these folders from extra_model_paths.yaml
YAML="$COMFY/extra_model_paths.yaml"
if ! grep -q "Workflows-Comfy/checkpoints" "$YAML" 2>/dev/null; then
  if [ "$VERIFY" = 1 ]; then echo "  ! $YAML does not list the workflow models"; bad=1; else
    cat >> "$YAML" <<EOF
  checkpoints: |
    Workflows-Comfy/checkpoints
  latent_upscale_models: |
    Workflows-Comfy/latent_upscale_models
EOF
    echo "  ✓ model folders added to extra_model_paths.yaml"; touch "$PREFIX/workflows.restart"
  fi
fi
# the authors' fp16 video decoder and 8-bit Ref2VA are optional: the Studio's graphs use bf16 Ref2VA, 8-bit decoder
[ -s "$MODELS/MiniMax-H3-Comfy/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors" ] \
  || { echo "  ! MiniMax H3 Ref2VA is not installed: bash oneshot-setup-spark.sh --ref2va"; bad=1; }
if [ -e "$PREFIX/workflows.restart" ] && bash "$(dirname "$0")/comfy.sh" status | grep -q running; then
  echo "  restart ComfyUI to load the new nodes: bash spark/comfy.sh stop (the next render starts it)"
fi
exit $bad
