#!/bin/bash
# FLUX3-K on a Mac (--flux3): flux3-mlx (Apache-2.0) pinned in $PREFIX/flux3-venv, weights 41 GB. Usage: [--verify]
set -euo pipefail
# FLUX Kommunity License (non-commercial; outputs commercial only for Qualifying Users); gated: hf auth login first
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
DIR="${FLUX3_MODEL_DIR:-$HOME/flux3-models}"
FLUX3_MLX_REPO="https://github.com/felix-ab/flux3-mlx"
FLUX3_MLX_COMMIT="a965659be4f099be014e533694d7425a5345e64c"
VERIFY=0; [ "${1:-}" != --verify ] || VERIFY=1
ok() { echo "  ✓ $*"; }
die() { echo "  ✗ $*" >&2; exit 1; }
echo; echo "==> FLUX3-K: flux3-mlx @ ${FLUX3_MLX_COMMIT:0:8} -> $PREFIX/flux3-venv, weights (41 GB) -> $DIR"
PY="$PREFIX/flux3-venv/bin/python"
if ! "$PY" -c "import flux3_mlx" 2>/dev/null || [ "$("$PY" -c 'import importlib.metadata as m, json; print(json.loads(m.distribution("flux3-mlx").read_text("direct_url.json"))["vcs_info"]["commit_id"])' 2>/dev/null)" != "$FLUX3_MLX_COMMIT" ]; then
  [ "$VERIFY" = 0 ] || die "flux3-mlx is missing or not at the pinned commit"
  [ -x "$PY" ] || /opt/homebrew/bin/python3.11 -m venv "$PREFIX/flux3-venv"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q "flux3-mlx[pup] @ git+$FLUX3_MLX_REPO@$FLUX3_MLX_COMMIT"
fi
ok "flux3-mlx $("$PY" -c 'import importlib.metadata as m; print(m.version("flux3-mlx"))'), mlx $("$PY" -c 'import mlx.core as mx; print(mx.__version__)')"
if [ "$VERIFY" = 0 ]; then
  echo "  FLUX 3 Image Klein is under the FLUX Kommunity License: read it, and accept it on the model page, first:"
  echo "  https://huggingface.co/black-forest-labs/flux-3-image-klein"
  mkdir -p "$DIR"
  "$PY" - "$DIR" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
root = sys.argv[1]
for repo, folder, patterns in (
        ("black-forest-labs/flux-3-image-klein", "FLUX3-K-Fast", ["model.safetensors", "ae.safetensors", "text_encoder/*", "LICENSE.md", "NOTICE.md"]),
        # top-level names only: *.md would also take text_encoder/README.md and leave no room for the link
        ("black-forest-labs/flux-3-image-klein-base", "FLUX3-K-Full", ["model.safetensors", "LICENSE.md", "NOTICE.md"]),
        ("VisualInference/FLUX-3-Image-Klein-pup-MLX-8bit", "FLUX3-K-pup-MLX-8bit", None)):
    snapshot_download(repo, local_dir=os.path.join(root, folder), allow_patterns=patterns, max_workers=4)
for name in ("ae.safetensors", "text_encoder"):  # Full shares Fast's text encoder and decoder
    link = os.path.join(root, "FLUX3-K-Full", name)
    if not os.path.lexists(link):
        os.symlink(os.path.join("..", "FLUX3-K-Fast", name), link)
PY
fi
for f in FLUX3-K-Fast/model.safetensors FLUX3-K-Fast/ae.safetensors FLUX3-K-Fast/text_encoder/config.json \
         FLUX3-K-Full/model.safetensors FLUX3-K-Full/ae.safetensors FLUX3-K-Full/text_encoder/config.json \
         FLUX3-K-pup-MLX-8bit/system_prompts.json; do
  [ -s "$DIR/$f" ] || die "missing $DIR/$f"
done
ok "FLUX3-K Fast and Full (bf16), text encoder, decoder and BFL's prompt upsampler (8-bit MLX)"
