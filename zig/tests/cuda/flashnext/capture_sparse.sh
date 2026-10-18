#!/usr/bin/env bash
# Usage: capture_sparse.sh WORK [IMAGE]: the sparse attention's kernels and selection fixtures (capture_sparse.py).
set -euo pipefail
WORK=$(readlink -f "$1"); IMAGE=${2:-tensorfold-qwen38:v0.6.1}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
mkdir -p "$WORK/cache" "$WORK/out" "$WORK/fixtures"
docker run --rm --name fn-capture-sparse --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e TENSORFOLD_NO_UPDATE_CHECK=1 -v "$WORK/cache":/cache -v "$REPO":/src:ro -v "$WORK/out":/out -v "$WORK/fixtures":/fixtures -w /out \
  "$IMAGE" python -u /src/zig/tests/cuda/flashnext/capture_sparse.py --out /out --tools /src/tools/zig --fixtures /fixtures
python3 -B "$REPO/tools/zig/triton_aot_manifest.py" --cache "$WORK/cache/triton" \
  --launches "$WORK/out/launches.json" --mount /cache/triton --out "$WORK/out/manifest.json"
