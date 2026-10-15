#!/usr/bin/env bash
# Usage: capture_mtp.sh WORK [IMAGE]: the MTP head's own Triton kernels (capture_mtp.py) into WORK/out, no checkpoint.
set -euo pipefail
WORK=$(readlink -f "$1"); IMAGE=${2:-tensorfold-qwen38:v0.6.1}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
mkdir -p "$WORK/cache" "$WORK/out"
docker run --rm --name fn-capture-mtp --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e TENSORFOLD_NO_UPDATE_CHECK=1 -v "$WORK/cache":/cache -v "$REPO":/src:ro -v "$WORK/out":/out -w /out \
  "$IMAGE" python -u /src/zig/tests/cuda/flashnext/capture_mtp.py --out /out --tools /src/tools/zig
python3 -B "$REPO/tools/zig/triton_aot_manifest.py" --cache "$WORK/cache/triton" \
  --launches "$WORK/out/launches.json" --mount /cache/triton --out "$WORK/out/manifest.json"
