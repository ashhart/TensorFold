#!/usr/bin/env bash
# Usage: pack.sh OUT_KERNELS CAPTURE_OUT...: one kernel set from the captures' manifest.json and jit.json.
set -euo pipefail
OUT=$1; shift
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
ARGS=()
for d in "$@"; do ARGS+=(--manifest "$d/manifest.json" --jit "$d/jit.json"); done
python3 -B "$REPO/tools/zig/triton_aot_pack.py" "${ARGS[@]}" --out "$OUT"
