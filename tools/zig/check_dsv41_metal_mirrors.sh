#!/bin/sh
# Compile the DeepSeek V4.1 family's Metal sources at build time.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
dir="$root/zig/kernels/metal/deepseek_v41"
[ -d "$dir" ] || { echo "metal mirrors: $dir not found"; exit 2; }
[ -f "$root/zig/kernels/metal/nax.h" ] || { echo "metal mirrors: nax.h not found"; exit 2; }
command -v xcrun >/dev/null 2>&1 || { echo "metal mirrors: xcrun not on PATH"; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "metal mirrors: python3 not on PATH"; exit 2; }

exec python3 "$root/tools/zig/dsv41_metal_struct_sizes.py" "$dir"
