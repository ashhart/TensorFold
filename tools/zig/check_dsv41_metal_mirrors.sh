#!/bin/sh
# Compile the DeepSeek V4.1 family's Metal sources at build time, exactly as the engine concatenates
# them at run time, so their `static_assert(sizeof(...) == N)` mirror checks fail the BUILD, not the boot.
# The same pass also enforces COMPLETENESS and a MANIFEST:
#
#   * every struct a kernel takes by `constant T&` or `device T*` must assert its own size in the unit that
#     compiles it (a new kernel argument cannot land without one),
#   * every struct the sources define must keep the size the Metal compiler gives it today, as recorded in
#     `zig/kernels/metal/deepseek_v41/struct_sizes.txt` (`--update` regenerates it), so any Metal-side edit or
#     compiler layout change is a build error even for a struct no Zig type mirrors.
#
# The work is `tools/zig/dsv41_metal_struct_sizes.py`: it reproduces the engine's three run-time joins
# (csa2.zig:91, csa2_index.zig:164, metal/pipelines.zig:122) in a temporary tree that also holds a copy of
# `nax.h` (the sources `#include "../nax.h"`), compiles every unit with `-Xclang -fdump-record-layouts`,
# and reports the compiler's own message on a failure.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
dir="$root/zig/kernels/metal/deepseek_v41"
[ -d "$dir" ] || { echo "metal mirrors: $dir not found"; exit 2; }
[ -f "$root/zig/kernels/metal/nax.h" ] || { echo "metal mirrors: nax.h not found"; exit 2; }
command -v xcrun >/dev/null 2>&1 || { echo "metal mirrors: xcrun not on PATH"; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "metal mirrors: python3 not on PATH"; exit 2; }

exec python3 "$root/tools/zig/dsv41_metal_struct_sizes.py" "$dir"
