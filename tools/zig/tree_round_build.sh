#!/bin/sh
set -eu
: "${ZIG:?set ZIG to the installed pinned 0.17.0 compiler}"
[ "$($ZIG version)" = "0.17.0" ]
tree_round_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$tree_round_root"
tree_round_sdk=$(xcrun --sdk macosx --show-sdk-path)
export ZIG_GLOBAL_CACHE_DIR="$tree_round_root/.zig-cache/global"
"$ZIG" test -OReleaseSafe zig/src/core/tree_round.zig
"$ZIG" build-exe -OReleaseSafe -target aarch64-native -mcpu=apple_m1 --dep metal --dep tree_round --dep tree_round_gpu -Mroot=zig/tests/tree_round.zig --dep metal --dep tree_round --dep tree_round_sources -Mtree_round_gpu=zig/src/core/tree_round_gpu.zig -Mtree_round=zig/src/core/tree_round.zig -Mmetal=zig/src/metal/metal.zig -Mtree_round_sources=zig/tree_round_sources.zig -F "$tree_round_sdk/System/Library/Frameworks" -L "$tree_round_sdk/usr/lib" -framework Metal -framework Foundation -lc -lobjc -femit-bin=.zig-cache/tf-tree-round
