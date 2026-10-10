#!/bin/sh
set -eu
: "${ZIG:?set ZIG to the installed pinned 0.17.0 compiler}"
[ "$($ZIG version)" = "0.17.0" ]
draft_ops_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$draft_ops_root"
draft_ops_sdk=$(xcrun --sdk macosx --show-sdk-path)
export ZIG_GLOBAL_CACHE_DIR="$draft_ops_root/.zig-cache/global"
"$ZIG" test -OReleaseSafe -target aarch64-native -mcpu=apple_m1 --dep metal --dep draft_ops_sources -Mroot=zig/src/core/draft_ops.zig -Mmetal=zig/src/metal/metal.zig -Mdraft_ops_sources=zig/draft_ops_sources.zig -F "$draft_ops_sdk/System/Library/Frameworks" -L "$draft_ops_sdk/usr/lib" -framework Metal -framework Foundation -lc -lobjc
"$ZIG" build-exe -OReleaseSafe -target aarch64-native -mcpu=apple_m1 --dep metal --dep draft_ops -Mroot=zig/tests/draft_ops.zig --dep metal --dep draft_ops_sources -Mdraft_ops=zig/src/core/draft_ops.zig -Mmetal=zig/src/metal/metal.zig -Mdraft_ops_sources=zig/draft_ops_sources.zig -F "$draft_ops_sdk/System/Library/Frameworks" -L "$draft_ops_sdk/usr/lib" -framework Metal -framework Foundation -lc -lobjc -femit-bin=.zig-cache/tf-draft-ops
