#!/bin/sh
set -eu
: "${ZIG:?set ZIG to the installed pinned 0.17.0 compiler}"
[ "$($ZIG version)" = "0.17.0" ]
bf16_topk_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$bf16_topk_root"
bf16_topk_sdk=$(xcrun --sdk macosx --show-sdk-path)
export ZIG_GLOBAL_CACHE_DIR="$bf16_topk_root/.zig-cache/global"
"$ZIG" test -OReleaseSafe zig/src/core/bf16_topk.zig
"$ZIG" build-exe -OReleaseSafe -target aarch64-native -mcpu=apple_m1 --dep metal --dep bf16_topk --dep bf16_topk_gpu -Mroot=zig/tests/bf16_topk.zig --dep metal --dep bf16_topk --dep bf16_topk_sources -Mbf16_topk_gpu=zig/src/core/bf16_topk_gpu.zig -Mbf16_topk=zig/src/core/bf16_topk.zig -Mmetal=zig/src/metal/metal.zig -Mbf16_topk_sources=zig/bf16_topk_sources.zig -F "$bf16_topk_sdk/System/Library/Frameworks" -L "$bf16_topk_sdk/usr/lib" -framework Metal -framework Foundation -lc -lobjc -femit-bin=.zig-cache/tf-bf16-topk
