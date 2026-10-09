#!/bin/sh
set -eu
topk_zig=${ZIG:-zig}
topk_out=${1:?new task-created output directory required}
[ "$($topk_zig version)" = "$(tr -d '\r\n' < .zig-version)" ]
[ ! -e "$topk_out" ]
mkdir -p "$topk_out"
ZIG_GLOBAL_CACHE_DIR="$topk_out/global" "$topk_zig" build-exe -OReleaseSafe -lc -framework Metal -framework Foundation -lobjc --dep metal --dep bf16_topk --dep bf16_topk_gpu -Mroot=zig/tests/bf16_topk_bench.zig -Mmetal=zig/src/metal/metal.zig -Mbf16_topk=zig/src/core/bf16_topk.zig --dep metal --dep bf16_topk=bf16_topk --dep bf16_topk_sources -Mbf16_topk_gpu=zig/src/core/bf16_topk_gpu.zig -Mbf16_topk_sources=zig/bf16_topk_sources.zig -femit-bin="$topk_out/topk-bench"
"$topk_out/topk-bench" --cpu
