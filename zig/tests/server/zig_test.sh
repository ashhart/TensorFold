#!/usr/bin/env bash
# The server's unit tests (its module with the engine seam, tokenizer and template), then the engine seam's own. ZIG overrides the compiler.
set -euo pipefail
zig="${ZIG:-zig}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
"$zig" test -lc --dep engine_api --dep tokenizer --dep template \
  "-Mroot=$root/zig/src/server/root.zig" \
  --dep lanes "-Mengine_api=$root/zig/src/core/engine_api.zig" \
  "-Mlanes=$root/zig/src/core/lanes/lanes.zig" \
  "-Mtokenizer=$root/zig/src/core/tokenizer/tokenizer.zig" \
  "-Mtemplate=$root/zig/src/core/template/template.zig" "$@"
"$zig" test -lc --dep lanes "-Mroot=$root/zig/src/core/engine_api.zig" "-Mlanes=$root/zig/src/core/lanes/lanes.zig"
