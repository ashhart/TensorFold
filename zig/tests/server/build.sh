#!/usr/bin/env bash
# Builds fake_serve and the engine-less tensorfold-native into zig-out/server/ through the shared
# build graph (zig build server-cpu); ZIG overrides the compiler, OPT the mode.
set -euo pipefail
zig="${ZIG:-zig}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$root"
exec "$zig" build server-cpu -Doptimize="${OPT:-ReleaseSafe}" "$@"
