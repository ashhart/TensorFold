# Builds fake_serve and the engine-less tensorfold-native into zig-out/server/ (zig build server-cpu).
set -euo pipefail
zig="${ZIG:-zig}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$root"
exec "$zig" build server-cpu -Doptimize="${OPT:-ReleaseSafe}" "$@" # ZIG overrides the compiler, OPT the mode.
