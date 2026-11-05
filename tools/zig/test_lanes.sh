#!/usr/bin/env bash
# The lane core's tests, plus record_lanes.py's vectors and traces where TF_LANES_FIXTURES names them; ZIG picks the compiler.
set -euo pipefail
zig="${ZIG:-zig}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
lanes="$root/zig/src/core/lanes/lanes.zig"
sys="$root/zig/src/platform/sys.zig"
"$zig" test -lc --dep sys "-Mroot=$lanes" "-Msys=$sys"
for name in lanes_units lanes_replay; do
  "$zig" test -lc --dep lanes --dep sys "-Mroot=$root/zig/tests/$name.zig" --dep sys "-Mlanes=$lanes" "-Msys=$sys"
done
