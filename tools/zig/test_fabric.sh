#!/usr/bin/env bash
# The fabric's tests on this machine (CPU, in-process fakes) and the link test build; TF_MCDMA_SRC names an MCDMA checkout.
set -euo pipefail
zig="${ZIG:-zig}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
fabric="$root/zig/src/fabric/fabric.zig"
"$zig" test -lc "$fabric"
for name in fabric_handoff fabric_vectors; do
  "$zig" test -lc --dep fabric "-Mroot=$root/zig/tests/$name.zig" "-Mfabric=$fabric"
done
for name in fabric_collective fabric_sendrecv; do
  "$zig" test -lc -O ReleaseSafe --dep fabric "-Mroot=$root/zig/tests/$name.zig" "-Mfabric=$fabric"
done
mkdir -p "$root/build/linktest"
# A baseline arm64 target, not this machine's CPU, so the binary runs on any Apple Silicon Mac.
"$zig" build-exe -target aarch64-macos.26.0 -O ReleaseFast -fstrip -lc --dep fabric "-Mroot=$root/zig/tests/linktest.zig" "-Mfabric=$fabric" "-femit-bin=$root/build/linktest/tf-linktest"
src="${TF_MCDMA_SRC:-$root/build/mcdma}"
if [ ! -f "$src/rpc/libmcdma_rpc.c" ]; then
  echo "no MCDMA checkout at $src: skipping the libmcdma-rpc binding tests"
  exit 0
fi
out="$root/build/fabric"
mkdir -p "$out"
# Built as MCDMA's own `make -C rpc test` builds it, into this tree's ignored build/; nothing is installed.
if [ "$(uname -s)" = Darwin ]; then
  lib="$out/libmcdma-rpc.dylib"
  cc -std=c11 -O2 -Wall -Wextra -Werror -dynamiclib -o "$lib" "$src/rpc/libmcdma_rpc.c" "$src/rpc/libmcdma_rpc_metal.m" -fobjc-arc -framework Metal -framework Foundation
else
  lib="$out/libmcdma-rpc.so"
  cc -std=c11 -O2 -Wall -Wextra -Werror -shared -fPIC -o "$lib" "$src/rpc/libmcdma_rpc.c"
fi
MCDMA_RPC_LIBRARY="$lib" "$zig" test -lc --dep fabric "-Mroot=$root/zig/tests/fabric_library.zig" "-Mfabric=$fabric"
