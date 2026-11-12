#!/usr/bin/env bash
# The fabric's tests here (CPU, in-process fakes) and the link test build; TF_MCDMA_SRC names an MCDMA checkout.
set -euo pipefail
zig="${ZIG:-zig}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
fabric="$root/zig/src/fabric/fabric.zig"
verbs="$root/zig/src/fabric/verbs.zig"
canon="$root/zig/src/cluster/canon.zig"
# `sys` as zig/build/sys.zig wires it for the build's own modules
sys="$root/zig/src/platform/sys.zig"
# `bf16_round` as zig/build/bf16_round.zig wires it for the build's own modules
bf16_round="$root/zig/src/core/bf16_round.zig"
# fabric.zig imports the bf16_round and verbs modules, and the verbs module imports sys
"$zig" test -lc --dep bf16_round --dep sys --dep verbs "-Mroot=$fabric" "-Mbf16_round=$bf16_round" "-Msys=$sys" --dep sys "-Mverbs=$verbs"
# verbs.zig owns verbs_abi.zig. Their tests do not run inside the fabric module.
"$zig" test -lc --dep sys "-Mroot=$verbs" "-Msys=$sys"
for name in fabric_handoff fabric_vectors; do
  "$zig" test -lc --dep fabric --dep sys "-Mroot=$root/zig/tests/$name.zig" --dep bf16_round --dep sys --dep verbs "-Mfabric=$fabric" "-Mbf16_round=$bf16_round" "-Msys=$sys" --dep sys "-Mverbs=$verbs"
done
for name in fabric_collective fabric_sendrecv; do
  "$zig" test -lc -O ReleaseSafe --dep fabric --dep sys "-Mroot=$root/zig/tests/$name.zig" --dep bf16_round --dep sys --dep verbs "-Mfabric=$fabric" "-Mbf16_round=$bf16_round" "-Msys=$sys" --dep sys "-Mverbs=$verbs"
done
mkdir -p "$root/build/linktest"
# A baseline arm64 target, not this machine's CPU, so the binary runs on any Apple Silicon Mac.
"$zig" build-exe -target aarch64-macos.26.0 -O ReleaseFast -fstrip -lc --dep fabric --dep sys "-Mroot=$root/zig/tests/linktest.zig" --dep bf16_round --dep sys --dep verbs "-Mfabric=$fabric" "-Mbf16_round=$bf16_round" "-Msys=$sys" --dep sys "-Mverbs=$verbs" "-femit-bin=$root/build/linktest/tf-linktest"
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
  shared=(-dynamiclib)
  stub="$out/libmcdma-fabric-stub.dylib"
else
  lib="$out/libmcdma-rpc.so"
  cc -std=c11 -O2 -Wall -Wextra -Werror -shared -fPIC -o "$lib" "$src/rpc/libmcdma_rpc.c"
  shared=(-shared -fPIC)
  stub="$out/libmcdma-fabric-stub.so"
fi
MCDMA_RPC_LIBRARY="$lib" "$zig" test -lc --dep fabric --dep sys "-Mroot=$root/zig/tests/fabric_library.zig" --dep bf16_round --dep sys --dep verbs "-Mfabric=$fabric" "-Mbf16_round=$bf16_round" "-Msys=$sys" --dep sys "-Mverbs=$verbs"
if [ ! -f "$src/tests/stub_link_verbs.c" ]; then
  echo "no stub verbs in $src: skipping mcdma.Endpoint over libmcdma-fabric"
  exit 0
fi
# libmcdma-fabric on MCDMA's stub verbs (roce0, roce1, tb0..), as its tests/test_link.py builds it; no device is opened
cc -std=c11 -O1 -g -Wall -Wextra -Werror -DMCDMA_LINK_TEST_INTERFACES "${shared[@]}" -o "$stub" "$src/rpc/libmcdma_fabric.c" "$src/rpc/link_verbs.c" "$src/rpc/link_tb.c" "$src/rpc/link_xchg.c" "$src/tests/stub_link_verbs.c" -lpthread
TF_MCDMA_FABRIC_STUB="$stub" "$zig" test -lc -O ReleaseSafe --dep fabric --dep sys "-Mroot=$root/zig/tests/fabric_mcdma_stub.zig" --dep bf16_round --dep sys --dep verbs "-Mfabric=$fabric" "-Mbf16_round=$bf16_round" "-Msys=$sys" --dep sys "-Mverbs=$verbs"
# the cluster's bring-up on the same stub: two nodes from one cluster file to their endpoints and an agreed plan
TF_MCDMA_FABRIC_STUB="$stub" "$zig" test -lc -O ReleaseSafe --dep cluster --dep fabric --dep sys "-Mroot=$root/zig/tests/cluster_mcdma_stub.zig" --dep fabric --dep canon --dep sys "-Mcluster=$root/zig/src/cluster/cluster.zig" --dep bf16_round --dep sys --dep verbs "-Mfabric=$fabric" "-Mbf16_round=$bf16_round" "-Msys=$sys" --dep sys "-Mverbs=$verbs" --dep sys "-Mcanon=$canon"
