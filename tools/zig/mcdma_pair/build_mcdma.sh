#!/usr/bin/env bash
# libmcdma-fabric from an MCDMA tree as its rpc/Makefile builds it, or an offline stand-in (--stub, --loop): README.md
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
stub=0
loop=0
if [ "${1:-}" = --stub ]; then stub=1; shift; elif [ "${1:-}" = --loop ]; then loop=1; shift; fi
src="${1:-${TF_MCDMA_SRC:-$here/mcdma}}"
out="${2:-$src/build}"
[ -f "$src/rpc/libmcdma_fabric.c" ] || { echo "build_mcdma.sh: no MCDMA tree at $src (rpc/libmcdma_fabric.c)" >&2; exit 2; }
mkdir -p "$out"
out="$(cd "$out" && pwd)"
cd "$src/rpc"
files=(libmcdma_fabric.c link_verbs.c link_tb.c link_xchg.c)
if [ "$(uname -s)" = Darwin ]; then
  ext=dylib; shared=(-dynamiclib); verbs=(-lrdma); extra=()
else
  ext=so; shared=(-shared -fPIC); verbs=(-libverbs -lrt); extra=(-ldl)
fi
if [ "$loop" = 1 ]; then
  lib="$out/libloop-fabric.$ext"
  cc -std=c11 -O2 -Wall -Wextra -Werror -D_DEFAULT_SOURCE "${shared[@]}" -fvisibility=hidden -o "$lib" "$here/loop_fabric.c" -lpthread
elif [ "$stub" = 1 ]; then
  lib="$out/libmcdma-fabric-stub.$ext"
  # tests/test_link.py's flags, plus -shared; the stub stands in for every verbs call
  cc -std=c11 -O1 -g -Wall -Wextra -Werror -DMCDMA_LINK_TEST_INTERFACES "${shared[@]}" -o "$lib" "${files[@]}" ../tests/stub_link_verbs.c -lpthread ${extra[@]+"${extra[@]}"}
else
  lib="$out/libmcdma-fabric.$ext"
  # rpc/Makefile's $(OUT)/$(FABRIC) rule: only the mcdma_fabric_* functions are exported
  cc -std=c11 -O2 -Wall -Wextra -Werror "${shared[@]}" -fvisibility=hidden -o "$lib" "${files[@]}" "${verbs[@]}" -lpthread ${extra[@]+"${extra[@]}"}
fi
if command -v sha256sum >/dev/null 2>&1; then sum="$(sha256sum "$lib" | cut -d' ' -f1)"; else sum="$(shasum -a 256 "$lib" | cut -d' ' -f1)"; fi
echo "$lib"
echo "sha256 $sum" >&2
if [ "$ext" = dylib ]; then otool -L "$lib" | tail -n +2 >&2; else ldd "$lib" >&2 || true; fi
