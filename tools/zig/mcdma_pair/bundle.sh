#!/usr/bin/env bash
# The hardware window's kit, so each host is "copy, run, collect": the binaries, scripts, MCDMA's sources and plans
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
zig="${ZIG:-zig}"
src="${TF_MCDMA_SRC:?TF_MCDMA_SRC names the MCDMA tree whose fabric sources go in the bundle (e725b45 or later)}"
[ -f "$src/rpc/libmcdma_fabric.c" ] || { echo "bundle.sh: no rpc/libmcdma_fabric.c in $src" >&2; exit 2; }
# the hosts files, wherever they are (outside the tree), checked before anything is built
for hosts in "$@"; do
  [ -f "$hosts" ] || { echo "bundle.sh: no hosts file $hosts" >&2; exit 2; }
done
(cd "$root" && "$zig" build tf-mcdma-pair-dist)
out="$root/build/mcdma-pair/tf-mcdma-pair-bundle"
# a bundle's settings are only this run's pairs: an earlier set moves aside (nothing is deleted)
if [ -d "$out/settings" ]; then mv "$out/settings" "$out/settings.before-$(date -u +%Y%m%dT%H%M%SZ)"; fi
mkdir -p "$out/mcdma/rpc" "$out/mcdma/tests" "$out/settings"
for t in aarch64-macos aarch64-linux-gnu x86_64-linux-gnu; do
  mkdir -p "$out/bin/$t"
  cp "$root/zig-out/bin/mcdma-pair/$t/tf-mcdma-pair" "$out/bin/$t/"
done
scripts="README.md common.sh run_macos.sh run_linux.sh compare.py compare_log.py settings.py build_mcdma.sh stub_proof.sh loop_fabric.c hosts.example.json"
for f in $scripts; do
  cp "$here/$f" "$out/"
done
# MCDMA's link and fabric sources and its stub verbs, under its own license: enough for build_mcdma.sh on each host
cp "$src/rpc/Makefile" "$src/rpc/"*.c "$src/rpc/"*.h "$out/mcdma/rpc/"
cp "$src/tests/stub_link_verbs.c" "$out/mcdma/tests/"
for f in LICENSE PROVENANCE.md; do [ -f "$src/$f" ] && cp "$src/$f" "$out/mcdma/"; done
sum() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
{
  echo "tensorfold_commit $(git -C "$root" rev-parse HEAD) ($(git -C "$root" status --porcelain | wc -l | tr -d ' ') uncommitted paths at bundling)"
  echo "branch $(git -C "$root" rev-parse --abbrev-ref HEAD)"
  echo "bundled_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "zig $("$zig" version)"
  for t in aarch64-macos aarch64-linux-gnu x86_64-linux-gnu; do echo "binary_$t $(sum "$out/bin/$t/tf-mcdma-pair")"; done
  echo "mcdma_src $src"
  # rpc/Makefile's BUILD_ID: a checksum of the fabric's sources, which fabric-check prints too
  echo "mcdma_build_id $(cd "$src/rpc" && cat libmcdma_fabric.c fabric_bond_n.h fabric_open_qps.h fabric_trace.h fabric_trace_report.h mcdma_fabric.h link_verbs.c link_tb.c link_xchg.c link.h fabric_check.c | cksum | cut -d' ' -f1)"
  echo "libmcdma_fabric.c $(sum "$src/rpc/libmcdma_fabric.c")"
} >"$out/BUILD-INFO"
# this command, run again from here, for PLAN's port switch (settings.py prints it)
TF_MCDMA_PAIR_REBUILD="cd $(printf %q "$PWD") && ZIG=$(printf %q "$zig") TF_MCDMA_SRC=$(printf %q "$src") $(printf %q "$0")$(printf ' %q' "$@")"
export TF_MCDMA_PAIR_REBUILD
for hosts in "$@"; do
  python3 "$here/settings.py" "$hosts" "$out/settings" >/dev/null
  echo "pair $(python3 -c 'import json,sys; h=json.load(open(sys.argv[1])); print(h.get("pair", "-".join(h["hosts"])))' "$hosts") from $(basename "$hosts")" >>"$out/BUILD-INFO"
done
# only what this run put there: no results, proofs, earlier settings, older bundles' files or host library builds
kit=(BUILD-INFO bin mcdma settings)
for f in $scripts; do kit+=("$f"); done
tar -czf "$root/build/mcdma-pair/tf-mcdma-pair-bundle.tgz" --exclude tf-mcdma-pair-bundle/mcdma/build -C "$root/build/mcdma-pair" "${kit[@]/#/tf-mcdma-pair-bundle/}"
echo "$out"
echo "$root/build/mcdma-pair/tf-mcdma-pair-bundle.tgz"
cat "$out/BUILD-INFO" >&2
