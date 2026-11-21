#!/bin/sh
# Card 166 receipt on TOP's Spark through the Desk's GPU door (scratch, not for merge):
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/drama3-cuda/door-166.sh
# HEAD^ is the PR head b2dbff0a7 (upstream main e702cfdae plus the fixed tools/zig/drama3_compact_gb10.sh). main's script (copied beside it so
# it resolves the same root) must stop on ReplayMismatch; the fixed one must pass. The door's checkout is shallow and
# offline, so the pre-#4144 tree (HEAD with #4144's sources reverted) is rebuilt from door-166-base.patch and checked.
set -eu
pilot=/assets/drama3-pilot
fp8=$pilot/checkpoint
compact=$pilot/drama-3-NVFP4-Pruned
staged=$pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
nvcc=/usr/local/cuda/bin/nvcc
out=/work/out/drama3-166
here=docs/recipes/drama3-cuda
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
(
    cd "$compact"
    echo 'f447d4ba6cbb79ddadbd4148ec3b8eb6b5ea8bf54a13d24f552f93474b0dd07d  MANIFEST.sha256' | sha256sum -c
    sha256sum -c MANIFEST.sha256
)
export GIT_INDEX_FILE=/work/base.index
git read-tree HEAD
git rm -q --cached "$here/door-166.sh" "$here/door-166-base.patch" tools/zig/drama3_compact_gb10_main.sh
git apply --cached "$here/door-166-base.patch"
tree=$(git write-tree)
unset GIT_INDEX_FILE
[ "$tree" = 92decd118c3c7c9ac2a220eaf773c6cf61ec4b3a ] || { echo "the patch gave tree $tree, not HEAD without #4144" >&2; exit 2; }
base=$(echo "HEAD with #4144's sources reverted, rebuilt" | git -c user.name=desk -c user.email=desk@localhost commit-tree "$tree")
echo "commit $(git rev-parse HEAD) pre-feature $base (tree $tree)"
tar -C /work -xJf "$staged/$zig.tar.xz"
tar -C /work -xJf "$staged/$gcc.tar.xz"
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
main=0
tools/zig/drama3_compact_gb10_main.sh "$base" "$fp8" "$compact" "$fp8/codec.pth" "$nvcc" "$out/main" || main=$?
echo "== main's drama3_compact_gb10.sh exit $main"
for f in "$out"/main/*.log; do echo "-- $f"; grep '^codec requested' "$f" || true; tail -n 2 "$f"; done
status=0
tools/zig/drama3_compact_gb10.sh "$base" "$fp8" "$compact" "$fp8/codec.pth" "$nvcc" "$out/fixed" || status=$?
echo "== fixed drama3_compact_gb10.sh exit $status"
[ "$status" -eq 0 ] || for f in "$out"/fixed/*.log; do echo "-- $f"; tail -n 4 "$f"; done
# the door keeps only the log's tail: the receipt goes last
[ "$status" -eq 0 ] && cat "$out/fixed/RECEIPT.txt"
[ "$main" -ne 0 ] && [ "$status" -eq 0 ] && tail -n 1 "$out/main/compact.log" | grep -qx 'error: ReplayMismatch'
echo "RESULT main fails on ReplayMismatch, fixed passes"
