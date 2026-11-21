#!/bin/sh
# Drama 3 compact NVFP4 on one GB10, the checks host tests cannot make; RECEIPT.txt and every log go to RECEIPT_DIR.
set -eu
[ "$#" -eq 6 ] || { echo 'usage: drama3_compact_gb10.sh PRE_FEATURE_COMMIT FP8_MODEL COMPACT_MODEL CODEC NVCC RECEIPT_DIR' >&2; exit 2; }
base=$1 fp8=$2 compact=$3 codec=$4 nvcc=$5 out=$6
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
# the FP8 comparison needs a tree without the feature: once merged, merge-base with main is HEAD itself
if git -C "$root" grep -q d3_router_compact "$base" -- zig/kernels/cuda; then echo "$base already has d3_router_compact" >&2; exit 2; fi
mkdir -p "$out"
out=$(CDPATH= cd -- "$out" && pwd)
main_tree=$(mktemp -d "${TMPDIR:-/tmp}/drama3-main.XXXXXX")
trap 'git -C "$root" worktree remove --force "$main_tree"' EXIT
git -C "$root" worktree add --detach "$main_tree" "$base" >/dev/null
for tree in "$root" "$main_tree"; do
    (cd "$tree" && zig build drama3-generate -Dsm=121 -Dnvcc="$nvcc" -Doptimize=ReleaseFast)
done
(cd "$root" && zig build install fatbins kernel-contract-fatbins -Dsm=80,86,89,120,121 -Dnvcc="$nvcc" -Doptimize=ReleaseFast)
text='The quick brown fox jumps over the lazy dog.'
run() { # TREE MODEL NAME [ARGS...]: greedy codes into NAME.codes, stderr into NAME.log
    bin=$1/zig-out/bin/drama3-generate model=$2 name=$3
    shift 3
    "$bin" --model "$model" --text "$text" --greedy --max-frames 200 --out "$out/$name.codes" "$@" 2>"$out/$name.log"
    tail -n 1 "$out/$name.log" | grep -q '^RESULT drama3 OK'
}
run "$main_tree" "$fp8" fp8-main
run "$root" "$fp8" fp8-head
cmp "$out/fp8-main.codes" "$out/fp8-head.codes"
run "$root" "$compact" compact --codec "$codec" --replay 1
# a pruned checkpoint compacts only the layers that dropped experts: NVFP4 in all 40 and at least one compact router
grep -q 'routed experts NVFP4 in 40 of 40 layers; compact routers in [1-9][0-9]* of 40 layers' "$out/compact.log"
# batched prefill is gated within its own capacity: cross-capacity frames may differ near a tie (docs/drama3-prefill.md)
run "$root" "$compact" compact-batched --codec "$codec" --max-rows 64 --replay 1
cmp -s "$out/compact.codes" "$out/compact-batched.codes" && capacity=equal || capacity=differ
# gates 1-4 and the unmasked control; the word matches no limits record, so the pending one waits for the audit below
"$root/zig-out/bin/tf-cuda-test" kernel-contract 'compact router' >"$out/contract.log" 2>&1
status=0
"$root/zig-out/bin/tf-cuda-test" kernel-contract-limits "$root/zig-out/fatbin" >"$out/limits.log" 2>&1 || status=$?
"$root/tools/zig/drama3_limits_gate.sh" "$status" "$out/limits.log"
{
    echo "head $(git -C "$root" rev-parse HEAD) pre-feature $base"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    "$nvcc" --version | tail -n 1
    for f in "$fp8" "$compact" "$codec"; do echo "input $f: $(find "$f" -name '*.safetensors' -o -name '*.pth' | sort | xargs cat | sha256sum | cut -d' ' -f1)"; done
    echo "PASS fp8 greedy codes byte-identical to $base: $(sha256sum "$out/fp8-head.codes" | cut -d' ' -f1)"
    echo "PASS compact $(grep -o 'routed experts held.*' "$out/compact.log")"
    echo "PASS compact decode and replay: $(tail -n 1 "$out/compact.log")"
    echo "PASS compact batched prefill (64 rows) decode and replay: $(tail -n 1 "$out/compact-batched.log")"
    echo "INFO compact codes at 1 and 64 prompt rows $capacity"
    grep '^(filtered) SUMMARY' "$out/contract.log"
    grep '^RECORD drama3 d3_router_compact:' "$out/limits.log"
} >"$out/RECEIPT.txt"
cat "$out/RECEIPT.txt"
