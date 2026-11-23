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
# --replay 1 gates frames, graph, poison/reset and chunk 1 vs 1; main's CUDA codec may differ at chunk 4 or 16 (below)
replay() { # NAME [ARGS...]: exact checks pass; codec PCM differences stay within replay.zig's bound
    name=$1
    shift
    run "$root" "$compact" "$name" --codec "$codec" --replay 1 "$@" && return
    "$root/tools/zig/drama3_replay_gate.sh" "$out/$name.log"
}
replay compact-replay
# a pruned checkpoint compacts only the layers that dropped experts: NVFP4 in all 40 and at least one compact router
grep -q 'routed experts NVFP4 in 40 of 40 layers; compact routers in [1-9][0-9]* of 40 layers' "$out/compact-replay.log"
replay compact-batched-replay --max-rows 64
# the codec chunk never changes the frames; the PCM it changes is information, in 16-bit steps against chunk 1
for chunk in 1 4 16; do
    run "$root" "$compact" "compact-$chunk" --codec "$codec" --first-chunk "$chunk" --chunk "$chunk" --pcm-out "$out/compact-$chunk.pcm"
    cmp "$out/compact-1.codes" "$out/compact-$chunk.codes"
done
for chunk in 4 16; do # one command each, so a missing or short PCM stops the run under set -e
    python3 - "$out/compact-1.pcm" "$out/compact-$chunk.pcm" >"$out/pcm-$chunk.txt" <<'EOF'
import array, math, sys
a, b = (array.array('h', open(p, 'rb').read()) for p in sys.argv[1:])
assert len(a) == len(b), (len(a), len(b))
d = [x - y for x, y in zip(a, b)]
print(f'max abs {max(map(abs, d))}, RMS {math.sqrt(sum(x * x for x in d) / len(d)):.4f} over {len(d)} samples')
EOF
done
# batched prefill is gated within its own capacity: cross-capacity frames may differ near a tie (docs/drama3-prefill.md)
run "$root" "$compact" compact-batched --codec "$codec" --max-rows 64
cmp -s "$out/compact-1.codes" "$out/compact-batched.codes" && capacity=equal || capacity=differ
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
    echo "PASS compact $(grep -o 'routed experts held.*' "$out/compact-replay.log")"
    echo "PASS compact replay exact through PCM chunk 1 vs 1, then: $(tail -n 1 "$out/compact-replay.log")"
    echo "PASS compact batched prefill (64 rows) replay exact through PCM chunk 1 vs 1, then: $(tail -n 1 "$out/compact-batched-replay.log")"
    echo "PASS compact decode at codec chunk 1, frames byte-identical at chunks 4 and 16: $(tail -n 1 "$out/compact-1.log")"
    for chunk in 4 16; do echo "INFO compact PCM at codec chunk $chunk against 1, 16-bit steps: $(cat "$out/pcm-$chunk.txt")"; done
    echo "INFO compact codes at 1 and 64 prompt rows $capacity"
    grep '^(filtered) SUMMARY' "$out/contract.log"
    grep '^RECORD drama3 d3_router_compact:' "$out/limits.log"
} >"$out/RECEIPT.txt"
cat "$out/RECEIPT.txt"
