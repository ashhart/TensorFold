#!/bin/bash
# A9 kernel-controls runner (PR 5102 owner ask A9): the committed factor / tail /
# full-chain controls must LAUNCH the real CUDA kernels (not host-math recompute) and
# carry committed fail-before injectors. For each control this runner:
#   1. copies the kernel header from THIS checkout into a scratch dir (the repo tree and
#      its .zig-cache are never modified),
#   2. compiles a9_kernel_controls.cu against the VERBATIM header and runs it: all three
#      controls must print MATCH and exit 0 (GREEN at the checkout),
#   3. applies the control's committed injector (a9_inject_<c>.sh) to the scratch copy,
#      recompiles and runs: THAT control must print FAIL and exit non-zero while the
#      other two still MATCH (RED: deterministic, isolated, non-UB by design).
# Two independent runners = invoke this script from two separate checkouts of the same
# commit (e.g. the workspace and a throwaway git worktree) with different labels; both
# must grade the full GREEN + RED matrix.
#
# Usage:  bash a9_run.sh <label> [repo_root]
# Env:    NVCC (default /usr/local/cuda/bin/nvcc), ARCH (default sm_121)
# Logs:   a9_<label>_<control>_<clean|inject>.log + the injected header copy, next to
#         this script. Exit 0 iff every face graded true.
set -u
LBL="${1:?usage: bash a9_run.sh <label> [repo_root]}"
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${2:-$(cd "$DIR/../../../.." && pwd)}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_121}"
HDR="$ROOT/zig/kernels/cuda/qwen35_draft_lilicorr.cuh"
INC="$ROOT/zig/kernels/cuda"
SC=$(mktemp -d "/tmp/a9_${LBL}.XXXXXX")
trap 'rm -rf "$SC"' EXIT
rc=0
build() { # <out> <include-dir>
  "$NVCC" -O3 -std=c++17 -arch="$ARCH" -I"$2" -I"$INC" "$DIR/a9_kernel_controls.cu" -o "$1"
}
markers() { # <log> <failed-control|none>  — RED must fail ONLY its own control
  local log="$1" redc="$2" fails matches
  fails=$(grep -c "^A9-\(FACTOR\|TAIL\|CHAIN\): FAIL" "$log" || true)
  matches=$(grep -c "^A9-\(FACTOR\|TAIL\|CHAIN\): MATCH" "$log" || true)
  if [ "$redc" = none ]; then
    [ "$fails" -eq 0 ] && [ "$matches" -eq 3 ] && { echo "A9-$LBL clean.markers: OK"; return 0; }
  else
    own=$(grep -ci "A9-${redc}: FAIL" "$log" || true)
    [ "$own" -ge 1 ] && [ "$matches" -eq 2 ] && { echo "A9-$LBL $redc.red.markers: OK (own RED, 2 other MATCH)"; return 0; }
  fi
  echo "A9-$LBL markers($redc): GRADE-FAIL (fails=$fails matches=$matches)"; rc=1
}
mkdir -p "$SC/inc"
for C in factor tail chain; do
  # --- GREEN at the checkout
  cp "$HDR" "$SC/inc/"
  if build "$SC/a9_clean" "$SC/inc" && "$SC/a9_clean" > "$SC/$C.clean.log" 2>&1; then
    echo "A9-$LBL $C.clean.run: OK"
  else
    echo "A9-$LBL $C.clean.run: GRADE-FAIL"; tail -5 "$SC/$C.clean.log"; rc=1
  fi
  cp "$SC/$C.clean.log" "$DIR/a9_${LBL}_${C}_clean.log"
  markers "$SC/$C.clean.log" none
  # --- RED under the committed injector
  rm -f "$SC/inc/qwen35_draft_lilicorr.cuh"; cp "$HDR" "$SC/inc/"
  bash "$DIR/a9_inject_${C}.sh" "$SC/inc" || { echo "A9-$LBL $C.inject.apply: GRADE-FAIL"; rc=1; continue; }
  cp "$SC/inc/qwen35_draft_lilicorr.cuh" "$DIR/a9_${LBL}_${C}_injected_header.cuh"
  if build "$SC/a9_red" "$SC/inc" && "$SC/a9_red" > "$SC/$C.inject.log" 2>&1; then
    echo "A9-$LBL $C.inject.run: GRADE-FAIL (injected binary ran clean)"; rc=1
  fi
  cp "$SC/$C.inject.log" "$DIR/a9_${LBL}_${C}_inject.log"
  markers "$SC/$C.inject.log" "$C"
done
echo "A9-$LBL RUN-RC=$rc"
exit $rc
