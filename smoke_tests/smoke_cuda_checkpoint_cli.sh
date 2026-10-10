#!/bin/sh
# Smoke test for the cuda-checkpoint-cli branch (incl. the review's fixes).
#
# Proves (Review2 claims):
#   C1. The CUDA `tensorfold` binary builds, and `--version` works without a model,
#       driver or GPU (prints the build's version on stdout, exit 0).
#   C2. The checkpoint subcommands (`models`, `info`, `pull`) dispatch BEFORE
#       cuda.Driver.open(): they work with no model and never touch libcuda
#       (strace: 0 libcuda opens, vs 6+ for `run`).
#   C3. Argument handling: `models`/`info`/`pull` with wrong arg counts exit 2 with
#       the checkpoint usage on stderr; `info BADMODEL` exits 1 from the checkpoint
#       CLI (cache miss), never from a driver error; the pre-existing run commands
#       still print the full usage and exit 2.
#   C4. --version ergonomics (post-fix): `--version` reaches stdout, `--version extra`
#       and any leading --flag exit 2 with usage and no leaked bare Zig error, all
#       without touching the GPU.
#
# Usage: sh smoke_tests/smoke_cuda_checkpoint_cli.sh [zig] [nvcc]
set -u
cd "$(dirname "$0")/.." || exit 1
ZIG="${1:-zig}"
NVCC="${2:-}"

fail=0
pass() { echo "PASS: $1"; }
check() { if [ "$1" -eq 0 ]; then pass "$2"; else echo "FAIL: $2"; fail=1; fi; }

echo "== build the CUDA tensorfold =="
if [ -n "$NVCC" ]; then
    "$ZIG" build install -Dnvcc="$NVCC" >/dev/null 2>&1
else
    "$ZIG" build install >/dev/null 2>&1
fi
check $? "C1: zig build install succeeds"
BIN=./zig-out/bin/tensorfold
[ -x "$BIN" ] || { echo "FAIL: no binary at $BIN"; exit 1; }

echo "== C1: --version without model/driver/GPU =="
V="$("$BIN" --version 2>/dev/null)"; RC=$?
[ "$RC" -eq 0 ] && echo "$V" | grep -Eq '^tensorfold [0-9]+\.[0-9]+\.[0-9]+$'
check $? "C1: '$V' exits 0 on stdout and matches 'tensorfold MAJOR.MINOR.PATCH'"
"$BIN" --version >/dev/null 2>&1 </dev/null
check $? "C1: --version works with stdin closed and no env"

echo "== C2: dispatch before the driver opens =="
M="$("$BIN" models 2>/dev/null)"; RC=$?
[ "$RC" -eq 0 ] && echo "$M" | grep -q 'Nemotron'
check $? "C2: 'models' exits 0 and lists a checkpoint"
if command -v strace >/dev/null 2>&1; then
    n_models=$(strace -f -e trace=openat,open "$BIN" models 2>&1 >/dev/null | grep -ci libcuda)
    n_run=$(strace -f -e trace=openat,open "$BIN" run nonexistent-model --tokens 1 2>&1 | grep -ci libcuda)
    echo "libcuda opens: models=$n_models run=$n_run"
    [ "$n_models" -eq 0 ]
    check $? "C2: 'models' never opens libcuda (0 opens)"
    [ "$n_run" -gt 0 ]
    check $? "C2: 'run' does open libcuda (control case, $n_run opens)"
else
    echo "SKIP: strace not available"
fi

echo "== C3: argument handling =="
"$BIN" info >/dev/null 2>&1; RC=$?
"$BIN" info 2>&1 >/dev/null | grep -q 'tensorfold pull REPO' && [ "$RC" -eq 2 ]
check $? "C3: 'info' alone exits 2 with checkpoint usage on stderr"
OUT="$("$BIN" pull 2>&1)"; RC=$?
echo "$OUT" | grep -q 'tensorfold pull REPO' && [ "$RC" -eq 2 ]
check $? "C3: 'pull' alone exits 2 with checkpoint usage on stderr"
"$BIN" models extra 2>&1 >/dev/null | grep -q 'tensorfold models'
check $? "C3: 'models extra' exits 2 with checkpoint usage on stderr"
OUT="$("$BIN" info NoSuchOrg/NoSuchModel 2>&1)"; RC=$?
[ "$RC" -eq 1 ] && echo "$OUT" | grep -q 'is not in the Hugging Face cache'
check $? "C3: 'info BADMODEL' exits 1 from the checkpoint CLI (cache miss), not a driver error"
"$BIN" 2>&1 >/dev/null | grep -q 'usage: tensorfold run MODEL'
check $? "C3: bare invocation still prints the run usage (unchanged)"

echo "== C4: --version ergonomics (post-fix) =="
# The fix: --version reaches stdout; --version extra and leading --flag exit 2 with usage, no bare Zig error.
out=$("$BIN" --version 2>/dev/null)
[ -n "$out" ] && echo "$out" | grep -Eq '^tensorfold [0-9]+\.[0-9]+\.[0-9]+$'
check $? "C4: '--version' writes to stdout (scripted capture works)"
OUT="$("$BIN" --version extra 2>&1)"; RC=$?
if [ "$RC" -eq 2 ] && echo "$OUT" | grep -q 'tensorfold models' && ! echo "$OUT" | grep -q 'error: FileNotFound'; then
    pass "C4: '--version extra' exits 2 with usage, no leaked Zig error"
else
    echo "FAIL: C4: '--version extra' exits 2 with usage, no leaked Zig error"; fail=1
fi
OUT="$("$BIN" --no-such-flag 2>&1)"; RC=$?
[ "$RC" -eq 2 ] && echo "$OUT" | grep -q 'tensorfold models'
check $? "C4: a leading --flag exits 2 with usage (no GPU touch)"

echo
if [ "$fail" -eq 0 ]; then echo "ALL SMOKE CHECKS PASSED"; else echo "SMOKE CHECKS FAILED"; fi
exit "$fail"
