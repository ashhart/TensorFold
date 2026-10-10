#!/bin/sh
# F1: `tensorfold --version` prints to stderr, not stdout (std.debug.print).
# Convention: --version belongs on stdout; `$(tensorfold --version)` and
# `tensorfold --version | grep` capture nothing today.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$HERE/../zig-out/bin/tensorfold}

out=$("$BIN" --version 2>/dev/null)
if [ -n "$out" ]; then
    echo "PASS: --version writes to stdout: $out"
else
    echo "FAIL: --version stdout is empty; the version banner goes to stderr instead:"
    "$BIN" --version 2>&1 1>/dev/null | sed 's/^/    stderr: /'
    exit 1
fi
