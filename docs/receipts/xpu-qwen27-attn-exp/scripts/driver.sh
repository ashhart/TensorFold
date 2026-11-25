#!/bin/bash
# The table's order of passes: main, branch, main, branch, each a fresh server on card 0.
# usage: BINMAIN=... BINPR=... MODEL=... DRAFTER=... OUT=... driver.sh
HERE=$(cd "$(dirname "$0")" && pwd)
export ZE_AFFINITY_MASK=0 PLAIN=1 CELLS_GREEDY="8192:code;32768:code;131072:code"
export MODEL DRAFTER OUT
run() { # label bin
  BIN=$2 bash "$HERE/run_pass.sh" "$1" "" || { echo "pass $1 failed"; exit 1; }
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  echo "$(date -Is) pass $1 done"
}
run p1-main "$BINMAIN"
run p2-branch "$BINPR"
run p3-main "$BINMAIN"
run p4-branch "$BINPR"
echo "all passes done"
