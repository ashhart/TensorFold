#!/bin/bash
# The table's order of passes. usage: BINMAIN=... BINPR=... MODEL=... DRAFTER=... OUT=... driver.sh
# (each pass is one fresh server; PLAIN=1 on main's tree-15 passes, SAMPLED=1 on passes 1, 3 and 4)
HERE=$(cd "$(dirname "$0")" && pwd)
run() { # label bin tree plain sampled
  BIN=$2 PLAIN=$4 SAMPLED=$5 bash "$HERE/run_pass.sh" "$1" "$3" || { echo "pass $1 failed"; exit 1; }
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  echo "$(date -Is) pass $1 done"
}
export MODEL DRAFTER OUT
run p1-main-t15 "$BINMAIN" 15 1 1
run p2-main-t7 "$BINMAIN" 7 0 0
run p3-main-t3 "$BINMAIN" 3 0 1
run p4-pr "$BINPR" "" 0 1
run p5-pr "$BINPR" "" 0 0
run p6-main-t3 "$BINMAIN" 3 0 0
run p7-main-t7 "$BINMAIN" 7 0 0
run p8-main-t15 "$BINMAIN" 15 1 0
echo "all passes done"
