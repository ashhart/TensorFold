#!/bin/bash
# The table's passes, one fresh server each, in this order. usage:
#   BINMAIN=... BINHEAD=... MODEL=... DRAFTER=... OUT=... driver.sh
# BINMAIN: main's binary; BINHEAD: the head's. SAMPLED=1 adds the seeded cells (34 and 121,022 tokens);
# CELLS_GREEDY limits a pass to the named cells (a repeat pass of main's 3 and 15).
HERE=$(cd "$(dirname "$0")" && pwd)
run() { # label bin tree sampled [cells]
  CELLS_GREEDY=$5 BIN=$2 PLAIN=0 SAMPLED=$4 bash "$HERE/run_pass.sh" "$1" "$3" || echo "pass $1 failed, going on"
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  echo "$(date -Is) pass $1 done"
}
export MODEL DRAFTER OUT
run p2-main-t15 "$BINMAIN" 15 1
run p3-main-t3 "$BINMAIN" 3 1
run p4-main-t7 "$BINMAIN" 7 0
run p6-main-t3b "$BINMAIN" 3 0 "65536:code;131072:code"
run p7-main-t15b "$BINMAIN" 15 0 "0:code"
run p8-fix "$BINHEAD" "" 1
echo "all passes done"
