#!/bin/bash
# A-B-B-A: main, branch, branch, main
o=0
for SIDE in main branch branch main; do
  o=$((o+1))
  $WORK/scripts/run_pass.sh $o $SIDE || { echo "$(date +%T) driver: STOP at pass $o $SIDE" >> $WORK/journal; exit 1; }
done
echo "$(date +%T) driver: ALL DONE" >> $WORK/journal
echo done > $WORK/driver.done
