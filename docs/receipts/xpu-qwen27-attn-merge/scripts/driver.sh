#!/bin/bash
# bf16 then int8, each A-B-B-A (main, branch, branch, main); stop at the first failure
R=$WORK
for KV in bf16 int8; do
  o=0
  for SIDE in main branch branch main; do
    o=$((o+1))
    $R/scripts/run_ab.sh $KV $o $SIDE || { echo "$(date +%F\ %T) driver: STOP at $KV $o $SIDE" >> $R/journal; exit 1; }
  done
done
echo "$(date +%F\ %T) driver: ALL DONE" >> $R/journal
echo done > $R/driver.done
