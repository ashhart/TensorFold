#!/bin/bash
# one test binary at a time on card 0; log per binary
export ZE_AFFINITY_MASK=0 LC_ALL=C
R=$WORK
MP=$MODELS/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
for S in main branch; do
  for T in means seam close; do
    echo "$(date "+%F %T") tests $S $T: start" >> $R/journal
    timeout 1800 $R/tests-$S/bin/tf-xpu-qwen27-$T-test $MP > $R/tests-$S/$T.out 2>&1; rc=$?
    echo "$(date "+%F %T") tests $S $T: rc=$rc" >> $R/journal
    sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    [ $rc = 0 ] || { echo "$(date "+%F %T") tests: STOP at $S $T" >> $R/journal; exit 1; }
  done
done
echo "$(date "+%F %T") tests: ALL DONE" >> $R/journal
