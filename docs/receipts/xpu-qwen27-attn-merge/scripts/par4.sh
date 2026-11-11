#!/bin/bash
# four simultaneous streams, --parallel 4: bf16 then int8, each main then branch; stop at the first failure
R=$WORK; RUNS=$R/par4; mkdir -p $RUNS
export ZE_AFFINITY_MASK=0 LC_ALL=C
CTX=40000
step() { echo "$(date +%F\ %T) par4: $*" >> $R/journal; }
for KV in bf16 int8; do
  for SIDE in main branch; do
    N=$KV-$SIDE
    [ "$KV" = int8 ] && export XPU_KV_INT8=1 || unset XPU_KV_INT8
    step "$N server"
    $R/bin-$SIDE/tensorfold-native serve $MODELS/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 --drafter $MODELS/Qwen3.8-27B-DFlash2 --context $CTX --parallel 4 --host 127.0.0.1 --port 8091 --name m --no-update-check --prompt-cache-gib 0 > $RUNS/$N.server.log 2>&1 &
    SP=$!
    for i in $(seq 150); do curl -sf localhost:8091/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
    curl -sf localhost:8091/health >/dev/null || { step "$N server KO"; kill $SP; exit 1; }
    step "$N ready pid $SP, $(grep -o '[0-9]* streams of [0-9]* tokens' $RUNS/$N.server.log)"
    python3 -I $R/scripts/par4.py http://127.0.0.1:8091 par4-$N $RUNS 3 > $RUNS/$N.bench.log 2>&1; RC=$?
    step "$N bench rc=$RC"
    kill -INT $SP; for i in $(seq 90); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -9 $SP 2>/dev/null
    sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    step "$N stopped, page cache dropped"
    [ $RC = 0 ] || { step "$N FAILED"; exit 1; }
  done
done
step "ALL DONE"
