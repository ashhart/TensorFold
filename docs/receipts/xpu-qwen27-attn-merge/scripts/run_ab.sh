#!/bin/bash
# usage: run_ab.sh KV(bf16|int8) ORDER(1..4) SIDE(main|branch)   one card (0), one server, one pass of the cells
KV=$1; ORD=$2; SIDE=$3
R=$WORK; RUNS=$R/runs; mkdir -p $RUNS
export ZE_AFFINITY_MASK=0 XPU_GUARD_PDEV=0000:04:00.0 LC_ALL=C
[ "$KV" = int8 ] && export XPU_KV_INT8=1 || unset XPU_KV_INT8
BIN=$R/bin-$SIDE/tensorfold-native
MP=$MODELS/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
N=$KV-$ORD-$SIDE
step() { echo "$(date +%F\ %T) run $N: $*" >> $R/journal; }
step "server"
$BIN serve $MP --drafter $MODELS/Qwen3.8-27B-DFlash2 --context 132096 --parallel 1 --host 127.0.0.1 --port 8091 --name m --no-update-check --prompt-cache-gib 0 > $RUNS/$N.server.log 2>&1 &
SP=$!
for i in $(seq 300); do curl -sf localhost:8091/health >/dev/null && break; sleep 2; done
curl -sf localhost:8091/health >/dev/null || { step "server KO"; kill $SP; exit 1; }
step "ready pid $SP"
python3 -I $R/scripts/bench_merge.py http://127.0.0.1:8091 $N $RUNS 3 > $RUNS/$N.bench.log 2>&1; RC=$?
step "bench rc=$RC"
kill -INT $SP; for i in $(seq 90); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -9 $SP 2>/dev/null
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
step "stopped, page cache dropped"
[ $RC = 0 ] || { step "FAILED (a request without fingerprint or a bench error)"; exit 1; }
python3 -I $R/scripts/compare.py $RUNS /dev/null > $RUNS/.cmp.out 2>&1 || { step "FINGERPRINT MISMATCH: stop"; exit 2; }
step "ok, fingerprints equal so far"
