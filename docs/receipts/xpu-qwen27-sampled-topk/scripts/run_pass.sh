#!/bin/bash
# usage: run_pass.sh ORDER(1..4) SIDE(main|branch)   one card (0), the server at 1, 4 and 8 streams, bf16 KV, one pass of the cells
ORD=$1; SIDE=$2
R=$WORK; RUNS=$R/runs; mkdir -p $RUNS
export ZE_AFFINITY_MASK=0 LC_ALL=C
unset XPU_KV_INT8
BIN=$R/bin-$SIDE/tensorfold-native
MP=$MODELS/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
PORT=8091
J=$R/journal
step() { echo "$(date +%T) pass $ORD $SIDE: $*" >> $J; }
for P in 1 4 8; do
  N=p$ORD-$SIDE-par$P
  ss -ltn | grep -qE ":$PORT\b" && { step "port $PORT busy"; exit 1; }
  step "par$P server"
  $BIN serve $MP --drafter $MODELS/Qwen3.8-27B-DFlash2 --context 10240 --parallel $P --host 127.0.0.1 --port $PORT --name m --no-update-check --prompt-cache-gib 0 > $RUNS/$N.server.log 2>&1 &
  SP=$!
  for i in $(seq 150); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
  kill -0 $SP 2>/dev/null && curl -sf localhost:$PORT/health >/dev/null || { step "par$P server KO"; exit 1; }
  step "par$P ready pid $SP"
  ( while kill -0 $SP 2>/dev/null; do sleep 240; kill -0 $SP 2>/dev/null && step "par$P running: $(tail -n1 $RUNS/$N.bench.log | cut -c1-120)"; done ) &
  HB=$!
  if [ $P = 1 ]; then
    GREEDY=1 CELLS="0:code,text" python3 -I $R/scripts/bench_topk.py http://127.0.0.1:$PORT $N-greedy $RUNS $P 0 > $RUNS/$N-greedy.bench.log 2>&1; RCG=$?
    step "par$P greedy rc=$RCG"
  fi
  python3 -I $R/scripts/bench_topk.py http://127.0.0.1:$PORT $N $RUNS $P 3 > $RUNS/$N.bench.log 2>&1; RC=$?
  step "par$P bench rc=$RC"
  kill $HB 2>/dev/null
  kill -INT $SP; for i in $(seq 90); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -0 $SP 2>/dev/null && kill -9 $SP
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  step "par$P stopped, page cache dropped"
  [ $RC = 0 ] || { step "FAILED"; exit 1; }
done
