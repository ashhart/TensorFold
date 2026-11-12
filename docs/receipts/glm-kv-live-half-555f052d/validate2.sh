#!/usr/bin/env bash
# PR #2181 receipt on the rebased head: the nine-turn conversation, drafts on, on the recipe's checkpoint, four arms:
#   on    PR binary, TF_GLM_KV_LIVE_HALF=1
#   off   PR binary, variable unset
#   fresh PR binary, variable unset, --prompt-cache-gib 0 (every turn prefilled from scratch)
#   main  the PR's base (upstream main) binary, nothing set
# Rank 0's exact kept-state bytes come from /metrics (prompt_cache_bytes at each new state). Production (the MTP
# capture config) is stopped first and restored at the end.
set -u
R=$WORK/receipt2; OUT=$R/out; T=$WORK/TensorFold/tools/kvprobe
BPR=$WORK/bin-pr2/tensorfold-native; BMAIN=$WORK/bin-main/tensorfold-native
export PROBE_MODEL=$(ls -d $MODEL | head -1)
RESTORE_CONFIG=${RESTORE_CONFIG:-config/collect-mtp2.env}
log() { echo "=== $1 $(date -u +%H:%M:%S)"; }
mkdir -p $OUT
for b in $BPR $BMAIN; do
  log "binary $b $(sha256sum $b | cut -c1-16) node1 $(ssh -o BatchMode=yes RANK1_RING_IP sha256sum $b | cut -c1-16)"
done
log "model $PROBE_MODEL"
log "stop production"; (cd $SERVE_DIR && env -i HOME=$HOME PATH=/usr/local/bin:/usr/bin:/bin bash scripts/serve.sh stop 2>&1 | tail -1); sleep 25
poll() { # poll FILE: one line per change of (states, bytes) from rank 0's /metrics
  local last=""; while :; do
    m=$(curl -s -m 2 http://127.0.0.1:18080/metrics | grep -E "^tensorfold:prompt_cache_(states|bytes) " | tr "\n" " ")
    [[ -n "$m" && "$m" != "$last" ]] && { echo "$(date +%s.%N) $m" >> $1; last=$m; }; sleep 0.2; done
}
run_cfg() { # run_cfg NAME BINARY "ENV"
  local cfg=$1 bin=$2 penv=$3
  log "config $cfg"; mkdir -p $OUT/$cfg
  $R/start2.sh "PROBE_MODEL=$PROBE_MODEL PROBE_BINARY=$bin PROBE_CONTEXT=65536 TF_GLM_HOST_CACHE=1 $penv" | tail -2
  grep -m1 -E "serving .* drafts" $R/rank0.log | cut -c1-200
  if curl -s -m 3 http://127.0.0.1:18080/v1/models | grep -q id; then
    poll $OUT/$cfg/metrics.txt & local pp=$!
    python3 -u $T/lazy_receipt.py run --base http://127.0.0.1:18080 --config $cfg --out $OUT/$cfg --log $R/rank0.log \
      --lazy-ms 0 --pause 3 2>&1 | tail -12 | cut -c1-200
    sleep 2; kill $pp
  fi
  $R/stop2.sh; sleep 3
  cp $R/rank0.log $OUT/$cfg/server-r0.log; ssh -o BatchMode=yes RANK1_RING_IP "cat $R/rank1.log" > $OUT/$cfg/server-r1.log
}
run_cfg on    $BPR   "TF_GLM_KV_LIVE_HALF=1"
run_cfg off   $BPR   ""
run_cfg fresh $BPR   "PROBE_PROMPT_CACHE_GIB=0"
run_cfg main  $BMAIN ""
log "compare"; python3 - <<PY
import json, sys
sys.path.insert(0, "$T")
import lazy_receipt as L
dirs = ["$OUT/on", "$OUT/off", "$OUT/fresh", "$OUT/main"]
res = L.compare(dirs)
print(L.table(res))
json.dump(res, open("$OUT/compare.json", "w"), indent=1)
PY
log "restore production ($RESTORE_CONFIG)"
(cd $SERVE_DIR && timeout 1500 env -i HOME=$HOME PATH=/usr/local/bin:/usr/bin:/bin CONFIG=$RESTORE_CONFIG bash scripts/serve.sh start 2>&1 | tail -4)
docker ps --format "{{.Names}} {{.Status}}" | grep glm53; curl -s -m 5 http://127.0.0.1:8000/v1/models | grep -o "context_length[^,]*"
log "DONE"
