#!/usr/bin/env bash
# The live-half keep on the PR branch's binary (upstream main plus the change): the whole head against the live half,
# drafts off and then on, nine receipt turns each on the same binary; production stopped first and restored at the end.
set -u
N=$HOME/kvprobe/native; OUT=$HOME/kvprobe/receipt-pr; T=$HOME/kvprobe/TensorFold/tools/kvprobe
B=$HOME/kvprobe/TensorFold-pr/zig-out/native/bin/tensorfold-native
log() { echo "=== $1 $(date -u +%H:%M:%S)"; }
mkdir -p $OUT
log "binary $(sha256sum $B | cut -c1-16) node1 $(ssh -o BatchMode=yes RANK1_RING_IP sha256sum '~/kvprobe/TensorFold-pr/zig-out/native/bin/tensorfold-native' | cut -c1-16)"
log "stop production"; (cd ~/glm53-tensorfold-spark && env -i HOME=$HOME PATH=$PATH USER=$USER bash scripts/serve.sh stop 2>&1 | tail -1); sleep 25
run_cfg() { # run_cfg NAME "ENV" [serve args]
  local cfg=$1 penv=$2; shift 2
  log "config $cfg"; mkdir -p $OUT/$cfg
  $N/start-ranks-pr.sh "PROBE_CONTEXT=65536 TF_GLM_HOST_CACHE=1 $penv" "$@" | tail -2
  grep -m1 -E "serving .* drafts" $N/rank0.log | cut -c1-200
  if curl -s -m 3 http://127.0.0.1:18080/v1/models | grep -q id; then
    python3 -u $T/lazy_receipt.py run --base http://127.0.0.1:18080 --config $cfg --out $OUT/$cfg --log $N/rank0.log --lazy-ms 0 --pause 3 2>&1 | tail -12 | cut -c1-200
  fi
  $N/stop-ranks.sh; sleep 3
  cp $N/rank0.log $OUT/$cfg/server-r0.log; ssh -o BatchMode=yes RANK1_RING_IP "cat $N/rank1.log" > $OUT/$cfg/server-r1.log
  python3 -u $T/lazy_receipt.py collect $OUT/$cfg --log $OUT/$cfg/server-r0.log --log $OUT/$cfg/server-r1.log 2>&1 | tail -3 | cut -c1-200
}
run_cfg both "TF_GLM_KV_LIVE_HALF=0" --no-drafts
run_cfg live "TF_GLM_KV_LIVE_HALF=1" --no-drafts
run_cfg both-drafts "TF_GLM_KV_LIVE_HALF=0"
run_cfg live-drafts "TF_GLM_KV_LIVE_HALF=1"
log "compare"; python3 - <<EOF
import json, sys
sys.path.insert(0, "$T")
import lazy_receipt as L
res = L.compare(["$OUT/both", "$OUT/live", "$OUT/both-drafts", "$OUT/live-drafts"])
print(L.table(res))
json.dump(res, open("$OUT/compare.json", "w"), indent=1)
EOF
log "restore production"; (cd ~/glm53-tensorfold-spark && timeout 1200 env -i HOME=$HOME PATH=$PATH USER=$USER bash scripts/serve.sh start 2>&1 | tail -4)
docker ps --format "{{.Names}} {{.Status}}" | grep glm53; curl -s -m 5 http://127.0.0.1:8000/v1/models | grep -o "context_length[^,]*"
log "DONE"
