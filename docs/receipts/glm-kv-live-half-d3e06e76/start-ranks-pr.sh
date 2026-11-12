#!/usr/bin/env bash
# start-ranks2.sh "ENV_ASSIGNMENTS" [extra serve args]: start rank 1 on node1 then rank 0 here with the given env on
# both, wait for 18080 or a failure. Refuses to launch over a draining rank or a held bench lock.
set -u
penv=${1:-}; shift || true
for i in $(seq 1 40); do a=$(pgrep -f "[t]ensorfold-native serve" | wc -l); b=$(ssh -o BatchMode=yes RANK1_RING_IP "pgrep -f \"[t]ensorfold-native serve\" | wc -l"); [[ "$a" == 0 && "$b" == 0 ]] && break; sleep 3; done
flock -w 90 $GPU_LOCK true || { echo "LOCK HELD on node0"; exit 1; }
ssh -o BatchMode=yes RANK1_RING_IP "flock -w 90 $GPU_LOCK true" || { echo "LOCK HELD on node1"; exit 1; }
echo "launch rank 1 $(date -u +%H:%M:%S)"
ssh -o BatchMode=yes RANK1_RING_IP "cd $WORK/native && (setsid nohup env $penv ./launch-pr.sh 1 $* > rank1.log 2>&1 < /dev/null & disown)"
sleep 2
echo "launch rank 0 $(date -u +%H:%M:%S)"
cd $WORK/native && (setsid nohup env $penv ./launch-pr.sh 0 "$@" > rank0.log 2>&1 < /dev/null & disown)
for i in $(seq 1 60); do pgrep -f "[t]ensorfold-native serve" >/dev/null && break; sleep 2; done
echo "rank0 process seen after $((i*2)) s: $(pgrep -f "[t]ensorfold-native serve" | head -1) at $(date -u +%H:%M:%S)"
t0=$(date +%s)
while :; do
  if curl -s -m 3 http://127.0.0.1:18080/v1/models | grep -q id; then echo "READY after $(( $(date +%s) - t0 )) s"; exit 0; fi
  if ! pgrep -f "[t]ensorfold-native serve" >/dev/null; then echo "RANK0 GONE after $(( $(date +%s) - t0 )) s"; tail -n 8 rank0.log | cut -c1-200; exit 1; fi
  if grep -q -E "error:|refus|panic" rank0.log; then echo "RANK0 ERROR after $(( $(date +%s) - t0 )) s"; grep -E "error:|refus|panic" rank0.log | head -3 | cut -c1-200; exit 1; fi
  if (( $(date +%s) - t0 > 1500 )); then echo "TIMEOUT"; exit 1; fi
  sleep 10
done
