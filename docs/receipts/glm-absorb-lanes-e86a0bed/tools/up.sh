#!/bin/bash
# usage (on host-a): up.sh TREE LABEL CTX [VAR=VAL...] -- [serve args...]; starts rank 1 on host-b, rank 0 here, waits for health.
tree=$1 label=$2 ctx=$3; shift 3
. ~/glm-perf/env.sh
out=~/glm-perf/out/$label; [ -e $out ] && { echo "$out exists"; exit 1; }; mkdir -p $out
echo "$(date '+%F %T') $label up $tree $(cat ~/glm-perf/$tree/.tree) ctx $ctx args: $*" >> ~/glm-perf/runs.log
ssh host-b "~/glm-perf/rank.sh 1 $tree $label $ctx $(printf '%q ' "$@")"
sleep 2
~/glm-perf/rank.sh 0 $tree $label $ctx "$@"
t0=$(date +%s)
for i in $(seq 1 400); do
  sleep 3
  curl -s -m 2 127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "$(date '+%F %T') $label health after $(( $(date +%s) - t0 )) s" | tee -a ~/glm-perf/runs.log; exit 0; }
  kill -0 $(cat ~/glm-perf/rank0.pid) 2>/dev/null || break
done
echo "$(date '+%F %T') $label NOT READY" | tee -a ~/glm-perf/runs.log; tail -5 $out/server-rank0.log; exit 1
