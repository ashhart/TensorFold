#!/bin/bash
# usage (on host-a): down.sh LABEL; SIGINT both ranks' servers (process name match, not command line), wait, copy rank 1's log.
label=$1
stop='pkill -INT -x tensorfold-nati; for i in $(seq 1 120); do pgrep -x tensorfold-nati >/dev/null || exit 0; sleep 1; done; pkill -9 -x tensorfold-nati'
bash -c "$stop" &
ssh host-b "$stop" &
wait
scp -q host-b:glm-perf/out/$label/server-rank1.log ~/glm-perf/out/$label/ 2>/dev/null
echo "$(date '+%F %T') $label down" >> ~/glm-perf/runs.log
