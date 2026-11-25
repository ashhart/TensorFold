#!/bin/bash
# usage: stop.sh RANK; SIGINT to the wrapper rank.sh started (it forwards to the server), then waits up to 120 s.
pid=$(cat ~/fl-receipt/rank$1.pid 2>/dev/null) || exit 0
kill -INT $pid 2>/dev/null
for i in $(seq 1 120); do kill -0 $pid 2>/dev/null || exit 0; sleep 1; done
echo "rank $1 still up after 120 s; SIGKILL"; pkill -9 -P $pid; kill -9 $pid
