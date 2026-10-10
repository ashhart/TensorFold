#!/bin/sh
# usage: serve_tp.sh DEVS WORLD MODEL_DIR PORT [serve flags]: serve_check.py on a tp server, from the build directory.
set -u
devs=$1; world=$2; model=$3; port=$4; shift 4
bin=./zig-out/native-hip/bin/tensorfold-native
here=$(dirname "$0")
log=${TMPDIR:-/tmp}/serve_tp_$port
mkdir -p "$log"
pids=""
trap 'for p in $pids; do kill $p 2>/dev/null; done' EXIT
run() { exec env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=$devs "$bin" serve "$model" --tp "$world" --master 127.0.0.1 --master-port $((port + 1000)) --parallel 4 "$@"; }
r=1
while [ "$r" -lt "$world" ]; do
  run --rank $r "$@" > "$log/rank$r.log" 2>&1 &
  pids="$pids $!"
  r=$((r + 1))
done
run --rank 0 --port "$port" --name m "$@" > "$log/rank0.log" 2>&1 &
head=$!
pids="$pids $head"
tries=0
until curl -s "http://127.0.0.1:$port/health" > /dev/null; do
  tries=$((tries + 1))
  if [ "$tries" -gt 600 ] || ! kill -0 "$head" 2>/dev/null; then echo "server did not start"; tail -n 5 "$log"/rank*.log; exit 1; fi
  sleep 2
done
python3 "$here/serve_check.py" "http://127.0.0.1:$port" m 64
status=$?
kill "$head"
wait "$head" 2>/dev/null
for p in $pids; do wait "$p" 2>/dev/null; done
grep -hE "error|panic|FAIL" "$log"/rank*.log | head -5
exit $status
