#!/bin/bash
# One server pass. usage: BIN=... MODEL=... DRAFTER=... OUT=... run_pass.sh LABEL [TREE]
# TREE unset or empty: no XPU_TREE (the default). PLAIN=1 adds the plain ("draft": false) mode, SAMPLED=1 the seeded
# sampled cells (34 and 121,022 tokens). PORT defaults to 8091. The load-time window costs go to OUT/LABEL.window.log.
L=$1; T=$2; PORT=${PORT:-8091}; HERE=$(cd "$(dirname "$0")" && pwd)
export LC_ALL=C
[ -n "$T" ] && export XPU_TREE=$T || unset XPU_TREE
unset XPU_KV_INT8
mkdir -p "$OUT"
ss -ltn | grep -q ":$PORT " && { echo "port $PORT busy"; exit 3; }
"$BIN" serve "$MODEL" --drafter "$DRAFTER" --context 132096 --parallel 1 --host 127.0.0.1 --port "$PORT" --name m \
  --no-update-check --prompt-cache-gib 0 > "$OUT/$L.server.log" 2>&1 &
SP=$!
sleep 2; kill -0 $SP 2>/dev/null || { echo "server died at launch"; exit 1; }
for i in $(seq 300); do kill -0 $SP 2>/dev/null || break; curl -sf localhost:$PORT/health >/dev/null && break; sleep 2; done
curl -sf localhost:$PORT/health >/dev/null || { echo "server not ready"; kill $SP 2>/dev/null; exit 1; }
RC=0
CORP=${CORP:-$HERE/../inputs/corp_code.txt}
export CORP
MODES=draft; [ "$PLAIN" = 1 ] && MODES=draft,plain
CELLS="${CELLS_GREEDY:-0:code;32768:code;65536:code;131072:code}" MODES=$MODES python3 -I "$HERE/bench_tree.py" http://127.0.0.1:$PORT "$L" "$OUT" 3 \
  > "$OUT/$L.bench.log" 2>&1 || RC=1
if [ "$SAMPLED" = 1 ]; then
  CELLS="${CELLS_SAMPLED:-0:code;131072:code}" MODES=sampled python3 -I "$HERE/bench_tree.py" http://127.0.0.1:$PORT "$L-sampled" "$OUT" 3 \
    > "$OUT/$L-sampled.bench.log" 2>&1 || RC=1
fi
kill -INT $SP; for i in $(seq 90); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -9 $SP 2>/dev/null
wait $SP 2>/dev/null; ss -ltn | grep -q ":$PORT " && { echo "port $PORT still busy after the pass"; exit 4; }
grep -E "window ms by rows" "$OUT/$L.server.log" > "$OUT/$L.window.log"
exit $RC
