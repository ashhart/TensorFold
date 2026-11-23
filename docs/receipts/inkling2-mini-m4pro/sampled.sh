#!/bin/bash
cd REPO
CK=CHECKPOINT
BIN=BUILD; O=OUT
for mode in plain1 drafted1 drafted2 plain2; do
  EXTRA=""; case $mode in plain*) EXTRA="--no-drafts";; esac
  $BIN serve "$CK" --host 127.0.0.1 --port 8141 --context 8192 --parallel 4 --prompt-cache-gib 0 --no-thinking $EXTRA > $O/srv-$mode.log 2>&1 &
  SP=$!
  for i in $(seq 1 300); do grep -q "serving " $O/srv-$mode.log 2>/dev/null && break; sleep 1; done
  echo "== $mode"
  python3 tools/bench_concurrent.py http://127.0.0.1:8141/v1 m --alone --serial --levels 1,4 --tokens 256 --reps 3 --temperatures 0.7 --seed 1234 --label sampled-$mode --output $O/$mode.json > $O/$mode.out 2>&1
  echo "client exit $?"
  kill $SP; wait $SP 2>/dev/null; sleep 3
done
echo ALLDONE
