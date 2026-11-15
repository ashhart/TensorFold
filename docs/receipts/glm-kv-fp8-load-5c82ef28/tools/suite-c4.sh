# Decode at the widest admitted stream count (4), greedy and sampled, drafted; no alone or serial replays.
P=$1; T=~/fl-receipt/tools; B=http://127.0.0.1:$P
python3 $T/bench_concurrent.py $B local-model --levels 4 --tokens 256 --reps 3 --temperatures 0,1 \
  --label c4 --output c4.json > check-c4.log 2>&1; echo "RESULT c4 rc=$?"
