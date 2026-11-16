# Speed on one server started with --prompt-cache-gib 0: decode plain and drafted, then cold prompts by size.
P=$1; T=~/fl-receipt/tools; B=http://127.0.0.1:$P
python3 $T/openai_aligned.py nodraft $B local-model --tokens 256 --reps 5 --temperatures 0 --label nodraft \
  --output nodraft.json > check-nodraft.log 2>&1; echo "RESULT nodraft rc=$?"
python3 $T/bench_openai.py $B local-model --tokens 256 --reps 5 --temperatures 0,1 --label drafted \
  --output drafted.json > check-drafted.log 2>&1; echo "RESULT drafted rc=$?"
for n in 2048 8192 9600 32768 38400 65536; do
  python3 $T/bench_cache.py $B local-model --prompt-file $T/p$n.json --tokens 16 --require-cold \
    --output prefill$n.json > check-prefill$n.log 2>&1; echo "RESULT prefill$n rc=$?"
done
