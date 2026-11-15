# FP8 load A/B on one server started with --prompt-cache-gib 0: plain decode, then cold prompts with 256 drafted tokens.
P=$1; T=~/fl-receipt/tools; B=http://127.0.0.1:$P
python3 $T/openai_aligned.py nodraft $B local-model --tokens 256 --reps 5 --temperatures 0 --label nodraft \
  --output nodraft.json > check-nodraft.log 2>&1; echo "RESULT nodraft rc=$?"
for n in 2048 8192 32768 65536; do
  python3 $T/bench_cache.py $B local-model --prompt-file $T/p$n.json --tokens 256 --require-cold \
    --output prefill$n.json > check-prefill$n.log 2>&1; echo "RESULT prefill$n rc=$?"
done
