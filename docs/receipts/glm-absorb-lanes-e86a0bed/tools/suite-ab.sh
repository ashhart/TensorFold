#!/bin/bash
# usage (in out/LABEL on host-a): suite-ab.sh TREE; plain decode, then cold prompts of 8k and 32k with 16 drafted tokens.
T=~/glm-perf/$1/tools; A=~/glm-perf/$1/docs/receipts/glm-fp4local-80517fc3/tools; B=http://127.0.0.1:8090; P=~/glm-perf/prompts
PYTHONPATH=$T python3 $A/openai_aligned.py nodraft $B local-model --tokens 128 --reps 3 --temperatures 0 --label nodraft --output nodraft.json > check-nodraft.log 2>&1; echo "RESULT nodraft rc=$?"
for n in 8192 32768; do
  python3 $T/bench_cache.py $B local-model --prompt-file $P/p$n.json --tokens 16 --require-cold --output prefill$n.json > check-prefill$n.log 2>&1; echo "RESULT prefill$n rc=$?"
done
