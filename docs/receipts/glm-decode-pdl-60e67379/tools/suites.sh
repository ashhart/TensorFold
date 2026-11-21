#!/bin/bash
# usage (in a serve's run folder): suites.sh SUITE REPO PROMPTS; the suites each serve ran, against 127.0.0.1:8090.
# ab: plain decode, cold 8k and 32k prompts; dc: drafted decode, four streams; ex: drafted = plain, concurrent = alone,
# resumed = cold. PROMPTS is this folder's prompts/ (the cold 8k and 32k token lists).
suite=$1 T=$2/tools A=$2/docs/receipts/glm-fp4local-80517fc3/tools P=$3 B=http://127.0.0.1:8090
case $suite in
ab)
  PYTHONPATH=$T python3 $A/openai_aligned.py nodraft $B local-model --tokens 128 --reps 3 --temperatures 0 --label nodraft --output nodraft.json
  for n in 8192 32768; do python3 $T/bench_cache.py $B local-model --prompt-file $P/p$n.json --tokens 16 --require-cold --output prefill$n.json; done ;;
dc)
  python3 $T/bench_openai.py $B local-model --tokens 256 --reps 3 --temperatures 0 --label drafted --output drafted.json
  python3 $T/bench_concurrent.py $B local-model --levels 4 --tokens 256 --reps 2 --temperatures 0 --label c4 --output c4.json ;;
ex)
  python3 $A/golden_http.py 8090 $A/dc.json golden.json
  python3 $T/bench_concurrent.py $B local-model --levels 1,2,4 --tokens 256 --reps 3 --temperatures 0,1 --alone --serial --label ex --output conc.json
  python3 $A/resume_http.py 8090 resume.json ;;
esac
