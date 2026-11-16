#!/bin/bash
# usage (in out/LABEL on host-a): suite-ex.sh TREE; drafted = plain (golden), concurrent = alone, resumed = cold.
T=~/glm-perf/tools; P=8090; B=http://127.0.0.1:$P
python3 $T/golden_http.py $P $T/dc.json golden.json > check-golden.log 2>&1; echo "RESULT golden $(tail -1 check-golden.log)"
python3 $T/bench_concurrent.py $B local-model --levels 1,2,4 --tokens 256 --reps 3 --temperatures 0,1 --alone --serial \
  --label ex --output conc.json > check-conc.log 2>&1; echo "RESULT conc rc=$?"
python3 $T/resume_http.py $P resume.json > check-resume.log 2>&1; echo "RESULT resume $(tail -1 check-resume.log)"
