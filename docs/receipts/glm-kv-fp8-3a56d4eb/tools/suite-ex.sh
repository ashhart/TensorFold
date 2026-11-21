# Exactness on one server (run.sh runs it in out/LABEL): drafted = plain, concurrent = alone, resumed = cold.
P=$1; T=~/fl-receipt/tools; B=http://127.0.0.1:$P
python3 $T/golden_http.py $P $T/dc.json golden.json > check-golden.log 2>&1; echo "RESULT golden $(tail -1 check-golden.log)"
python3 $T/bench_concurrent.py $B local-model --levels 1,2,4 --tokens 256 --reps 3 --temperatures 0,1 --alone --serial \
  --label ex --output conc.json > check-conc.log 2>&1; echo "RESULT conc rc=$?"
python3 $T/resume_http.py $P resume.json > check-resume.log 2>&1; echo "RESULT resume $(tail -1 check-resume.log)"
