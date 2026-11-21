# Teacher-forced prompt logprobs (top 20) over the corpus sequences; the first run makes the ids through /tokenize.
P=$1; T=~/fl-receipt/tools; C=~/fl-receipt/corpus
[ -s $C/ids.json ] || python3 $T/mkcorpus.py $P $C/texts.json $C/ids.json $C/plan.json > mkcorpus.log 2>&1
echo "RESULT corpus $(tail -1 mkcorpus.log 2>/dev/null)"
python3 $T/s5_prompt_lp.py $P $C/ids.json lp.jsonl > check-lp.log 2>&1; echo "RESULT lp $(tail -1 check-lp.log)"
sha256sum lp.jsonl
