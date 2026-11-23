#!/bin/bash
# usage (on rank 0's host): run.sh TREE LABEL SUITE CTX [VAR=VAL...] -- [serve args...]
# Starts rank 1 on the peer, then rank 0 here, waits for health, runs suite-SUITE.sh, stops both; logs in out/LABEL.
tree=$1 label=$2 suite=$3 ctx=$4; shift 4
. ~/fl-receipt/fl-env.sh
out=~/fl-receipt/out/$label; [ -e $out ] && { echo "$out exists"; exit 1; }; mkdir -p $out
log() { echo "$(date '+%F %T') $label $*" | tee -a ~/fl-receipt/runs.log; }
log "serve $tree $(cd ~/fl-receipt/$tree && git log -1 --format=%h 2>/dev/null || cat .tree 2>/dev/null) ctx $ctx args: $*"
free -g | sed -n 2p > $out/free-before.txt; ssh host-b 'free -g | sed -n 2p' >> $out/free-before.txt
ssh host-b "~/fl-receipt/rank.sh 1 $tree $label $ctx $(printf '%q ' "$@")"
sleep 2
~/fl-receipt/rank.sh 0 $tree $label $ctx "$@"
t0=$(date +%s); ok=0
for i in $(seq 1 400); do
  sleep 3
  curl -s -m 2 127.0.0.1:$PORT/health >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $(cat ~/fl-receipt/rank0.pid) 2>/dev/null || break
done
if [ $ok = 1 ]; then
  log "health after $(( $(date +%s) - t0 )) s"
  free -g | sed -n 2p > $out/free-ready.txt; ssh host-b 'free -g | sed -n 2p' >> $out/free-ready.txt
  (cd $out && bash ~/fl-receipt/suite-$suite.sh $PORT) > $out/suite.log 2>&1
  log "suite $suite rc=$? : $(grep -h '^RESULT' $out/suite.log | tr '\n' ' ')"
else
  log "NOT READY"; tail -5 $out/mem-rank0/process.log
fi
~/fl-receipt/stop.sh 0; ssh host-b '~/fl-receipt/stop.sh 1'
sleep 3; scp -q -r host-b:fl-receipt/out/$label/mem-rank1 $out/ 2>/dev/null
for r in 0 1; do echo "rank $r rc=$(grep -o '"returncode": [-0-9]*' $out/mem-rank$r/receipt.json 2>/dev/null)"; done | tee -a $out/stop.txt
log "serve END"
