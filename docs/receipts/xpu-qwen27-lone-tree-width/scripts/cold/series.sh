#!/bin/bash
# Cold-start acceptance series for #3052 (lone-stream tree width): the FIRST request of a fresh server, no warm-up.
# Arms: A = adaptive (head binary, no XPU_TREE), F15 / F7 / F3 = main binary with XPU_TREE=15 / 7 / 3.
# Prompts, in this order: short (34 tokens), long (7,588), p28946 (28,946), p121022 (121,022).
# One cell = one fresh server, one request, then the server is stopped.
# Each prompt runs SERVERS rounds; the arm order alternates by round (A F15 F7 F3, then F3 F7 F15 A, ...).
# usage: series.sh [--dry-run]
# env (required): BHEAD BMAIN (binaries) MODEL DRAFTER CORP (corp_code.txt, see build_corp_code.sh); optional: CARD=0
# SERVERS=3 PORT=8093 OUT=<dir>. The card's driver libraries must already be on the library path.
HERE=$(cd "$(dirname "$0")" && pwd)
CARD=${CARD:-0}; SERVERS=${SERVERS:-3}; PORT=${PORT:-8093}
: "${BHEAD:?}" "${BMAIN:?}" "${MODEL:?}" "${DRAFTER:?}" "${CORP:?}"
OUT=${OUT:-$PWD/cold-$(date +%Y%m%d-%H%M)}
DRY=0; [ "$1" = "--dry-run" ] && DRY=1
ARMS=(A F15 F7 F3); PROMPTS=(short long p28946 p121022)
arm_bin() { case $1 in A) echo "$BHEAD";; *) echo "$BMAIN";; esac; }
arm_tree() { case $1 in A) echo "";; F15) echo 15;; F7) echo 7;; F3) echo 3;; esac; }
cells() { # prints "prompt round arm" in run order
  for p in "${PROMPTS[@]}"; do for r in $(seq "$SERVERS"); do
    if [ $((r % 2)) = 1 ]; then for a in "${ARMS[@]}"; do echo "$p $r $a"; done
    else for a in F3 F7 F15 A; do echo "$p $r $a"; done; fi
  done; done
}
# coarse: the 34- and 7,588-token cells took ~40 s each, the deep ones are longer (prefill)
NCELLS=$(cells | wc -l); PER=${PER_CELL_S:-110}
echo "cells: $NCELLS (${#PROMPTS[@]} prompts x ${#ARMS[@]} arms x $SERVERS fresh servers)," \
  "about $((NCELLS * PER / 60)) min at ${PER}s a cell; out: $OUT"
if [ $DRY = 1 ]; then
  cells | awk '{t = ($3 == "A" ? "default (adaptive)" : substr($3, 2))
    printf "%2d  %-5s round %s  %-3s  tree=%s\n", NR, $1, $2, $3, t}'
  for f in "$BHEAD" "$BMAIN" "$MODEL" "$DRAFTER" "$CORP" "$HERE"/{cold_request,summarize,mkprompts}.py; do
    [ -e "$f" ] && echo "ok      $f" || echo "MISSING $f"; done
  ss -ltn | grep -q ":$PORT " && echo "PORT $PORT BUSY" || echo "port $PORT free"
  pgrep -af 'tensorfold-nativ[e]' | cut -c1-110 | sed 's/^/running: /'
  python3 -I "$HERE/mkprompts.py" "$CORP" "$(mktemp -d)" || echo "PROMPT CHECK FAILED"
  echo "dry run: no server started, no lock taken"; exit 0
fi
export LC_ALL=C ZE_AFFINITY_MASK=$CARD
unset XPU_KV_INT8
mkdir -p "$OUT/raw" "$OUT/prompts"; LOG=$OUT/driver.log
log() { echo "$(date -Is) $*" >> "$LOG"; }
python3 -I "$HERE/mkprompts.py" "$CORP" "$OUT/prompts" >> "$LOG" 2>&1 || { log "prompts: check failed"; exit 1; }
for f in "$BHEAD" "$BMAIN"; do [ -x "$f" ] || { log "binary missing $f"; exit 1; }; done
sha256sum "$BHEAD" "$BMAIN" > "$OUT/binaries.sha256"
exec 9>/tmp/xpu-card$CARD.lock; log "waiting for the card $CARD lock"; flock 9; log "lock taken"
trap 'log DRIVER_EXIT' EXIT
busy() { # another tensorfold-native on this card: its pid, or nothing
  for p in $(pgrep -f 'tensorfold-nativ[e] serve'); do
    mask=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^ZE_AFFINITY_MASK=//p')
    [ "$mask" = "$CARD" ] && echo $p
  done
}
n=0
while read -r P R A; do
  n=$((n + 1)); L=$P-$A-s$R; B=$(arm_bin $A); T=$(arm_tree $A)
  ss -ltn | grep -q ":$PORT " && { log "port $PORT busy, stop"; exit 3; }
  [ -n "$(busy)" ] && { log "card $CARD held by pid $(busy), stop"; exit 4; }
  log "cell $n/$NCELLS $L start (tree=${T:-default})"
  ( [ -n "$T" ] && export XPU_TREE=$T || unset XPU_TREE
    exec timeout -k 120 -s INT 30m "$B" serve "$MODEL" --drafter "$DRAFTER" --context 132096 --parallel 1 \
      --host 127.0.0.1 --port $PORT --name m --no-update-check --prompt-cache-gib 0 ) > "$OUT/raw/$L.server.log" 2>&1 &
  SP=$!
  for i in $(seq 300); do
    kill -0 $SP 2>/dev/null || break
    curl -sf localhost:$PORT/health >/dev/null && break
    sleep 2
  done
  RC=1
  if curl -sf localhost:$PORT/health >/dev/null; then # the one request, nothing before it
    timeout 900 python3 -I "$HERE/cold_request.py" http://127.0.0.1:$PORT "$OUT/prompts/$P.txt" "$OUT/raw/$L.jsonl" \
      arm=$A prompt=$P server=$R order=$n tree=${T:-default} > "$OUT/raw/$L.request.log" 2>&1; RC=$?
  else log "cell $L: server not ready"; fi
  kill -INT $SP 2>/dev/null
  for i in $(seq 90); do kill -0 $SP 2>/dev/null || break; sleep 1; done
  kill -9 $SP 2>/dev/null
  wait $SP 2>/dev/null
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null
  log "cell $n/$NCELLS $L done rc=$RC: $(tail -1 "$OUT/raw/$L.request.log" 2>/dev/null)"
done < <(cells)
python3 -I "$HERE/summarize.py" "$OUT" > "$OUT/summary.md" 2>> "$LOG"; log "all cells done, summary.md written"
touch "$OUT/DONE"
