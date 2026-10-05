#!/bin/bash
# Watch one dev run on this node and kill it before it takes the node down: GB10's GPU allocations are the host's RAM,
# and running out has rebooted aiai (the host watchdog). Started by tools/dsv41_run2.sh inside tf-dev on each node
# (docker exec -d: same PID namespace and root as the run, so the kill is immediate), it lives only as long as the run:
#
#   dsv41_memwatch.sh TAG [MIN_GIB] [LOG]
#   dsv41_memwatch.sh --stop TAG [GRACE_S] [LOG]     # end of the run: its leftovers killed after GRACE_S, watcher off
#
# TAG: the run's --run-tag (its processes: "dsv41_serial_run.py ... --run-tag TAG"). While any of them lives,
# MemAvailable is read every TF_MEMWATCH_PERIOD seconds (0.25); below MIN_GIB (3) the run's processes get SIGKILL and
# a KILL line says why (LOG, default /tf/memwatch.log, and stderr). The watcher exits when the run's processes are gone,
# when none appeared within TF_MEMWATCH_WAIT seconds (120), after TF_MEMWATCH_MAX_S seconds (4000, past run2.sh's
# timeout) or on SIGTERM (run2.sh stops it at the end of the run). It kills nothing else.
set -u
if [ "${1:-}" = --stop ]; then
  # a rank still there after GRACE_S (the other failed or was killed) would wait on a collective forever
  TAG=${2:?usage: dsv41_memwatch.sh --stop TAG [GRACE_S] [LOG]}
  GRACE=${3:-0}
  LOG=${4:-/tf/memwatch.log}
  PAT="dsv41_serial_run\.py .*--run-tag ${TAG}( |\$)"
  for _ in $(seq 1 "$GRACE"); do pgrep -f -- "$PAT" >/dev/null 2>&1 || break; sleep 1; done
  if pgrep -f -- "$PAT" >/dev/null 2>&1; then
    pkill -KILL -f -- "$PAT"
    echo "$(date '+%F %T') memwatch $TAG: stop: killed the run's processes still there after ${GRACE}s" | tee -a "$LOG"
  fi
  pkill -TERM -f -- "dsv41_memwatch\.sh ${TAG}( |\$)"
  # bash runs the TERM trap after the watcher's current sleep: gone when this returns (at most ~2 s after a KILL)
  for _ in $(seq 1 40); do pgrep -f -- "dsv41_memwatch\.sh ${TAG}( |\$)" >/dev/null 2>&1 || break; sleep 0.1; done
  grep -h -- "memwatch $TAG: KILL" "$LOG" 2>/dev/null
  exit 0
fi
TAG=${1:?usage: dsv41_memwatch.sh TAG [MIN_GIB] [LOG]}
MIN_GIB=${2:-3}
LOG=${3:-/tf/memwatch.log}
MEMINFO=${TF_MEMWATCH_MEMINFO:-/proc/meminfo}
PERIOD=${TF_MEMWATCH_PERIOD:-0.25}
WAIT=${TF_MEMWATCH_WAIT:-120}
MAX_S=${TF_MEMWATCH_MAX_S:-4000}
# the run's processes; this script's own command line never holds the pattern (it is built here, not passed)
PAT="dsv41_serial_run\.py .*--run-tag ${TAG}( |\$)"
MIN_KB=$(awk -v g="$MIN_GIB" 'BEGIN { printf "%d", g * 1048576 }')

say() { echo "$(date '+%F %T') memwatch $TAG: $*" | tee -a "$LOG" >&2; }
avail_kb() { awk '/^MemAvailable:/ { print $2; exit }' "$MEMINFO"; }
alive() { pgrep -f -- "$PAT" >/dev/null 2>&1; }
trap 'say "stopped"; exit 0' TERM INT

start=$SECONDS
while ! alive; do
  if (( SECONDS - start >= WAIT )); then say "no process of the run within ${WAIT}s, exiting"; exit 0; fi
  sleep 1
done
say "watching (kill below ${MIN_GIB} GiB MemAvailable, now $(( $(avail_kb) / 1048576 )) GiB)"
low=$(avail_kb)
while alive; do
  a=$(avail_kb)
  (( a < low )) && low=$a
  if (( a < MIN_KB )); then
    pids=$(pgrep -f -- "$PAT" | tr '\n' ' ')
    pkill -KILL -f -- "$PAT"
    say "KILL: MemAvailable $(awk -v k="$a" 'BEGIN { printf "%.2f", k / 1048576 }') GiB < ${MIN_GIB} GiB: killed the run (pids ${pids% })"
    sleep 2
    say "after the kill: MemAvailable $(( $(avail_kb) / 1048576 )) GiB"
    exit 3
  fi
  if (( SECONDS - start >= MAX_S )); then say "past ${MAX_S}s, exiting (the run goes on unwatched)"; exit 0; fi
  sleep "$PERIOD"
done
say "the run ended; lowest MemAvailable $(awk -v k="$low" 'BEGIN { printf "%.1f", k / 1048576 }') GiB"
exit 0
