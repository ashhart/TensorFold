#!/usr/bin/env bash
# One watchdog tick for the tensorfold-dsv41 compose pair (head here, worker over ssh). Run by
# systemd/tensorfold-dsv41-watchdog.timer every minute (`make watch-install`), or by hand: `make watch-once`.
#
# Adapted from jayleaton/deepseek-v41-tensorfold-spark scripts/serve.sh (watch_tick, lease_fresh, alert and the
# DSV41_TF_LOCKED heal line), Apache License 2.0, Copyright 2026 Jay Leaton (https://x.com/jayleaton).
# Modified for tensorfold dsv41-cuda: the compose pair healed through `make restart`; a `make down` holds across
# reboots; vLLM holding the pair stands down; /health's fatal / stalled fields and an idle probe count as bad; a dead
# container counts twice; the lease file's mtime is its expiry; logs of both ranks are saved before a heal.
#
# Stands down (exit 0, the bad-tick count reset) while:
#   the lease is unexpired ($STATE/lease, `make lease`: its mtime is the expiry), an up / down / restart holds the
#   lock, both containers are absent after a deliberate `make down` ($STATE/stopped), a vLLM container runs on either
#   node, or the head container is younger than WATCH_GRACE and /health does not answer yet (loading).
# A tick is bad when a container is not running, /health is not 200 or reports fatal / stalled, or the idle probe
# (a 1-token chat request every WATCH_PROBE_S while nothing runs) fails. Both containers absent with no marker (a
# reboot while serving) is bad too: the watchdog is then the boot start. A dead container counts 2; WATCH_FAILS bad
# ticks heal: WATCH_HEAL=1 restarts both ranks in the background (at most once every WATCH_MIN_HEAL s), WATCH_HEAL=0
# (the default) only alerts. WATCH_ALERT is an optional command given the message. Exit 1: unhealthy (the unit
# treats it as success).
set -uo pipefail

DIR="${WATCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
env_of() { grep -E "^$1=" "$DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2-; }
setting() { local v="${!1:-}"; [[ -n "$v" ]] || v=$(env_of "$1"); echo "${v:-$2}"; }   # the environment, .env, a default

STATE="${XDG_STATE_HOME:-$HOME/.local/state}/tensorfold-dsv41"
WORKER=$(setting WORKER aiai2-ib)
NAME=$(setting WATCH_CONTAINER tensorfold-dsv41-serve-1)
VLLM_NAMES=$(setting VLLM_NAMES 'dsv41-exl3-head|dsv41-exl3-worker|deepseek4flash-tp2-vllm-dspark-1')
PORT=$(setting PORT 8888)
SERVED_NAME=$(setting SERVED_NAME DeepSeek-v4.1-Flash-EXL3)
BASE="http://127.0.0.1:$PORT"
WATCH_GRACE=$(setting WATCH_GRACE 1800)          # matches the Makefile's WAIT_TIMEOUT
WATCH_FAILS=$(setting WATCH_FAILS 3)
WATCH_HEAL=$(setting WATCH_HEAL 0)
WATCH_MIN_HEAL=$(setting WATCH_MIN_HEAL 1800)
WATCH_PROBE_S=$(setting WATCH_PROBE_S 600)
WATCH_ALERT=$(setting WATCH_ALERT "")

mkdir -p "$STATE"
log() { echo "[$(date '+%F %T')] watch: $*" | tee -a "$STATE/watch.log"; }
alert() { log "ALERT: $*"; [[ -z "$WATCH_ALERT" ]] || $WATCH_ALERT "$*" || true; }
reset() { echo 0 > "$STATE/fails"; }
wssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "$@"; }

# a container's state: running / exited / ... / absent (worker: unreachable when ssh fails)
state_head() { docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo absent; }
state_worker() {
    local out rc
    out=$(wssh "docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null || echo absent" 2>/dev/null); rc=$?
    [[ $rc == 0 && -n "$out" ]] && echo "$out" || echo unreachable
}
vllm_running() {
    docker ps --format '{{.Names}}' 2>/dev/null | grep -qxE "$VLLM_NAMES" && return 0
    wssh "docker ps --format '{{.Names}}'" 2>/dev/null | grep -qxE "$VLLM_NAMES"
}

# a 1-token chat completion; the key goes to curl through a header file descriptor, never on its command line
probe() {
    local key; key=$(env_of TF_API_KEY)
    curl -sf -m 90 -o /dev/null "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -H @<(printf 'Authorization: Bearer %s\n' "$key") \
        -d "{\"model\": \"$SERVED_NAME\", \"max_tokens\": 1, \"temperature\": 0, \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}"
}

tick() {
    local now fails head worker bad="" weight=1 code body age last
    now=$(date +%s); fails=$(cat "$STATE/fails" 2>/dev/null || echo 0)
    if [[ -f "$STATE/lease" ]] && (( $(stat -c %Y "$STATE/lease") > now )); then
        reset; log "lease until $(date -d "@$(stat -c %Y "$STATE/lease")" '+%F %T'): standing down"; return 0; fi
    if ! flock -n "$STATE/lock" true; then reset; log "an up / down / restart is running: standing down"; return 0; fi
    if vllm_running; then reset; log "vLLM holds the pair: standing down"; return 0; fi
    head=$(state_head); worker=$(state_worker)
    if [[ "$head" == absent && ( "$worker" == absent || "$worker" == unreachable ) ]]; then
        if [[ -f "$STATE/stopped" ]]; then reset; log "stopped on purpose ($(cat "$STATE/stopped")): standing down"; return 0; fi
        [[ "$worker" == unreachable ]] && { log "head absent, worker unreachable over ssh (not counted)"; return 0; }
        bad="both containers absent and no stop marker (a reboot while serving?)"
    elif [[ "$head" != running ]]; then bad="head container $head"; weight=2
    elif [[ "$worker" == unreachable ]]; then log "worker unreachable over ssh (not counted)"
    elif [[ "$worker" != running ]]; then bad="worker container $worker"; weight=2
    fi
    if [[ -z "$bad" && "$head" == running ]]; then
        age=$(( now - $(date -d "$(docker inspect -f '{{.State.StartedAt}}' "$NAME")" +%s) ))
        body=$(curl -s -m 10 -w '\n%{http_code}' "$BASE/health" || true)
        code=${body##*$'\n'}; body=${body%$'\n'*}
        if [[ "$code" == 000 && $age -lt $WATCH_GRACE ]]; then reset; log "loading (${age}s)"; return 0
        elif [[ "$code" != 200 ]]; then bad="/health $code ${body:0:300}"
        elif [[ "$(jq -r '.fatal // empty' <<< "$body")" != "" ]]; then bad="fatal: $(jq -r .fatal <<< "$body")"
        elif [[ "$(jq -r '.stalled // false' <<< "$body")" == true ]]; then
            bad="stalled: one engine call for $(jq -r .call_age_s <<< "$body")s"
        elif [[ "$(jq -r '.requests_running // 0' <<< "$body")" == 0 ]] && \
             (( now - $(cat "$STATE/last_probe" 2>/dev/null || echo 0) >= WATCH_PROBE_S )); then
            echo "$now" > "$STATE/last_probe"     # every WATCH_PROBE_S: a hung probe shows up as stalled after it
            probe || bad="idle probe failed (a 1-token request)"
        fi
    fi
    if [[ -z "$bad" ]]; then reset; return 0; fi
    fails=$((fails + weight)); echo "$fails" > "$STATE/fails"
    log "bad tick ($fails of $WATCH_FAILS): $bad"
    (( fails >= WATCH_FAILS )) || return 1
    last=$(cat "$STATE/last_heal" 2>/dev/null || echo 0)
    if [[ "$WATCH_HEAL" != 1 ]]; then alert "unhealthy ($bad); WATCH_HEAL=0, not restarting"; return 1; fi
    if (( now - last < WATCH_MIN_HEAL )); then alert "unhealthy ($bad); healed $((now - last))s ago, waiting"; return 1; fi
    echo "$now" > "$STATE/last_heal"; reset
    docker logs --tail 200 "$NAME" > "$STATE/heal-$now-head.log" 2>&1 || true
    wssh "docker logs --tail 200 $NAME" > "$STATE/heal-$now-worker.log" 2>&1 || true
    alert "unhealthy ($bad); restarting both ranks (logs: $STATE/heal-$now-*.log)"
    # TF_DSV41_LOCKED: the Makefile's up / down / restart take this lock unless it is set, so they don't block on
    # the lock held here (flock on a new open file description would wait for this one forever)
    setsid env TF_DSV41_LOCKED=1 flock "$STATE/lock" make -C "$DIR" --no-print-directory restart \
        >> "$STATE/heal.log" 2>&1 < /dev/null &
    return 1
}

tick
