#!/usr/bin/env bash
# The failed-transport check on two hosts (tf-tp-test failfast V transport); `run_transport.sh` alone prints how.
set -u
usage() {
    cat << 'TXT'
run_transport.sh MASTER_ADDR PEER_SSH OUTDIR    (on rank 0's host, as root; rank 1 runs over ssh)
  Two hosts with this script and tf-tp-test (built with the mailbox kernel) at the same path on both. Before the steps,
  on both hosts: no GPU process and memlock unlimited. Steps: T0 (suite), T1 (victim 1), T3 (victim 0), fail-fast on;
  tf-tp-test injects only when both links are idle. After each step, on both hosts: the kernel log since the step
  began and every HCA *err* counter against the step's start. A matching line, a counter that grew, an unexpected rc
  or a failure line prints STOP: and ends the run (exit 1), with no later step.
run_transport.sh t2 OUTDIR    (one host: TF_TP_ALLOW_HOST_ONLY=1)
  T2 (fail-fast off, a hold) runs both ranks on this host over the shm wire; tf-tp-test refuses it on an RDMA wire.
TXT
}
[ $# -ge 2 ] || { usage; exit 2; }
KIT=$(cd "$(dirname "$0")" && pwd)
BIN=$KIT/tf-tp-test
KLOG='EXT4-fs|smmu|iommu|mlx5|AER|nvme|Xid'
BAD='RDMA write of seq|local protection|NOT drained|GPU sync FAILED|REFUSED|refused ERR|panic'
"$BIN" kernel > /dev/null 2>&1 || { echo "STOP: $BIN has no mailbox kernel: rebuild with -Dtp-nvcc=PATH or -Dtp-fatbin=PATH" >&2; exit 2; }

if [ "${1:-}" = t2 ]; then
    OUT=${2:?outdir}
    [ "${TF_TP_ALLOW_HOST_ONLY:-}" = 1 ] || { echo "STOP: T2 is host-only: set TF_TP_ALLOW_HOST_ONLY=1" >&2; exit 2; }
    mkdir -p "$OUT"
    for r in 0 1; do
        TF_TP_RANK=$r TF_TP_MASTER=127.0.0.1 TF_TP_PORT=29990 TF_COMM_BACKEND=roce TF_TP_ROCE_WIRE=shm TF_TP_FAILFAST=0 \
            timeout 300 "$BIN" failfast 1 transport 200 5 > "$OUT/T2-r$r.log" 2>&1 &
    done
    wait
    grep -h '^PASS\|^FAIL\|REFUSED\|regions kept' "$OUT"/T2-r*.log
    exit 0
fi

MASTER=${1:?master address}; PEER=${2:?peer ssh target}; OUT=${3:?outdir}
mkdir -p "$OUT"
stop() { echo "STOP: $*" | tee -a "$OUT/verdict.txt"; exit 1; }

precheck() {
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" ] || { echo "a GPU process runs"; return 1; }
    [ "$(bash -c 'ulimit -l unlimited 2> /dev/null; ulimit -l')" = unlimited ] || { echo "memlock not unlimited"; return 1; }
    echo "ok $(hostname) $(uname -r)"
}
counters() {
    find -L /sys/class/infiniband/ -maxdepth 5 -path '*ports*counters*' -name '*err*' 2> /dev/null | sort | while read -r f; do
        echo "$f $(cat "$f" 2> /dev/null)"
    done
}
klog() { # kernel lines since epoch $1 matching $2 (journalctl must answer: an unreadable log is a STOP)
    journalctl -k --since "@$1" --no-pager -o short-iso 2> /dev/null | grep -E "$2"
    return "${PIPESTATUS[0]}"
}
probe() { # this host's tf-tp-test QPs and MRs once a second until it exits
    local pid="" t
    for _ in $(seq 200); do pid=$(pgrep -nx tf-tp-test) && break; sleep 0.1; done
    [ -z "$pid" ] && return
    for t in $(seq 180); do
        kill -0 "$pid" 2> /dev/null || break
        echo "== t=$t pid $pid"
        rdma resource show qp pid "$pid" 2>&1
        rdma resource show mr pid "$pid" 2>&1
        sleep 1
    done
}
on() { # NODE CMD ARGS...: a command or one of the functions above, here (local) or on the peer
    local node=$1 fn=$2 def
    shift 2
    if [ "$node" = local ]; then "$fn" "$@"; return; fi
    def=$(declare -f "$fn" 2> /dev/null)
    ssh -o BatchMode=yes "$PEER" "${def:+$def
}$fn $(printf '%q ' "$@")"
}
grew() { # BEFORE AFTER: counters that went up
    awk 'NR == FNR { b[$1] = $2; next } ($1 in b) && $2 > b[$1] { print $1 " " b[$1] " -> " $2 }' "$1" "$2"
}

for n in local peer; do
    v=$(on $n precheck) || stop "$n: $v"
    echo "precheck $n: $v" | tee -a "$OUT/verdict.txt"
done
on peer test -x "$BIN" || stop "peer: no $BIN (the kit must sit at the same path on both hosts)"

PORT=29900
step() { # NAME WANT_RC -- ARGS...
    local name=$1 want=$2 n rc rc0 rc1 hits g
    shift 3
    PORT=$((PORT + 10))
    declare -A t
    for n in local peer; do
        t[$n]=$(on $n date +%s)
        on $n counters > "$OUT/$name.$n.err0"
    done
    on local probe > "$OUT/$name.local.rdma.txt" 2>&1 &
    on peer probe > "$OUT/$name.peer.rdma.txt" 2>&1 &
    local envs=(TF_TP_MASTER="$MASTER" TF_TP_PORT=$PORT TF_COMM_BACKEND=roce TF_TP_ROCE_FALLBACK=error TF_TP_MAILBOX_MAX_KB=1024
        TF_TP_ROCE_FAST=1 TF_TP_FAILFAST_GRACE_S=15)
    ssh -o BatchMode=yes "$PEER" "ulimit -l unlimited && env $(printf '%q ' "${envs[@]}") TF_TP_RANK=1 timeout 300 $(printf '%q ' "$BIN" "$@"); echo \"EXIT rank 1 rc \$? at \$(date +%s.%N)\"" > "$OUT/$name-r1.log" 2>&1 &
    ( ulimit -l unlimited && env "${envs[@]}" TF_TP_RANK=0 timeout 300 "$BIN" "$@"; echo "EXIT rank 0 rc $? at $(date +%s.%N)" ) > "$OUT/$name-r0.log" 2>&1
    wait
    rc0=$(grep -oP '^EXIT rank 0 rc \K[0-9]+' "$OUT/$name-r0.log")
    rc1=$(grep -oP '^EXIT rank 1 rc \K[0-9]+' "$OUT/$name-r1.log")
    echo "$name: rank 0 rc ${rc0:-?}, rank 1 rc ${rc1:-?} (want $want)" | tee -a "$OUT/verdict.txt"
    for n in local peer; do
        on $n counters > "$OUT/$name.$n.err1"
        g=$(grew "$OUT/$name.$n.err0" "$OUT/$name.$n.err1")
        [ -n "$g" ] && stop "$name $n: HCA error counters grew: $g"
        hits=$(on $n klog "${t[$n]}" "$KLOG"); rc=$?
        [ "$rc" -ne 0 ] && stop "$name $n: the kernel log is unreadable (journalctl rc $rc)"
        [ -n "$hits" ] && { echo "$hits" > "$OUT/$name.$n.klog.txt"; stop "$name $n: kernel lines: $hits"; }
    done
    hits=$(grep -hE "$BAD" "$OUT/$name-r0.log" "$OUT/$name-r1.log")
    [ -n "$hits" ] && stop "$name: $hits"
    [ "$rc0" = "$want" ] && [ "$rc1" = "$want" ] || stop "$name: rc $rc0 / $rc1, want $want"
    grep -h 'halted\|quiesced\|idle before\|regions kept' "$OUT/$name"-r*.log | sed "s/^/  /"
}
step T0 0 -- suite
step T1 70 -- failfast 1 transport 200
step T3 70 -- failfast 0 transport 200
echo "PASS transport check (T0, T1, T3)" | tee -a "$OUT/verdict.txt"
