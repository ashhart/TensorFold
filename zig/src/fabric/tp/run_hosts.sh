#!/usr/bin/env bash
# One rank of the two-host TP=2 gate over RoCE: `run_hosts.sh RANK RANK0_ADDR ./tf-tp-test OUTDIR` on both hosts.
set -u
RANK=${1:?rank}; MASTER=${2:?master address}; BIN=${3:?tf-tp-test}; OUT=${4:?outdir}
mkdir -p "$OUT"
[ -n "${NCCL_SOCKET_IFNAME:-}" ] && export NCCL_SOCKET_IFNAME # else NCCL picks the interface
# NCCL uses the RoCE wire's devices unless NCCL_IB_HCA is set: the names without `:gid` (NCCL reads `:N` as a port)
[ -n "${TF_TP_ROCE_HCA:-}" ] && [ -z "${NCCL_IB_HCA:-}" ] && NCCL_IB_HCA=$(echo "$TF_TP_ROCE_HCA" | sed 's/:[0-9]*//g; s/ //g')
[ -n "${TF_TP_ROCE_HCA:-}" ] && export TF_TP_ROCE_HCA
"$BIN" kernel > /dev/null 2>&1 || { echo "error: $BIN has no mailbox kernel: rebuild with -Dtp-nvcc=PATH or -Dtp-fatbin=PATH" >&2; exit 2; }
ulimit -l unlimited 2> /dev/null || echo "warning: memlock is $(ulimit -l) KiB: ibv_reg_mr of the RoCE region may fail" >&2
# neither named: NCCL takes the ports the RoCE picker chose with the peer (`tf-tp-test hca`, both ranks), not its own
if [ -z "${TF_TP_ROCE_HCA:-}${NCCL_IB_HCA:-}" ]; then
    # hca gives up after TF_TP_PLAN_TIMEOUT (default 300 s) with its reason; the cap here only catches a hang past that
    CAP=$(awk 'BEGIN { t = ARGV[1] + 0; if (t <= 0) t = 300; printf "%d", int(t) + 31 }' "${TF_TP_PLAN_TIMEOUT:-300}")
    echo "hca: rank $RANK picking the ports with the peer (up to $CAP s)"
    PICK=$(TF_TP_RANK=$RANK TF_TP_MASTER=$MASTER TF_TP_PORT=29700 timeout "$CAP" "$BIN" hca 2> "$OUT/hca.log")
    rc=$?
    if [ "$rc" -eq 124 ]; then
        echo "error: $BIN hca still running after $CAP s: killed ($OUT/hca.log)" >&2
        exit 3
    fi
    if [ "$rc" -ne 0 ]; then
        echo "error: $BIN hca exited $rc: $(grep -h '^FAIL\|^error' "$OUT/hca.log" | tail -n 1 | grep . || tail -n 1 "$OUT/hca.log") ($OUT/hca.log)" >&2
        exit 3
    fi
    if ! echo "$PICK" | grep -Eqx 'NCCL_IB_HCA=[A-Za-z0-9_.-]+(,[A-Za-z0-9_.-]+)*'; then
        echo "error: $BIN hca printed '$PICK', not NCCL_IB_HCA=name,... ($OUT/hca.log)" >&2
        exit 3
    fi
    NCCL_IB_HCA=${PICK#NCCL_IB_HCA=}
    echo "hca: rank $RANK NCCL_IB_HCA=$NCCL_IB_HCA ($(grep -h '^INFO hca' "$OUT/hca.log"))"
fi
[ -n "${NCCL_IB_HCA:-}" ] && export NCCL_IB_HCA
{ hostname; nvidia-smi -L; ibv_devices 2>&1; ulimit -l; env | grep -E '^(TF_TP_ROCE_HCA|NCCL_IB_HCA|NCCL_SOCKET_IFNAME)='; } > "$OUT/host.txt" 2>&1
PORT=29700
step() { # NAME BACKEND ARGS...: this host's rank; log OUT/NAME.log ending with an EXIT line
    local name=$1 backend=$2; shift 2
    PORT=$((PORT + 10))
    ( TF_TP_RANK=$RANK TF_TP_MASTER=$MASTER TF_TP_PORT=$PORT TF_COMM_BACKEND=$backend TF_TP_ROCE_FALLBACK=error \
        timeout 900 "$BIN" "$@"; echo "EXIT rank $RANK rc $? at $(date +%s.%N)" ) > "$OUT/$name.log" 2>&1
    echo "$name: $(grep -h '^SUMMARY\|^PASS roce-host\|^INJECT\|^EXIT' "$OUT/$name.log" | tr '\n' ' ')"
}
step roce-host roce roce-host
step suite-roce roce suite
step suite-nccl nccl suite
for backend in roce nccl; do
    for mode in fatal kill; do
        for victim in 0 1; do step "ff-$backend-$mode-$victim" "$backend" failfast "$victim" "$mode"; done
    done
done
