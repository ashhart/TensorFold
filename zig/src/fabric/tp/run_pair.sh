#!/usr/bin/env bash
# Two ranks of tf-tp-test on one two-GPU host: `[SUITES=..] [FAILFAST=..] run_pair.sh BIN OUTDIR`.
set -u
BIN=${1:?bin}; OUT=${2:?outdir}
mkdir -p "$OUT"
nvidia-smi > "$OUT/smi.txt" 2>&1
nvidia-smi topo -m > "$OUT/topo.txt" 2>&1
ldconfig -p | grep -E "libnccl|libcuda" > "$OUT/libs.txt" 2>&1
PORT=29500
pair() { # NAME BACKEND ARGS...: both ranks, logs OUT/NAME.rR.log ending with an EXIT line
    local name=$1 backend=$2 r; shift 2
    PORT=$((PORT + 10))
    for r in 0 1; do
        ( TF_TP_RANK=$r TF_TP_DEVICE=$r TF_TP_PORT=$PORT TF_COMM_BACKEND=$backend NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
            timeout 600 "$BIN" "$@"; echo "EXIT rank $r rc $? at $(date +%s.%N)" ) > "$OUT/$name.r$r.log" 2>&1 &
    done
    wait
}
# ffsum OUT NAME VICTIM: a fail-fast pair's verdict line (exit codes, exit times after the injection)
ffsum() {
    [ -f "$1/$2.r0.log" ] && [ -f "$1/$2.r1.log" ] || { echo "ffsum: $1/$2.r0.log or .r1.log missing" >&2; return 1; }
    awk -v name="$2" -v victim="$3" '
        FNR == 1 { r = (FILENAME ~ /\.r1\.log$/) ? 1 : 0 }
        !(r in ex) && match($0, /EXIT rank [0-9] rc [0-9]+ at [0-9.]+/) {
            split(substr($0, RSTART, RLENGTH), f, " "); rc[r] = f[5]; at[r] = f[7]; ex[r] = 1
        }
        r == victim && !inj && match($0, /INJECT [A-Za-z0-9_]+ rank [0-9]+ at [0-9.]+/) {
            split(substr($0, RSTART, RLENGTH), g, " "); t0 = g[6]; inj = 1
        }
        END {
            if (!inj || !(0 in ex) || !(1 in ex)) { print name ": FAIL (no inject / exit line)"; exit }
            for (i = 0; i < 2; i++) { want[i] = (i == victim && name ~ /kill/) ? 137 : 70; dt[i] = at[i] - t0 }
            ok = rc[0] == want[0] && rc[1] == want[1] && dt[1 - victim] < 3.0
            printf "%s: %s rc [%d, %d] (want [%d, %d]), exit after injection: rank0 %.3f s, rank1 %.3f s\n", name,
                ok ? "PASS" : "FAIL", rc[0], rc[1], want[0], want[1], dt[0], dt[1]
        }' "$1/$2.r0.log" "$1/$2.r1.log"
}
for backend in ${SUITES:-mailbox nccl}; do
    pair "suite-$backend" "$backend" suite
done
: > "$OUT/failfast.txt"
for backend in ${FAILFAST:-nccl mailbox}; do
    for mode in fatal kill; do
        for victim in 0 1; do
            name=ff-$backend-$mode-$victim
            pair "$name" "$backend" failfast "$victim" "$mode"
            ffsum "$OUT" "$name" "$victim" >> "$OUT/failfast.txt"
        done
    done
done
grep -h "^SUMMARY\|^FAIL" "$OUT"/suite*.log
cat "$OUT/failfast.txt"
