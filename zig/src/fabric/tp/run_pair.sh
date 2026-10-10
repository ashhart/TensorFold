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
for backend in ${SUITES:-mailbox nccl}; do
    pair "suite-$backend" "$backend" suite
done
: > "$OUT/failfast.txt"
for backend in ${FAILFAST:-nccl mailbox}; do
    for mode in fatal kill; do
        for victim in 0 1; do
            name=ff-$backend-$mode-$victim
            pair "$name" "$backend" failfast "$victim" "$mode"
            python3 - "$OUT" "$name" "$victim" >> "$OUT/failfast.txt" <<'EOF'
import re, sys
out, name, victim = sys.argv[1], sys.argv[2], int(sys.argv[3])
logs = [open(f"{out}/{name}.r{r}.log").read() for r in (0, 1)]
inj = re.search(r"INJECT \w+ rank \d+ at ([\d.]+)", logs[victim])
ex = [re.search(r"EXIT rank \d rc (\d+) at ([\d.]+)", l) for l in logs]
if not inj or not all(ex):
    print(f"{name}: FAIL (no inject / exit line)"); sys.exit()
t0 = float(inj.group(1))
rcs = [int(e.group(1)) for e in ex]
dt = [float(e.group(2)) - t0 for e in ex]
want = [137 if (r == victim and "kill" in name) else 70 for r in (0, 1)]
ok = rcs == want and dt[1 - victim] < 3.0
print(f"{name}: {'PASS' if ok else 'FAIL'} rc {rcs} (want {want}), exit after injection: rank0 {dt[0]:.3f} s, rank1 {dt[1]:.3f} s")
EOF
        done
    done
done
grep -h "^SUMMARY\|^FAIL" "$OUT"/suite*.log
cat "$OUT/failfast.txt"
