#!/bin/bash
# a tier of the serial engine's checks on both nodes, one process (one weight load) a group (tools/dsv41_suite.py):
#
#   tools/dsv41_suite2.sh quick|full|long|NAME[,NAME..] [extra dsv41_run2.sh args, e.g. --suite-baseline /tf/out/X.json]
#   TF_SUITE_PROVE=1 tools/dsv41_suite2.sh quick    # then every test again in a fresh process: the same fingerprints
#
# Each group runs through tools/dsv41_run2.sh (prepared weights, the OOM guard) with the environment it needs
# (TF_DSV41_KV=fp8 for fp8); rank 0's output lands in out/suite-<stamp>/<group>.log of this checkout, the results
# JSON beside it (copied from aiai's ~/tensorfold/out). The summary is the last line; exit 1 on a FAIL or an ERROR,
# a test of the tier without a result (its group's process died: memwatch, OOM, NCCL) or a run that exited non-zero.
set -e
cd /home/urtho/dev/_experiments/ai/tensorfold
SPEC=${1:?usage: dsv41_suite2.sh quick|full|long|NAME[,NAME..] [dsv41_run2.sh args]}
shift
S="python3 tools/dsv41_suite.py"
GROUPS_=$($S groups "$SPEC")
STAMP=$(date +%F-%H%M%S)
OUT=out/suite-$STAMP
mkdir -p "$OUT"
SHOW='^\[(result|suite|baseline|rank 1\] \[suite)|memory guard|weights loaded|prepared folder|memwatch|Error|error'
rc=0
# both ranks read --suite-baseline: a baseline under /tf/out exists on aiai only (the suite JSONs land there), so copy
# it to aiai2 first (rank 1 died on the missing file and rank 0 then waited in a collective)
prev=""
for a in "$@"; do
  if [ "$prev" = --suite-baseline ]; then
    case "$a" in /tf/out/*) f=${a#/tf/out/}
      ssh aiai "cat tensorfold/out/$f" | ssh aiai2 "docker exec -i tf-dev sh -c 'mkdir -p /tf/out && cat > /tf/out/$f'" \
        || { echo "cannot copy the baseline $a to aiai2"; exit 1; };;
    esac
  fi
  prev=$a
done
for g in $GROUPS_; do
  envs=$($S env "$g" | tr '\n' ' ')
  echo "=== group $g: $($S flags "$g")${envs:+ ($envs)}"
  # shellcheck disable=SC2086
  env $envs TF_TAIL_LINES=1000000 tools/dsv41_run2.sh --suite "$SPEC" --group "$g" \
    --suite-out "/tf/out/suite-$STAMP-$g.json" "$@" > "$OUT/$g.log" 2>&1 || { echo "group $g: run exited non-zero"; rc=1; }
  grep -E "$SHOW" "$OUT/$g.log" || tail -20 "$OUT/$g.log"
  scp -q "aiai:tensorfold/out/suite-$STAMP-$g.json" "$OUT/" 2>/dev/null || true
done
$S merge --spec "$SPEC" "$OUT"/*.log > "$OUT/summary.txt" || rc=1
if [ "${TF_SUITE_PROVE:-0}" = 1 ]; then
  echo "=== prove: each test in a fresh process"
  for name in $($S fresh "$SPEC" | cut -f1); do
    envs=$($S testenv "$name" | tr '\n' ' ')
    # shellcheck disable=SC2086,SC2046
    env $envs TF_TAIL_LINES=1000000 tools/dsv41_run2.sh $($S argv "$name") "$@" > "$OUT/fresh-$name.txt" 2>&1 || rc=1
    grep -E '^\[result\]' "$OUT/fresh-$name.txt" || { echo "$name: no [result] line (see $OUT/fresh-$name.txt)"; rc=1; }
  done
  cat "$OUT"/*.log > "$OUT/suite-all.txt"
  $S prove "$OUT/suite-all.txt" "$OUT"/fresh-*.txt | tee "$OUT/prove.txt" || rc=1
fi
cat "$OUT/summary.txt"
echo "logs: $OUT"
exit $rc
