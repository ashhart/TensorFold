#!/bin/bash
# the branch's checks in order: kernel contract and its limits, the card tests one at a time on card 0, then zig build test
export ZE_AFFINITY_MASK=0 LC_ALL=C
R=$WORK; T=$R/tests; mkdir -p $T
cd $TREE
Z=$ZIG
F="-Doptimize=ReleaseFast -Dxpu-dpas=true -Dngen-include=$NGEN"
step() { echo "$(date +%T) tests: $*" >> $R/journal; }
step "contract build"
$Z build tf-xpu-contract-run -Dxpu-contract=true $F > $T/contract-build.out 2>&1 || { step "contract build FAILED"; exit 1; }
step "contract run"
./zig-out/bin/tf-xpu-contract-run --device 0 2>&1 | tr -d '\000' > $T/contract-run.out
step "contract rc=${PIPESTATUS[0]}"
$Z build xpu-kernel-contract-limits -Dxpu-contract=true $F > $T/limits.out 2>&1; step "limits rc=$?"
step "device tests build"
$Z build xpu-qwen27-tests $F > $T/qwen27-tests-build.out 2>&1 || { step "device tests build FAILED"; exit 1; }
MP=$MODELS/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
for X in means seam close; do
  step "$X start"
  timeout 1800 ./zig-out/bin/tf-xpu-qwen27-$X-test $MP > $T/branch-$X.out 2>&1; rc=$?
  step "$X rc=$rc"
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
done
step "zig build test start"
$Z build test $F > $T/zig-build-test.out 2>&1; rc=$?
step "zig build test rc=$rc"
echo $rc > $T/zig-build-test.rc
step "ALL DONE"
