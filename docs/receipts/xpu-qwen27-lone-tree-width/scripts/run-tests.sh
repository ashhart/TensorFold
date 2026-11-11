#!/bin/bash
# the three device tests, one at a time on card 0. usage: BIN_DIR=<zig-out/bin> MODELS=<dir> run-tests.sh OUTDIR
export ZE_AFFINITY_MASK=0 LC_ALL=C
MP=$MODELS/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
mkdir -p "$1"
for T in means seam close; do
  timeout 1800 "$BIN_DIR/tf-xpu-qwen27-$T-test" "$MP" > "$1/$T.out" 2>&1 || { echo "$T failed"; exit 1; }
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
done
