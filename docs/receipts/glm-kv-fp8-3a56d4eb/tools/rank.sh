#!/bin/bash
# usage: rank.sh RANK TREE LABEL CTX [VAR=VAL...] -- [serve args...]; one rank of ~/fl-receipt/TREE, detached, under
# memory_receipts.py (out/LABEL/mem-rankN/, its process.log is the server log); its pid in ~/fl-receipt/rankN.pid.
r=$1 tree=$2 label=$3 ctx=$4; shift 4
. ~/fl-receipt/fl-env.sh
while [ $# -gt 0 ] && [ "$1" != -- ]; do export "$1"; shift; done; shift
out=~/fl-receipt/out/$label; mkdir -p $out
cd ~/fl-receipt/$tree || exit 1
TF_GLM_RANK=$r setsid nohup python3 ~/fl-receipt/tools/memory_receipts.py --rank $r --context-tokens $ctx --interval 1 \
  --output $out/mem-rank$r -- zig-out/native/bin/tensorfold-native serve "$GLM" --host 127.0.0.1 --port $PORT \
  --name local-model --no-update-check --context $ctx "$@" > $out/wrap-rank$r.log 2>&1 < /dev/null &
echo $! > ~/fl-receipt/rank$r.pid
