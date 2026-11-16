#!/bin/bash
# usage: rank.sh RANK TREE LABEL CTX [VAR=VAL...] -- [serve args...]; NSYS=1 in the VARs starts it under an idle nsys session glmN.
r=$1 tree=$2 label=$3 ctx=$4; shift 4
. ~/glm-perf/env.sh
NSYS=0
while [ $# -gt 0 ] && [ "$1" != -- ]; do export "$1"; shift; done; shift
out=~/glm-perf/out/$label; mkdir -p $out
cd ~/glm-perf/$tree || exit 1
pre=""
[ "$NSYS" = 1 ] && pre="nsys launch --session-new=glm$r --trace=cuda,nvtx,osrt --cuda-graph-trace=node"
TF_GLM_RANK=$r setsid nohup $pre zig-out/native/bin/tensorfold-native serve "$GLM" --host 127.0.0.1 --port $PORT \
  --name local-model --no-update-check --context $ctx "$@" > $out/server-rank$r.log 2>&1 < /dev/null &
echo $! > ~/glm-perf/rank$r.pid
