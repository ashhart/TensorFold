#!/bin/bash
# One TP=2 rank of the EXL3 Flash Next server: MODEL_DIR RANK PORT MASTER_PORT [serve args]; start rank 1 first.
# Environment: LINES (default 38), RESERVE_MB, MEMORY_RESERVE_GIB, CONTEXT, PARALLEL (default 2), TENSORFOLD_BIN,
# TF_FLASHNEXT_EXL3_GRAPH (1: graphs, 0: the host cut).
set -euo pipefail
model=$1 rank=$2 port=$3 master=$4
shift 4
bin=${TENSORFOLD_BIN:-./zig-out/native/bin/tensorfold-native}
if [ -z "${TENSORFOLD_CHECKPOINT_SHA256:-}" ]; then
  TENSORFOLD_CHECKPOINT_SHA256=$(cat "$model/model.safetensors.index.json" "$model/config.json" | sha256sum | cut -c1-64)
fi
export TENSORFOLD_CHECKPOINT_SHA256
# TF_FLASHNEXT_EXL3_GRAPH=1 replays the decode, verify, draft and prefill graphs; GRAPH_ASYNC=1 builds them on a worker.
export TF_FLASHNEXT_EXL3_GRAPH=${TF_FLASHNEXT_EXL3_GRAPH:-1}
export TF_FLASHNEXT_EXL3_GRAPH_ASYNC=${TF_FLASHNEXT_EXL3_GRAPH_ASYNC:-0}
export TF_FLASHNEXT_EXPERT_LINES=${LINES:-38}
[ -n "${RESERVE_MB:-}" ] && export TF_FLASHNEXT_EXPERT_RESERVE_MB=$RESERVE_MB
[ -n "${MEMORY_RESERVE_GIB:-}" ] && export TENSORFOLD_MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB
CUDA_VISIBLE_DEVICES=$rank exec "$bin" serve "$model" --backend cuda --context "${CONTEXT:-32768}" --tp 2 --rank "$rank" \
  --master 127.0.0.1 --master-port "$master" --port "$port" --no-update-check --parallel "${PARALLEL:-2}" --prompt-cache-gib 0 "$@"
