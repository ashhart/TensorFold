#!/bin/bash
# Launch one TP=2 rank of the EXL3 Qwen3.8-Flash-Next server on two RTX 5060 Ti (16 GB), drafts on at every stream count.
# Usage: tools/exl3_serve_tp2.sh MODEL_DIR RANK PORT MASTER_PORT [extra serve args]; start rank 1 first, then rank 0.
# The MTP head is loaded by default (rank 0 holds it); `--no-drafts` in the extra arguments serves without it.
# Environment: LINES (expert lines a layer, default 38), RESERVE_MB (planner reserve), MEMORY_RESERVE_GIB, CONTEXT,
# PARALLEL (streams, default 2), TENSORFOLD_BIN.
set -euo pipefail
model=$1 rank=$2 port=$3 master=$4
shift 4
bin=${TENSORFOLD_BIN:-./zig-out/native/bin/tensorfold-native}
export TENSORFOLD_CHECKPOINT_SHA256=${TENSORFOLD_CHECKPOINT_SHA256:-$(cat "$model/model.safetensors.index.json" "$model/config.json" | sha256sum | cut -c1-64)}
export TF_FLASHNEXT_EXPERT_LINES=${LINES:-38}
[ -n "${RESERVE_MB:-}" ] && export TF_FLASHNEXT_EXPERT_RESERVE_MB=$RESERVE_MB
[ -n "${MEMORY_RESERVE_GIB:-}" ] && export TENSORFOLD_MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB
CUDA_VISIBLE_DEVICES=$rank exec "$bin" serve "$model" --backend cuda --context "${CONTEXT:-32768}" --tp 2 --rank "$rank" \
  --master 127.0.0.1 --master-port "$master" --port "$port" --no-update-check --parallel "${PARALLEL:-2}" --prompt-cache-gib 0 "$@"
