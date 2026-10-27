#!/bin/bash
# Launch one TP=2 rank of the EXL3 Qwen3.8-Flash-Next server with the measured-fastest settings on two RTX 5060 Ti (16 GB).
# Usage: tools/exl3_serve_tp2.sh MODEL_DIR RANK PORT MASTER_PORT [extra serve args]; start rank 1 first, then rank 0.
# FAST=0 uses the plain defaults (38 lines, drafts loaded). FAST=1 (default) drops the MTP head, trims the reserve to 512 MB and sizes the
# shared BF16 KV window for PARALLEL lanes instead of the default 8, which together let 68 lines per layer fit on a 16 GB card.
# Plain decode is faster than drafting on offloaded experts here; see docs/plans/exl3-on-upstream-final-report.md.
set -euo pipefail
model=$1 rank=$2 port=$3 master=$4
shift 4
bin=${TENSORFOLD_BIN:-./zig-out/native/bin/tensorfold-native}
export TENSORFOLD_CHECKPOINT_SHA256=${TENSORFOLD_CHECKPOINT_SHA256:-$(cat "$model/model.safetensors.index.json" "$model/config.json" | sha256sum | cut -c1-64)}
export TF_FLASHNEXT_KVPAGE=1 TF_FLASHNEXT_KVPAGE_SHARED=1 TF_FLASHNEXT_EXL3_GRAPH=1 TF_FLASHNEXT_EXL3_GRAPH_PREFILL=1
export TF_FLASHNEXT_EXL3_GRAPH_ASYNC=${TF_FLASHNEXT_EXL3_GRAPH_ASYNC:-0}
if [ "${FAST:-1}" = 1 ]; then
  export TF_FLASHNEXT_KVPAGE_SHARED_LANES=${PARALLEL:-2}
  if [ "${PARALLEL:-2}" = 1 ]; then
    # one user: a 0.5 GiB device reserve (the supported minimum) and a 256 MB planner reserve let 80 lines fit
    export TF_FLASHNEXT_EXPERT_LINES=${LINES:-80} TF_FLASHNEXT_EXPERT_RESERVE_MB=${RESERVE_MB:-256} TENSORFOLD_MEMORY_RESERVE_GIB=${MEMORY_RESERVE_GIB:-0.5}
  else
    export TF_FLASHNEXT_EXPERT_LINES=${LINES:-68} TF_FLASHNEXT_EXPERT_RESERVE_MB=${RESERVE_MB:-512}
  fi
  extra=(--no-drafts)
else
  export TF_FLASHNEXT_EXPERT_LINES=${LINES:-38}
  extra=()
fi
CUDA_VISIBLE_DEVICES=$rank exec "$bin" serve "$model" --backend cuda --context "${CONTEXT:-32768}" --tp 2 --rank "$rank" \
  --master 127.0.0.1 --master-port "$master" --port "$port" --no-update-check --parallel "${PARALLEL:-2}" --prompt-cache-gib 0 "${extra[@]}" "$@"
