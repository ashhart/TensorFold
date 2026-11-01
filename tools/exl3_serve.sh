#!/bin/bash
# Serve the EXL3 Qwen3.8-Flash-Next pack on one 16 GB CUDA GPU (sm_120): the routed experts sit in pinned host memory and
# each layer keeps LINES of them on the device (default 95). Plain decode (no MTP head); 0.5 GiB device reserve.
# Usage: tools/exl3_serve.sh MODEL_DIR PORT [extra serve args]; CUDA_VISIBLE_DEVICES picks the card.
# Environment: LINES, RESERVE_MB (planner reserve), MEMORY_RESERVE_GIB, CONTEXT, PARALLEL, TENSORFOLD_BIN.
set -euo pipefail
model=$1 port=$2
shift 2
bin=${TENSORFOLD_BIN:-./zig-out/native/bin/tensorfold-native}
export TENSORFOLD_CHECKPOINT_SHA256=${TENSORFOLD_CHECKPOINT_SHA256:-$(cat "$model/model.safetensors.index.json" "$model/config.json" | sha256sum | cut -c1-64)}
export TF_FLASHNEXT_EXPERT_LINES=${LINES:-95} TF_FLASHNEXT_EXPERT_RESERVE_MB=${RESERVE_MB:-256} TENSORFOLD_MEMORY_RESERVE_GIB=${MEMORY_RESERVE_GIB:-0.5}
exec "$bin" serve "$model" --backend cuda --context "${CONTEXT:-32768}" --port "$port" --no-update-check \
  --parallel "${PARALLEL:-1}" --prompt-cache-gib "${CACHE_GIB:-0}" --no-drafts "$@"
