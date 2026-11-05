#!/bin/bash
# Serve the EXL3 Flash Next pack on one CUDA GPU with routed experts in pinned host memory: MODEL_DIR PORT [args].
# Environment: LINES, RESERVE_MB, MEMORY_RESERVE_GIB, CONTEXT, PARALLEL, CACHE_GIB, TENSORFOLD_BIN.
set -euo pipefail
model=$1 port=$2
shift 2
bin=${TENSORFOLD_BIN:-./zig-out/native/bin/tensorfold-native}
if [ -z "${TENSORFOLD_CHECKPOINT_SHA256:-}" ]; then
  TENSORFOLD_CHECKPOINT_SHA256=$(cat "$model/model.safetensors.index.json" "$model/config.json" | sha256sum | cut -c1-64)
fi
export TENSORFOLD_CHECKPOINT_SHA256
export TF_FLASHNEXT_EXPERT_LINES=${LINES:-95} TF_FLASHNEXT_EXPERT_RESERVE_MB=${RESERVE_MB:-256}
export TENSORFOLD_MEMORY_RESERVE_GIB=${MEMORY_RESERVE_GIB:-0.5}
exec "$bin" serve "$model" --backend cuda --context "${CONTEXT:-32768}" --port "$port" --no-update-check \
  --parallel "${PARALLEL:-1}" --prompt-cache-gib "${CACHE_GIB:-0}" --no-drafts "$@"
