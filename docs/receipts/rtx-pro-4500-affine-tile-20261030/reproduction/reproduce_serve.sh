#!/usr/bin/env bash
set -Eeuo pipefail
: "${TF_SRC:?Set TF_SRC to the tested checkout}"
: "${MODELS:?Set MODELS to the parent of both pinned checkpoint directories}"
export TENSORFOLD_CUDA_KERNELS=native
export TENSORFOLD_MEMORY_RESERVE_GIB=0.5
exec "$TF_SRC/zig-out/native/bin/tensorfold-native" serve "$MODELS/Qwen3.8-27B-MLX-4bit" \
  --drafter "$MODELS/Qwen3.8-27B-DFlash2" --name bench --host 127.0.0.1 --port 18080 \
  --context 131072 --parallel 1 --kv-cache fp8 --prompt-precision int8 \
  --vision --prompt-cache-gib "${PROMPT_CACHE_GIB:-8}" --max-tokens 2048
