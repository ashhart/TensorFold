#!/bin/bash
# Compiles DeepSeek-V4.1 kernel copies to fatbins in the NGC PyTorch image with -Xptxas -v; reports spills / stack.
set -euo pipefail
# usage: OUT=<dir> SM=121,120 bash ptxas.sh name[:flags] ...   (flags comma separated as in cuda.zig, e.g. "x3ld:-O3")
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:?set OUT}"
SM="${SM:-121}"
IMAGE="${TF_ZIG_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
mkdir -p "$OUT"
TORCH="-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20"
cmds=""
for spec in "$@"; do
  name="${spec%%:*}"; flags="${spec#*:}"; [ "$flags" = "$spec" ] && flags="-O3"
  gen=""
  for sm in ${SM//,/ }; do gen="$gen -gencode=arch=compute_${sm},code=sm_${sm}"; done
  cmds="$cmds nvcc -fatbin $TORCH ${flags//,/ } $gen -Xptxas -v -o /out/$name.fatbin /k/$name.cu > /out/$name.ptxas 2>&1 && echo built $name || { echo FAILED $name; tail -20 /out/$name.ptxas; } ;"
done
docker run --rm --network none --memory "${TF_ZIG_DOCKER_MEMORY:-12g}" --memory-swap "${TF_ZIG_DOCKER_MEMORY:-12g}" -u "$(id -u):$(id -g)" -v "$HERE:/k:ro" -v "$OUT:/out" --entrypoint bash "$IMAGE" -c "$cmds"
for spec in "$@"; do
  name="${spec%%:*}"
  f="$OUT/$name.ptxas"
  [ -f "$f" ] || continue
  fns=$(grep -c "Compiling entry function" "$f" || true)
  spill=$(grep -E "bytes spill (stores|loads)" "$f" | grep -vc " 0 bytes spill stores, 0 bytes spill loads" || true)
  stack=$(grep -E "bytes stack frame" "$f" | grep -vc " 0 bytes stack frame" || true)
  echo "$name: $fns entry functions; with spills: $spill; with a stack frame: $stack"
done
