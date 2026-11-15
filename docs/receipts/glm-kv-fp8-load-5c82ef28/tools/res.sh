#!/bin/bash
# usage: res.sh NVCC DIR... : glm_glue for sm 80,86,89,120,121 from DIR/zig/kernels/cuda, FP8 chunk kernels' usage per SM
nvcc=$1; shift
for d in "$@"; do
  g=""; for sm in 80 86 89 120 121; do g="$g -gencode=arch=compute_$sm,code=sm_$sm"; done
  $nvcc -fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
    -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 -I $d/zig/kernels/cuda $g \
    -o $d.glm_glue.fatbin $d/zig/kernels/cuda/glm_glue.cu || exit 1
  echo "== $d $($nvcc --version | tail -2 | head -1)"
  $(dirname $nvcc)/cuobjdump -res-usage $d.glm_glue.fatbin 2>&1 | awk '/arch = /{a=$0} /Function tf_glm_chunks_(16|32)(_fp8)?:/{f=$2; getline; sub(/^[ \t]+/,""); print a" | "f" "$0}' | sed 's/.*arch = sm_/sm_/' | sort
done
