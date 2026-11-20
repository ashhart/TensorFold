#!/bin/sh
# d3_cfg_mix's kernel contract on TOP's Spark (sm_121, nvcc 13.0) through the Desk's GPU door:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/drama3-cuda/cfg-mix-gb10.sh
# Gates 1-4, the planted non-finite fill and the limits check; exits with tf-cuda-test's status.
set -eu
staged=/assets/drama3-pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
nvcc=/usr/local/cuda/bin/nvcc
tar -C /work -xJf "$staged/$zig.tar.xz"
tar -C /work -xJf "$staged/$gcc.tar.xz"
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
echo "commit $(git -c safe.directory='*' rev-parse HEAD)"
"$nvcc" --version | tail -n 1
zig build install fatbins kernel-contract-fatbins -Dsm=121 -Dnvcc=$nvcc -Doptimize=ReleaseFast --prefix /work/sm121
zig build -Dgpu=spark -Dnvcc=$nvcc --prefix /work/sm121
exec /work/sm121/bin/tf-cuda-test kernel-contract 'd3_cfg_mix'
