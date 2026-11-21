#!/bin/sh
# d3_cfg_mix's kernel contract on a DGX Spark (GB10, sm_121) with Zig 0.17.0 and CUDA 13.0's nvcc on PATH:
#   sh docs/recipes/drama3-cuda/cfg-mix-gb10.sh
# Gates 1-4, the planted gamma - 1 control (must go red) and the limits check; exits with tf-cuda-test's status.
set -eu
nvcc=$(command -v nvcc)
echo "commit $(git rev-parse HEAD)"
"$nvcc" --version | tail -n 1
zig build install fatbins kernel-contract-fatbins -Dsm=121 -Dnvcc="$nvcc" -Doptimize=ReleaseFast --prefix zig-out/sm121
zig build -Dgpu=spark -Dnvcc="$nvcc" --prefix zig-out/sm121
exec zig-out/sm121/bin/tf-cuda-test kernel-contract d3_cfg_mix
