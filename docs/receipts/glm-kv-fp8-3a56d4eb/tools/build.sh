#!/bin/bash
# Build each tree under ~/fl-receipt given on the command line with the GB10 kernels; writes ~/fl-receipt/build-<tree>.{log,done}.
export PATH=/usr/local/bin:$PATH
for t in "$@"; do
  cd ~/fl-receipt/$t || exit 1
  { zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 && zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dtp-nvcc=/usr/local/cuda/bin/nvcc; echo "rc=$?"; } > ~/fl-receipt/build-$t.log 2>&1
  tail -1 ~/fl-receipt/build-$t.log > ~/fl-receipt/build-$t.done
done
