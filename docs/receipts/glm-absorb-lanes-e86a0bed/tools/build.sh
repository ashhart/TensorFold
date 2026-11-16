#!/bin/bash
# usage: build.sh TREE...; each ~/glm-perf/TREE with the GB10 serving kernels; ~/glm-perf/build-TREE.{log,done}.
export PATH=/usr/local/bin:$PATH
for t in "$@"; do
  cd ~/glm-perf/$t || exit 1
  { zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dtp-nvcc=/usr/local/cuda/bin/nvcc; echo "rc=$?"; } > ~/glm-perf/build-$t.log 2>&1
  tail -1 ~/glm-perf/build-$t.log > ~/glm-perf/build-$t.done
done
