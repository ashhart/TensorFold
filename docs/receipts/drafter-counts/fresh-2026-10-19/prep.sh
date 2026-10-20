#!/bin/bash
# #1001 fresh gfx1151 run, step 1 (no service touched): merge the three PR heads onto main 76aafcf2 and build.
#   bash ~/tf-1001-fresh-runs/prep.sh > ~/tf-1001-fresh-runs/prep.log 2>&1
set -u
R=$HOME/tf-1001-fresh-runs; mkdir -p $R
Z=$HOME/zig/zig-x86_64-linux-0.17.0/zig
F=https://github.com/ThinkOffApp/TensorFold.git
cd $HOME/tf-1001 || exit 1
git fetch -q $F benchmark-rocm-host-identity:refs/remotes/fork/benchmark-rocm-host-identity \
  hip-refuses-drafter:refs/remotes/fork/hip-refuses-drafter \
  benchmark-drafter-counts:refs/remotes/fork/benchmark-drafter-counts || exit 1
rm -rf $HOME/tf-1001-fresh; git worktree prune
git worktree add -q -B tf1001-fresh $HOME/tf-1001-fresh 76aafcf2 || exit 1
cd $HOME/tf-1001-fresh || exit 1
git -c user.name=ThinkOffApp -c user.email=233579950+ThinkOffApp@users.noreply.github.com \
  merge -q --no-edit fork/benchmark-rocm-host-identity fork/hip-refuses-drafter fork/benchmark-drafter-counts || exit 1
{ echo "merge $(git rev-parse HEAD)"; echo "main 76aafcf2"; for b in benchmark-rocm-host-identity hip-refuses-drafter benchmark-drafter-counts; do echo "$b $(git rev-parse fork/$b)"; done; } | tee $R/heads.txt
df -h / | tail -1; free -g | sed -n 2p
$Z build native-hip -Dhip-arch=gfx1151 -Dhipcc=/usr/bin/hipcc -Dhip-include=/usr/include \
  -Dhip-device-lib=/usr/lib/llvm-21/lib/clang/21/amdgcn/bitcode -Doptimize=ReleaseFast -j8 > $R/build.log 2>&1
echo "build_exit=$?"
sha256sum $HOME/tf-1001-fresh/zig-out/native-hip/bin/tensorfold-native | tee $R/binary.sha256
echo PREPDONE
