# PR 3733 asks 3 to 5

## 4. An H3 clip checkpoint on main against this head

This comparison can't be run on the Spark. Ash's main at `a6cde902d` has two tools that print H3 velocities
from a checkpoint. `tf-h3-dit` handles FastH3 clip cases and `tf-h3-image` handles picture cases (`zig/build/h3_host.zig:104-105`).
Both run only on Metal. CUDA's H3 on main is `libtf_h3.so` (`zig build tf-h3`). That library runs the transformer blocks
on rows the caller prepares. It has no case reader, no input and output projections, and no velocity output. The
clip case exporter, `tools/zig/h3_case.py`, lives on the `drowzeys/TensorFold` fork (`docs/recipes/minimax-h3-metal.md:47-49`).
It is in neither tree. The GB10 job `docs/recipes/minimax-h3-cuda/plant-203.sh` (side branch `gpu/203-plant`, `ba8ada1c7`) also lists every FastH3 checkpoint and H3 case file on that host.

On the clip path, this head differs from main only in `checkpoint.zig`. There, `Checkpoint.open` maps the same bytes with the
same flags and keeps the fd until `deinit`. `git diff a6cde902d HEAD -- zig/src/families/h3/h3.zig
zig/src/families/h3/geometry.zig zig/src/families/h3/block.zig zig/src/families/h3/cuda*.zig` is empty.

## CI on #3733

At `1ceaab3de`, every job (`fmt`, `zig`, `zig-b`, `zig-windows`, `pytest`, `release-paths`, `benchmark-clients`,
`comfyui-moss-installer`, `zig-macos-dev-tools`) is `QUEUED` and needs a maintainer's approval to run for a fork.
`host-nightly-*` is skipped. No job is red. Ash's main is not green either: its latest `No GPU` and `Zig format` runs at `75a86a0772` are
queued or pending, and the two before them, at `a6cde902d` and `924d50a9d`, were cancelled.
