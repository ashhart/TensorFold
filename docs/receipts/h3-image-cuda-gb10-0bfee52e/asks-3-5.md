# PR 3733 asks 3 to 5

## 3. A planted control for every new kernel

Each of the 19 decoder and prompt kernels has a "first block only" control (`zig/tests/cuda/contract_h3_register.zig`).
TOP job `top-1c5bc91282e3` ran `tf-cuda-test kernel-contract` at `1ee412bf0` on the GB10, and every control went red at
`shapes`, the gate it is planted for. Each control's gate line and verdict are in `controls-top-1c5bc91282e3.txt`, 38 lines for
19 controls. For example:

```
FAIL shapes cuda h3 image decoder h3d_attention control: first block only: at 1x261x64 output 64 is -nan where float64 gives 9.543158e-1: nan of its bound
CONTROL cuda h3 image decoder h3d_attention control: first block only: red at shapes, as planted
FAIL shapes cuda h3 image prompt h3p_heads rotary control: first block only: at 8x128x128 output 1024 is 3.984375e-1 where float64 gives 4.902344e-1: 11.95 of its bound
CONTROL cuda h3 image prompt h3p_heads rotary control: first block only: red at shapes, as planted
```

In the same log, all 122 H3 gate lines without a control are `PASS`.
The Desk stopped that job in `tf_md_conv3x3`'s float64 shapes reference, a moss kernel that is not part of H3.

## 4. An H3 clip checkpoint on main against this head

This comparison can't be run on the Spark. Ash's main at `a6cde902d` has two tools that print H3 velocities
from a checkpoint. `tf-h3-dit` handles FastH3 clip cases and `tf-h3-image` handles picture cases (`zig/build/h3_host.zig:104-105`).
Both run only on Metal. CUDA's H3 on main is `libtf_h3.so` (`zig build tf-h3`). That library runs the transformer blocks
on rows the caller prepares. It has no case reader, no input and output projections, and no velocity output. The
clip case exporter, `tools/zig/h3_case.py`, lives on the `drowzeys/TensorFold` fork (`docs/recipes/minimax-h3-metal.md:47-49`).
It is in neither tree. The GB10 job `docs/recipes/minimax-h3-cuda/plant-203.sh` (side branch `gpu/203-plant-h3`, `30c9f2302`) also searches that host for FastH3 checkpoints and H3 case files.

The transformer diff against Ash's main is empty for `h3.zig`, `geometry.zig`, `block.zig` and the CUDA block sources.
The shared Metal loader does change: `tensors.zig` now uses the span-based `read_ahead.zig`, and `checkpoint.zig` is the
CUDA-side checkpoint reader. Static inspection is therefore not a substitute for the unavailable velocity comparison.

## CI on #3733

At `c203accb0` (23 Nov 05:2x): `pytest`, `release-paths` and `comfyui-moss-installer` pass. `zig` and `zig-macos-dev-tools`
were cancelled. `zig-windows` and `benchmark-clients (windows-latest)` are in progress. `fmt`, `zig-b` and
`benchmark-clients (ubuntu-latest)` are queued, and `host-nightly-*` is skipped. No job is red. Ash's main isn't green either:
`No GPU` and `Zig format` at `bdcb6e6086` are queued or pending, and every run before that at `df0f420161`, `a325f58258`
and `833d8ec465` was cancelled or is still pending.
