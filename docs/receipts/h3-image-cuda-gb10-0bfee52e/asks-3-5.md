# PR 3733 asks 3 to 5

## 3. A planted control for every new kernel

Each of the 19 decoder and prompt kernels has a "first block only" control (`zig/tests/cuda/contract_h3_register.zig`).
TOP job `top-a4a49aacbcd1` ran `tf-cuda-test kernel-contract -tf_md_conv3x3` at `93f592f6b` on the GB10, and every control
went red at `shapes`, the gate it is planted for. Each control's gate line and verdict are in
`controls-top-a4a49aacbcd1.txt`, 38 lines for 19 controls. For example:

```
FAIL shapes cuda h3 image decoder h3d_attention control: first block only: at 1x261x64 output 64 is -nan where float64 gives 9.543158e-1: nan of its bound
CONTROL cuda h3 image decoder h3d_attention control: first block only: red at shapes, as planted
FAIL shapes cuda h3 image prompt h3p_heads rotary control: first block only: at 8x128x128 output 1024 is 3.984375e-1 where float64 gives 4.902344e-1: 11.95 of its bound
CONTROL cuda h3 image prompt h3p_heads rotary control: first block only: red at shapes, as planted
```

In the same log (`contract-h3-top-a4a49aacbcd1.txt`), every H3 gate line without a control is `PASS` (86) or `N/A` (10).

## 4. An H3 clip checkpoint on main against this head

This comparison can't be run on the Spark. Ash's main at `a6cde902d` has two tools that print H3 velocities
from a checkpoint. `tf-h3-dit` handles FastH3 clip cases and `tf-h3-image` handles picture cases (`zig/build/h3_host.zig:104-105`).
Both run only on Metal. CUDA's H3 on main is `libtf_h3.so` (`zig build tf-h3`). That library runs the transformer blocks
on rows the caller prepares. It has no case reader, no input and output projections, and no velocity output. The
clip case exporter, `tools/zig/h3_case.py`, lives on the `drowzeys/TensorFold` fork (`docs/recipes/minimax-h3-metal.md:47-49`).
It is in neither tree. On this Mac, a search of the home directory and mounted volumes, six levels deep, matched 122 `*.h3case.safetensors` files under
`h3-image-runs/`. All of them are MiniMax-H3-Image picture cases (`text`, `video`, `video_velocity`, no `audio`). The only
models are `MiniMax-H3-Image` and `ComfyUI-MiniMax-H3-Image`; there is no FastH3 clip checkpoint. On TOP, job
`top-a4a49aacbcd1`'s `find / -xdev` for `*fasth3*`, `*h3*case*`, `case.safetensors` and `*minimax*h3*` found 0 paths.

The transformer diff against Ash's main is empty for `h3.zig`, `geometry.zig`, `block.zig` and the CUDA block sources.
The shared Metal loader does change: `tensors.zig` now uses the span-based `read_ahead.zig`, and `checkpoint.zig` is the
CUDA-side checkpoint reader. Static inspection is therefore not a substitute for the unavailable velocity comparison.

## 5. CUDA selftests and the GB10 kernel contract

TOP job `top-7c6183220113` ran `plant-203.sh` (side branch `gpu/203-plant-h3`, `30c9f2302`) on the GB10 and exited 0.
Both selftests pass: `decoder-selftest.txt` and `prompt-selftest.txt` end in `selftest: pass` and `PASS every gate`.
`tf-cuda-test kernel-contract h3` (`contract-h3-top-7c6183220113.txt`) runs the H3 kernels' gates, controls and live limits:

```
(filtered) SUMMARY 34 kernels: 34 pass gates 1-4, 0 fail, 0 known findings; 19 controls: 19 red as planted, 0 missed; 0 limit problems; gates 8447 ms, limits 1 ms
```

The five-SM audit, `kernel-contract-limits` over `-Dsm=80,86,89,120,121` fatbins, reports `105 problems`, none of them H3.
These problems already exist on Ash's main: against `bdcb6e608` this head adds only `h3_image_decoder.cu`, `h3_image_prompt.cu`,
`h3_image_cuda_records.zig` and their line in `registry_joins.zig` under `zig/kernels/cuda` and `zig/src/core/contract`.
Job `top-a4a49aacbcd1` (`plant-203.sh` at `93f592f6b`) ran the full contract without `tf_md_conv3x3`'s reference,
the Lead's ask 5. Its log is `contract.log` in the job's out folder, 2176 lines. Both selftests pass again, and every H3
line is as above. The run still hit the script's 45-minute limit (exit 124) in `window_rows`'s paged cases, before the
GLM, MLA, q35 contiguous, MTP rows, weight-only and family-set suites and the run's SUMMARY. Its only red lines outside
the controls are four `qwen35` nonfinite gates (`head 18 element 0: finite where float64 gives nan`). `qwen35`'s kernels
and `contract_q35_attn.zig` are the same as on Ash's main. The same job's five-SM audit again reports `105 problems`:
stale records on 44 launches, none H3, all in `limits-stale-top-a4a49aacbcd1.txt`.

## CI on #3733

At `e27a4ecf4` (23 Nov), every job passes: `zig`, `zig-b`, `zig-windows`, `zig-macos-dev-tools`, `fmt`, `pytest`,
`release-paths (ubuntu-24.04)`, `comfyui-moss-installer` and `benchmark-clients` on Ubuntu and Windows.
`host-nightly-linux` and `host-nightly-macos` are skipped. At `64fdb3215`, `zig` was red at lean check on a two-line
comment in `contract_limits.zig` from this branch. Ash's main is not red there, and `e27a4ecf4` fixes it.
