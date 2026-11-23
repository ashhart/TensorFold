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
It is in neither tree. On this Mac, a search of the home directory and mounted volumes, six levels deep, matched 122 `*.h3case.safetensors` files under
`h3-image-runs/`. All of them are MiniMax-H3-Image picture cases (`text`, `video`, `video_velocity`, no `audio`). The only
models are `MiniMax-H3-Image` and `ComfyUI-MiniMax-H3-Image`; there is no FastH3 clip checkpoint. The earlier TOP search
scrolled out of the door's log. `plant-203.sh` at `42a57c64f` (side branch `gpu/203-plant-h3`) now writes it to `checkpoints.txt`.

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

The five-SM audit, `kernel-contract-limits` over `-Dsm=80,86,89,120,121` fatbins, reports `105 problems`.
All of them are stale records on 47 launches that aren't H3: `muse_linear` (17), `muse_glue`, `exl3_experts`, `GLM` (4 each),
`qwen35`, `flashnext_hcn`, `exl3_grouped` (3 each), `nemo_audio`, `muse_dflash2` (2 each), and `qmmf`, `qmm_prefill`,
`glm_glue`, `flashnext_mtp` and `drama3` (1 each). The list is in `limits-stale-top-7c6183220113.txt`. No H3 record is stale.
These problems already exist on Ash's main: against `bdcb6e608` this head adds only `h3_image_decoder.cu`, `h3_image_prompt.cu`,
`h3_image_cuda_records.zig` and their line in `registry_joins.zig` under `zig/kernels/cuda` and `zig/src/core/contract`.
Without `h3`, the full contract has never finished on TOP. Job `top-1c5bc91282e3` spent 75 minutes in moss's `tf_md_conv3x3`
float64 shapes reference, whose code is the same on Ash's main, and the Desk stopped it there. Its 3629 lines up to that
point show every H3 gate passing. The Lead left that one non-H3 reference out of ask 5, so `kernel-contract` now takes
`-WORD`s (`c759a1372`): `kernel-contract -tf_md_conv3x3` runs every other kernel's gates and controls, the back and after
suites, and every limit record, `tf_md_conv3x3`'s included. One job at `93f592f6b` (`plant-203.sh` on `gpu/203-plant-h3`) runs
it, both selftests, the five-SM audit and ask 4's search. That job is in TOP's queue.

## CI on #3733

At `04f9ad328` (23 Nov), every job passes: `zig`, `zig-b`, `zig-windows`, `zig-macos-dev-tools`, `fmt`, `pytest`,
`release-paths (ubuntu-24.04)`, `comfyui-moss-installer` and `benchmark-clients` on Ubuntu and Windows.
`host-nightly-linux` and `host-nightly-macos` are skipped. No job is red.
