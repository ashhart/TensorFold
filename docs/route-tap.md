# MoE route tap

`TF_ROUTE_TAP=/path/to/trace` is an optional, off-by-default switch for inspecting or monitoring MoE routing.
Unset or empty, the tap is inert: no host thread, no device copies, no encoder barrier or blit, no extra wait work, and the engine's tokens and speed stay as they are. Tap-off records the same command-buffer work as a build without the tap.

When the path is set, rank 0 of a tapped family writes a versioned trace of each layer's global top-k
(ids and weights) after the step's existing wait. Other ranks do not open the file. The host ring is
independent of the writer thread; a full ring drops the newest record.

## Tapped families

The switch is honoured only for these `model_type` names:

- `glm5_next`
- `glm_moe_dsa`
- `qwen3_5_moe`
- `qwen4_exp`
- `nemotron_h`
- `nemotron_h_omni`
- `minimax_m3_vl`
- `step5_vl`
- `lfm2_moe`

Every other family is refused by name while the variable is set, including `deepseek_v41`.
Unset, those families open as they do today.

These family and backend pairs never copy, so `admit` refuses them by name while the variable is set:

- XPU `nemotron_h`
- Metal `qwen4_exp` (Flash Next)
- Metal `qwen3_5_moe`

## CUDA and HIP graphs

While the tap is live, tapped families run without CUDA or HIP graphs. Per-layer device-to-host copies run in host code that decode graph capture also runs (`cuda/graph.zig` `beginCapture`, HIP's twin). A pageable copy inside a capture invalidates it; a pinned copy would still replay into the same captured slot every step, with no host bookkeeping. CUDA Nemotron skips MTP and window capture when the tap is live, so its glue kernels run eagerly (`Head.launch` and `verifyMasked` already fall back when no graph was captured). `beginCapture` still refuses a live tap if a caller tries to start one. Tap-off is unchanged: graphs capture and replay as they do today.

A failed native open after `admit` calls `stop`, so the pinned ring and records are freed. `close` still stops a successful open.

The host ring for those copies is pinned (`cuMemHostAlloc` / `hipHostMalloc`) so a live tap does not stage through pageable memory twice a layer.

A step's slots become readable after an event recorded behind that step's copies, not after an unrelated stream synchronize. A copy that fails clears its slot and counts as a drop; `stop` logs the drop count.

## Trace format (TFR1)

Little-endian. A 16-byte header, then records:

| Offset | Type | Field |
| --- | --- | --- |
| 0 | `[4]u8` | magic `TFR1` |
| 4 | `u16` | version (`1`) |
| 6 | `u16` | flags (`1` = weights present) |
| 8 | `u16` | reserved top-k hint (`0`: each record carries its own `n`, which is `rows × k` and varies) |
| 10 | `u16` | reserved |
| 12 | `[4]u8` | pad |

Each record:

| Offset | Type | Field |
| --- | --- | --- |
| 0 | `u32` | step |
| 4 | `u16` | layer |
| 6 | `u16` | `n` (ids in this record) |
| 8 | `n × i32` | expert ids (global, before localisation) |
| 8+4n | `n × f32` | routing weights |

A worked example is [`route-tap-sample.tfr1`](route-tap-sample.tfr1).
`python3 tools/route_tap_bench.py decode docs/route-tap-sample.tfr1` prints its records.

## When copies happen

- CUDA: async device-to-host of the pick and weight buffers after the global top-k, before localisation.
- HIP: the same, before `tf_moe_localize` or `tf_nemo_localize`.
- Metal: a blit of each layer's picks into a shared-storage ring in the same command buffer. The buffer barrier sits inside that blit and runs only when the tap is live. The host reads only that buffer's slots at its existing wait. Pipelined later buffers stay unpublished until they land. A slot that is still all-zero after the wait is dropped, not written. No extra wait is added. With the variable unset, `afterWait` / `afterLand` and `reservedCount` are not called.

## Checks

Host tests in `zig/src/core/route_tap.zig` cover ring wraparound, the header/record layout (including `top_k = 0`), the tapped names, the no-copy family/backend pairs, that an unset variable is a no-op, that `stop` frees the ring, and that a failed open after `admit` (the `Guard` path) frees it under the testing allocator. `tools/route_tap_bench.py decode` fails on an all-zero record.

On a GPU host, `tools/route_tap_bench.py` records token equality (tap on vs off), decode speed with the variable unset against the same binary, and the on-cost at 1 and 8 streams. A receipt names only the host RAM class and the GPU model.

On a Mac, `tools/run_metal_proofs.py` runs tap-on vs tap-off token equality for GLM (`glm5_next`), GLM-DSA (`glm_moe_dsa`), Step5 (`step5_vl`), MiniMax (`minimax_m3_vl`) and LFM2 (`lfm2_moe`). The blit barrier runs only when the tap is live. `tools/run_metal_ab.py` A/Bs main against this head with the tap unset (chat T0 at 1 and 2 streams; code T0 at 2 and 4; code T1 at 1 and 4), then checks tap-on token equality and shutdown leaks.
