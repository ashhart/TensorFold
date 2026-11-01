# MoE route tap

`TF_ROUTE_TAP=/path/to/trace` is an optional, off-by-default switch for inspecting or monitoring MoE routing.
Unset or empty, the tap is inert: no host thread, no device copies, and the engine's tokens and speed stay as they are.

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
- Metal: a blit of each layer's picks into a shared-storage ring in the same command buffer; the host reads only that buffer's slots at its existing wait. Pipelined later buffers stay unpublished until they land. A slot that is still all-zero after the wait is dropped, not written. No extra wait is added.

## Checks

Host tests in `zig/src/core/route_tap.zig` cover ring wraparound, the header/record layout (including `top_k = 0`), the tapped names, and that an unset variable is a no-op. `tools/route_tap_bench.py decode` fails on an all-zero record.

On a GPU host, `tools/route_tap_bench.py` records token equality (tap on vs off), decode speed with the variable unset against the same binary, and the on-cost at 1 and 8 streams. A receipt names only the host RAM class and the GPU model.
