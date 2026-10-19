# DeepSeek-V4.1-1017: parallel sparse attention on M3 Ultra

The measured prototype increased median decode throughput from 4.0519 to
4.8595 forwards/s on one M3 Ultra, a factor of 1.1993. All generated token IDs
and the tested teacher and lane logits match the predecessor byte for byte.
These measurements use commit `61ab890f46e1f4eb173eb72cd89c0e1766845331`.
They do not establish the speed of a later binary or qualify the complete
model against the original checkpoint.

## Runtime and inputs

[build.json](build.json) pins the upstream base, Zig 0.17.0,
`-Dcpu=apple_m3` and the four measured binaries. The machine has an M3 Ultra,
80 GPU cores and 256 GB unified memory. The checkpoint and original 1017
Engram tables are the same as in the [pair receipt](../ds1017-tp2-2026-10-19/model.json).
The [pack manifest](../ds1017-tp2-2026-10-19/pack-sha256.json) and both Engram
SHA256s were verified before execution.

Both modes stream Engram from local SSD and use split weight residency with
periodic refresh. `TF_DSV41_SERIAL` is unset in both modes. The labels
`serial` and `parallel` in the measurements distinguish the attention
pipeline, rather than the command encoder mode. The new pipeline is opt-in
with `TF_DSV41_PARALLEL_ATTENTION=1`; the default keeps the predecessor.

## Decode measurements

Each mode ran three times with the same 32 prompt IDs and 128 generated IDs.
Prefill produces the first completion token, leaving 127 measured decode
forwards. The table reports forwards/s, rather than 128 divided by decode time.

| Mode | Run 1 | Run 2 | Run 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Predecessor | 4.0519 | 4.0507 | 4.0536 | 4.0519 |
| Parallel attention | 4.8595 | 4.8532 | 4.8628 | 4.8595 |

[measurements.json](measurements.json) retains each wall time, prefill time and
GPU time. Decode GPU time fell from about 30.64 to 25.43 seconds. The separate
268-token teacher prefill became slightly slower, from 7.4010 to 7.5690
seconds. The result therefore does not show that every workload improves.

All six generation runs passed the execution I/O gate, with zero page-ins.
Each counted 126,844,928 Engram bytes; measured disk reads stayed within that
allowance. This gate permits counted Engram disk I/O.

[checks.json](checks.json) preserves each execution and residency record.

## Numerical and operator checks

The GPU suite passed 100 tests, the reference suite 28 and the EXL3 GPU suite
29. The new operator matched predecessor bytes across 51 shapes, including
tile edges and nonfinite inputs. Repeated runs preserved the inputs and guards.
The isolated warmed attention timings used 32 samples per case:

| Position | Predecessor median | Parallel median |
| --- | ---: | ---: |
| 128 | 0.8975 ms | 0.1802 ms |
| 8191 | 4.2574 ms | 0.7584 ms |

The 268-token teacher sequence matches the original single-rank logit SHA256
`910189ca833f31f7f905667f2df2f5b767391deac975dd34e7782ab694ad8eb2`
with the predecessor, the parallel pipeline and one-token teacher chunks.
Two lane slots at context 8192 and chunk 16 finish at positions 134/134 with
zero bit disagreements against solo execution. All six generation runs match
the [frozen token IDs](../ds1017-tp2-2026-10-19/prompts/gen-short-tokens.json).

## Reproduction and limits

Build the runner and GPU checks at the pinned commit. Use the same checkpoint,
Engram tables, token map and [prompt IDs](../ds1017-tp2-2026-10-19/prompts/).
Set `DSV41_ENGRAM_STREAM=1`, `TF_DSV41_SPLIT_RESIDENCY=1` and
`TF_DSV41_RESIDENCY_REFRESH=1`. Run `tf-dsv41-run generate` three times per
mode with 128 completion tokens. Leave `TF_DSV41_PARALLEL_ATTENTION` unset
for the predecessor and set it to `1` for the new pipeline. Compare the token
files and teacher `logits.f32`, and retain each `residency.json`, including
failed gates.

This receipt covers single-rank execution. It supplies no new TP2 or HTTP
serving result. The kernel-contract registration and explicit 256-thread
reservation were added later in `cdfa5203`; those changes have a separate
[runtime and performance receipt](../ds1017-tp2-slots-2026-10-20/README.md). Whole-model FP64, MTP, long-context and
Rust-normalizer qualification remain open. The protected measurement window
restored the original Qwen service and reranker and verified both responses.
