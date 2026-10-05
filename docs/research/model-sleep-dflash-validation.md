# Configurable DFlash taps and sleep/wake

This integration combines sleep support from [PR #386](https://github.com/ashhart/TensorFold/pull/386)
with the pending DFlash v1 implementation in [PR #347](https://github.com/ashhart/TensorFold/pull/347).
The combined sleep PR includes compatibility checks and the sleep-cache changes needed for that
combination, with the drafting dependency's original commits preserved.
The [machine-readable receipt](model-sleep-dflash-results.json) records the qualified source digest,
checkpoint revisions, token hashes and lifecycle measurements.

## Configuration and scope

The drafter's `dflash_config.target_layer_ids` selects target hidden states in its declared order.
For `z-lab/Qwen3.5-4B-DFlash`, these are `[1, 5, 9, 13, 17, 21, 25, 29]` in a 32-layer target.
Legacy DFlash2 checkpoints without that field retain `[5, 19, 33, 47, 61]`.

Before initializing CUDA, startup checks nonempty, unique integer IDs, layer bounds, matching hidden
width, declared target depth and vocabulary, mask-token bounds, and the drafter projection's input
width. Unknown drafter architectures fail explicitly. Both prefill paths and both verification paths
preserve the declared order, including a descending order.

The IDs must match the hidden states used to train the drafter. Changing this list alone does not
make an unrelated checkpoint compatible. DFlash v1 and DFlash2 still use their respective architectures;
this work does not add arbitrary model-family support or change Nemotron's integrated MTP path.

DFlash v1 can retain different context lengths in sliding and full-attention layers. Prefix snapshots
now record each layer's row count and validate it before allocating restored tensors. Wake also checks
the drafter's tap order, window settings, block size and packing mode against the served runtime.
The startup estimate accounts for full-attention draft caches separately from sliding caches.

## Checkpoints

| Role | Checkpoint | Revision |
|---|---|---|
| Target | `capyctl/FrogNano-4B-2609-MLX-4bit` | `7175ee2b94eae649dfd1e7605cd2dc3dcd1e3e2f` |
| Drafter | `z-lab/Qwen3.5-4B-DFlash` | `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf` |

Validation uses single-device CUDA, checkpoint precision, BF16 prefill, 64 generated tokens,
temperature 1, top-k 20, top-p 0.95 and seed 1234. The MLX name describes the target's stored
weight format; these results do not qualify the Metal backend.

## Direct lifecycle results

| Configuration | Context / prompt | Cycles | Cached tokens per request | Median sleep / wake |
|---|---|---|---|---|
| Weights only, one stream | 4096 / 2048 | 2 | 0 | 1.85 / 6.82 s |
| Retained prefixes, one stream | 4096 / 2048 | 2 | 2047 | 2.18 / 6.64 s |
| Retained prefixes, eight streams | 16384 / 8192 | 2 | 8191 | 7.22 / 9.48 s |

All six cycles pass. Each drafted reply equals its serial counterpart, each concurrent reply equals
its solo counterpart, and tokens remain identical across wake. Every sleeping runtime reaches zero
CUDA allocated and reserved bytes. Retained prefixes restore lazily with zero load failures or memory
misses. The eight-stream snapshot contains 3,498,900,736 bytes.

The initial single-stream reply accepts 38 draft tokens; the eight initial concurrent replies accept
303 altogether. The long-prefix case exercises five draft layers with 4095 context rows and one with
8191 rows, exceeding the common-length assumption in the original sleep codec.

## Regression checks

Four CUDA tests pass for declared tap order across single and batched prefill and verification.
Host tests cover incompatible metadata before CUDA initialization, mixed-length prefix round trips,
corrupt snapshots before allocation, changed drafter settings during wake, and memory estimates for
full-attention draft layers. The metadata, ordering, snapshot and estimate regressions were reproduced
before their fixes.

Two authenticated HTTP cycles also pass with eight configured streams. They cover stream draining,
separate inference and lifecycle credentials, exact drafted/serial output, lazy prefix restoration,
and continuation of the same stored response after wake. The continued conversation reuses 51 tokens;
both cycles reach zero allocated and reserved CUDA bytes. Normal shutdown removes the snapshot files.

The focused host/CPU runs cover 567 distinct passing cases, with one collection skip. These runs
include the shared lifecycle, HTTP controls, cache store, both prefix codecs, dense-Qwen prompt-cache
bookkeeping, Nemotron sleep, family metadata and CUDA memory planning.

The existing `nvidia/Qwen3.8-27B-NVFP4` + `z-lab/Qwen3.8-27B-DFlash2` pairing was compared against
0.6.5 in release / integration / integration / release order, with three cold prompts per length
per pass. All first-token hashes match. Pooled median prefill times are:

| Prompt tokens | 0.6.5 | Integration | Change |
|---|---|---|---|
| 2048 | 0.694 s | 0.699 s | +0.77% |
| 8192 | 2.765 s | 2.781 s | +0.58% |

These differences are below 1%; they do not establish a speed change. This is a bounded 2K/8K
comparison, not the full 16K–64K prefill qualification. The DFlash2 reference also passes two
sleep/wake cycles with a 16K context and 8K prompt. Both restore the 8191-token prefix, preserve exact
serial/drafted output and reach zero CUDA allocated and reserved bytes while sleeping. Median sleep
and wake times are 13.15 and 30.20 seconds, with no cache load failures or memory misses.

Two-device operation, Metal, other DFlash v1 checkpoints, and BF16 drafter weights are not qualified here.
