# Full GLM-5.3 (`glm_moe_dsa`) on four DGX Sparks — design and plan

Status: **working on four DGX Sparks** (branch `glm-moe-dsa` off v0.5.0). Milestones M0-M5 and most of M6 are
done: four-rank CLI and collectives, the family, a fused CUDA-graph forward, MTP and DFlash2 drafts (drafted ==
serial), token-level DSA past 2,048 tokens, and a fast prompt path. Measured results and how to run it:
`docs/recipes/glm-5.3-full-tp4.md`. Remaining: the 1M-token context (DCP, M7) and a larger context by default. The
sections below are the original plan; estimates in them predate the measurements.

## Why

GLM-5.3 (753B total, ~40B active) is the largest open model that fits four DGX Sparks. Today TensorFold serves
GLM-5.3-Flash (`glm5_next`) on two ranks; the full model needs a new family and four ranks. On the same four Sparks
and the same 2.75 bpw EXL3 checkpoint, vLLM decodes in 79 ms per MTP k=2 step (23.8 prose / 30.9 code tok/s) and
prefills ~730 tok/s. A TensorFold engine should beat both, and bring TensorFold's exactness (drafted == serial).

## Checkpoint

`GLM-5.3-EXL3-2.75-mixedK` (ours, from stock zai-org BF16, exllamav3 1.5 fork): routed experts EXL3 `mul1`, a width
per expert (2/3/4-bit, mean 2.75; 81/158/17 of 256 on layer 3), MTP-layer experts 8-bit; q_a, kv_a (stored 640 = 576
+ zero pad), q_b, o_proj, shared-expert and dense-MLP linears and indexer wq_b 5-bit EXL3; kv_b, indexer wk /
weights_proj / k_norm, router, norms, embed, lm_head, MTP eh_proj bf16. 276 GB, ~68 GB per rank at TP 4.
Quality vs BF16 (65.5K-position KLD, top-1) is being measured with exllamav3 `model_diff`.

## What differs from GLM-5.3-Flash

| Component | GLM-5.3-Flash | Full GLM-5.3 | In TensorFold |
|---|---|---|---|
| Residual | 4 hyper-connection streams | plain | reuse `glue.residual_add`, drop HC |
| Token mixers | KDA + DSA MLA hybrid | 78 × MLA + DSA | drop KDA |
| MLA | NoPE, latent 512 | RoPE every layer: latent 512 + k_rope 64 = 576; q 192 nope + 64 rope; v 256; interleaved RoPE θ 8e6 | change `latent.py` (576 score / 512 value), new interleaved-RoPE kernel |
| Indexer | pooled top-512 pools | token-level top-2048 (V3.2 style), `indexer_types` full/shared (freq 4, offset 3), MTP reuses top-k | change `sparse.py` keys pools → tokens, streaming/tiled top-2048 |
| Experts | 4-bit mcg only (`glm5_next/cuda/exl3.cu`) | mul1, width per expert, 8-bit MTP | reuse universal `cuda/exl3/experts.py` |
| Dense linears | MLX 4-bit / bf16 | 5-bit EXL3 | reuse `cuda/exl3/linear.py` + `prefill.py` |
| Drafts | MTP + DFlash2 | MTP only (no DFlash2 exists) | reuse `mtp.py`, `drafter_choice.py` |

Recommendation: a new family `families/glm_moe_dsa/`, importing the MLA/DSA/glue/graph kernels from `glm5_next`
(or moving them to a shared `tensorfold/cuda/mla/`), rather than branching inside `glm5_next`.

## Four ranks

The two-rank limit is hard-coded in a few places (CLI `--tp` choices, `glm5_next` engine and weights `world = 2`,
`split.py` halves and `RankReader`, `capacity.gather_ints`); `comm.NCCL`, `forward.gather`, `glue.residual_add`,
`geometry.split_weights` and the decode top-k gather are already world-generic.

| Weight | Split | Per rank |
|---|---|---|
| q_a, kv_a, their norms; indexer; router; norms; embed | replicate | full |
| q_b | output columns (tiles + svh) | 16 of 64 heads |
| kv_b | rows by head | 16 heads |
| o_proj | input rows (tiles + suh) | K 4096 |
| routed / shared / dense gate-up | output columns | 512 / 512 / 3072 |
| routed / shared / dense down | input rows | K 512 / 512 / 3072 |
| lm_head | vocab rows | 38,720 |

MLA's latent is shared by all heads, so tensor parallelism cannot shard it: every rank writes the same 576-wide
latent and attends it with its 16 heads.

Collectives: two per layer (after o_proj and after the MLP), fp32 partials summed in rank order. At decode that is
~160 small all-gathers per step (74 KB per row) — latency-bound, ~8-13 ms per step if NCCL takes 50-80 µs over the
RoCE fabric. At prefill the fp32 all-gather is 4× the bytes of a bf16 all-reduce at four ranks (2× at two): ~15 ms
per collective per 2,048-row chunk, capping prefill near 850 tok/s. An exact reduce-scatter (all-to-all of fp32
column quarters, rank-order sum and residual on the owned quarter, bf16 all-gather of the result) keeps the same
bits for ~⅓ of the bytes (~2,300 tok/s ceiling from communication); it needs grouped send/recv in `comm.py`.

Decode estimate at k=2 (3 rows): ~8.4 GB of weights per rank per step (~38 ms at 220 GB/s) + ~11 ms collectives +
~4 ms attention and launches ≈ 52 ms, against vLLM's 79 ms.

## Memory per rank

Weights ~70 GB; admission reserve 12.8 GB; CUDA/NCCL ~4 GB (NCCL ≥ 2.30; 2.28.9 alone takes 15 GiB); workspaces and
graphs ~5 GB ⇒ ~35 GB for caches.

| Cache layout | per token per rank | 128K | 256K | 1M | fits |
|---|---|---|---|---|---|
| bf16 latent + bf16 index keys, replicated | 96 KB | 12.6 GB | 25 GB | 101 GB ✗ | ~360K |
| fp8 latent (656 B) + fp8 index keys | 54 KB | 7.1 GB | 14 GB | 57 GB ✗ | ~640K |
| context-parallel over 4 ranks, fp8 | ~16 KB | 2 GB | 4 GB | 16 GB ✓ | >2M |

1M needs the latent sharded across ranks (decode context parallel): tokens interleaved by position mod 4, each rank
attends its keys for all heads, partial (o, m, l) merged in rank order. It is exact per configuration (new bits vs
TP-only), costs one more collective per layer, so it would be a long-context startup mode. The token-level
indexer should shard the same way (local top-2048, ordered merge), which also cuts scoring 4×. Scoring 2,048 rows
against 1M keys in fp32 is 8.6 GB, so scoring must stream with a running top-2048 in any layout.

## Milestones (~45-55 person-days)

| # | Milestone | Days | Acceptance |
|---|---|---|---|
| M0 | world = N plumbing: `--tp 4`, grouped send/recv in `comm.py`, N-way `split.py` with EXL3 dense rules, `capacity.gather_ints(world)`, all-rank settings check, 4-way thread-comm test fixture | 3-4 | reassembled splits byte-equal the source; nccl-tests latency at 24-300 KB and 50 MB on four Sparks |
| M1 | `glm_moe_dsa` config and loader: universal EXL3 experts incl. 8-bit MTP, `Exl3Linear`, 640→576, geometry for 576 + token indexer | 4-5 | dequant equals ExLlamaV3 `reconstruct` per tensor; startup estimate within 3% of peak |
| M2 | eager TP4 serial, context ≤ 2,048: RoPE MLA, absorb/expand, MoE, dense MLP, vocab-split head | 7-9 | serves correct text on four Sparks; teacher-forced logits vs the vLLM serve of the same checkpoint: top-1 ≥ 98%, KL well under the quantization KL |
| M3 | CUDA graphs, rows 1-4 | 3 | graphs equal eager; serial step ≤ 45 ms |
| M4 | MTP drafts and policies | 3-4 | drafted equals serial; ≥ 24 prose / 31 code tok/s |
| M5 | token-level DSA past 2,048: V3.2 indexer, full/shared reuse, MTP shares top-k, streaming top-2048 | 6-8 | top-k equals a torch reference; drafted equals serial across the boundary; needles at 32K-256K |
| M6 | prefill: EXL3 prefill GEMMs, exact reduce-scatter + gather, half-chunk overlap | 5-6 | any chunking gives the same bits; ≥ 700 tok/s at 8-32K (stretch 1,500) |
| M7 | long context: fp8 latent (quality A/B), then 4-rank context parallel + sharded indexer | 10-12 | 1M needle; MemAvailable > 8 GB throughout |
| M8 | upstream: tiny synthetic `glm_moe_dsa` checkpoint tests (four ranks on one GPU), recipe doc, measurements | 3-4 | CI green without weights; pins and hashes recorded |

## Risks

1. Collective latency: ~160 serial all-gathers per step over RoCE; at 100 µs each decode loses its margin. Measure
   first (M0).
2. Prefill with the fp32 all-gather caps near 850 tok/s; beating vLLM needs the exact reduce-scatter.
3. No full-precision reference fits four Sparks: quality is judged against the vLLM serve of the same checkpoint,
   per-layer fp32 references, and the offline BF16 KLD.
4. Unified memory: 70 GB weights + 12.8 GB reserve leaves thin margins.
5. Unverified details: `indexer_types` layout and whether shared layers ship indexer weights; RoPE scaling at 1M.
6. The universal EXL3 experts kernel at 512-wide per-rank experts and 8-bit MTP is unbenchmarked (our vLLM port
   measured 398 µs per layer at 3 rows).
7. The checkpoint must be public for `MODELS`; license check.

## Questions for the maintainer

- A 4-rank engine and `--tp 4`: generalize core (comm, split, capacity, follower loop) to any world size, or {1, 2, 4}?
- New `glm_moe_dsa` family with shared MLA/DSA kernels, or extend `glm5_next`?
- EXL3-only first (no MLX/NVFP4 path)?
- Is the exact reduce-scatter acceptable under the exactness contract, and can `comm.py` gain send/recv?
- fp8 latent cache and context parallel: bits change per configuration, not per rank — acceptable?
- Test fixtures without four Sparks: thread-comm on one GPU in CI plus a real-hardware protocol run by us?
- MTP-only drafting for this family (no DFlash2 model exists)?
