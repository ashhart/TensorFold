# glm_moe_dsa: multi-sequence concurrency (N <= 8) — design notes (2026-10-01)

Today: single sequence at every layer (server Turns mutex, scalar device POS, per-layer caches without a sequence dim,
one DFlash2 ring, graph keys without stream identity). Reuse: cuda/scheduler.py, cuda/streams.py,
families/qwen3_5/cuda/multi.py (TP follower ADMIT/FILL/ROUND/DONE protocol over _share, interleaved STEP prefill).

## Change set, dependency order
1. Per-row addressing in fused kernels: POS[1] -> pos[R] + per-row slot/cache base (stacked [N, local, 576] or page
   table): _kv_write, _ik_write, _absorb, _qrope, _iq_rope, _attn_chunks, _attn_dcp, _index_scores, select (T = max
   bucket; per-row cnt + DCP candidate gathers). HARDEST (DCP).
2. State per stream: N slots of kc/ic/mkc/mic (limit/N or paged), per-stream carry + MTP backlog; admission by memory.
3. Exactness > 16 rows: FAST_ROWS / head buffers -> ~64, RoCE MAX_BYTES 1 MiB -> >=1.5 MiB, keep ATTN_DECODE tiling +
   rank-order sum for windows <=128 rows (bench_concurrent --alone requires token_sha == solo). HARD (RoCE size/latency).
4. Batched rounds in Runner: pack per-stream [tok+drafts] (8x3=24 MTP, 8x8=64 DFlash rows); graph key (R total, T
   bucket, pick) + per-row pos/slot table copied before replay; per-stream acceptance via streams.accept.
5. DFlash2 per stream: ring [N, KV, RING, hd] + pos_dev[N] (or snapshot/restore) + batched propose.
6. GlmMultiDecoder (live/admit/round/finish/drop/follow) + ADMIT/FILL/ROUND/DONE follower protocol; chunked prefill
   steps as separate forwards between decode rounds.
7. Engine/server: concurrent=True, generate -> Scheduler.submit, follow -> multi.follow, pass `parallel` through
   cuda_engine (__init__.py:38). Capture/stats per stream.

Memory: BF16 latent ~97 KB/token/rank (DCP1) -> ~35 GB of cache ~= 360K tokens TOTAL across streams (e.g. 8 x 32K OK);
NVFP4 KV (vLLM keys36 writer + b12x reader, ~30 KB/token) is the later lever for many long streams.
Source map with file:line refs: subagent report 10-01 (summarized in ~/.claude memory project_tensorfold_glm53_native).

## Phase A status (branch conc-phase-a, 2026-10-01)

Built: steps 1 (per-row tables, DCP 1 only), 2 (N fixed cache slots of `--context` tokens), 4 (MTP windows of up to
16 rows: N x (k + 1) <= FAST_ROWS, refused otherwise), 6 (`multi.GlmMultiDecoder`, one-shot messages ADMIT / FILL /
ROUND / DONE) and 7 (`--parallel N` -> Scheduler; followers run `multi.follow`). The fused kernels take a `ROWS`
constexpr (off: the old code exactly) with int32 position and cache-base tables; prompt fills run the one-stream
kernels on the stream's slot views with the chunking a lone request uses (prompt chunks are not row-invariant).
Deferred: step 3 (> 16 rows), step 5 (DFlash2 per stream; not loaded with --parallel > 1), DCP with several streams,
paged / shared-prefix caches, admission by measured memory (slots are allocated up front).
Tests: tests/cuda/test_glm_moe_dsa_multi.py (2, 3, 4 streams, staggered, greedy + seeded sampled, one prompt past
index_topk: each reply equals its lone reply; graphs replay; engine + Scheduler; memory flat over admit/finish).

## Fills between layers (branch conc-interleave, 2026-10-02)

Head-of-line blocking: a fill ran a whole prompt chunk (up to 4096 rows, ~5 s on 78 layers) between decode rounds.
While other streams decode, a chunk now goes TF_GLM53_FILL_LAYERS layers a step (default 8; 0: whole chunks) with a
decode round between steps (`Runner.prefill_chunk(layers=(lo, hi))`, `fused.compute` / `compute_prompt(layers=...)`;
FILL carries the layer range). Same chunk boundaries, rows, layers and kernels, only paused: the chunk's rows wait in
the prompt buffers (`pb`, `pb1`, which decode rounds never use; decode has the decoder's `vb` / `mb`), the two
micro-batches' reductions in flight are waited for by the main stream at the pause (decode collectives start after
them), the fill writes only its slot. A prompt with no stream decoding takes whole chunks, exactly as alone.
Tests: test_fill_between_layers_{four_ranks,graphs} (chunks of 512 rows, 1 or 2 layers a step, overlapped
micro-batches on the 4-rank run): every reply equals its lone reply; decode rounds ran inside paused chunks.
