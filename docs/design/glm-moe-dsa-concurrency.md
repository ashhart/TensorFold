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

## Phase B status (branch conc-phase-b, 2026-10-02)

Built: steps 3 (decode windows up to 32 rows) and 5 (DFlash2 per stream).
- Windows > 16 rows: `fused.Buffers(decode=True)` (the Runner's and GlmMultiDecoder's verify / MTP buffers) treats every
  window it serves as a decode window up to `DECODE_ROWS` = 32: RoCE one-shot (or all-gather + rank-order sum) reductions,
  ATTN_DECODE tiling, head / argmax buffers of the full width. Prompt buffers keep the old rule (<= FAST_ROWS 16), so
  prompt bits and `--parallel 1` are unchanged. Measured row-invariant at 32 rows (a row's hidden / logits / argmax in a
  32-row window == in 8-row and 1-row windows, below and past index_topk; MTP layer too) - no sub-window split needed.
  32 x 6144 fp32 = 768 KB fits RoCE MAX_BYTES (1 MiB); the RoCE startup check now also compares a 32-row reduce to the
  NCCL rank-order sum.
- `dflash.MultiDrafter` (port of MiaAI 0027): one GlmDrafter's weights, a 4096-slot ring per stream in one pool
  [kv, N * RING + trash, hd], every DFlash2 stream's block in one pass (per-segment attention and dynamic convolution),
  every stream's kept taps in one context update; graphs per stream count and per tap-row bucket. Drafts equal the solo
  drafter's (1 and 3 streams, tested).
- `GlmMultiDecoder`: per-request mode (serial / MTP / "dflash" / "auto", via `mtp_mode`), per-stream depth = min(block - 1,
  DECODE_ROWS / N - 1, DFLASH_CFG depth, room) and the chain's confidence cut per stream; auto keeps a per-stream EMA arm
  choice (one-stream rule) and its MTP backlog of up to 8 rows (written every round, drafting or not); prompt chunks feed
  the stream's drafter ring. Drafts are computed on every rank (identical gathered candidates), so ROUND stays one message.
  A lone DFlash2 stream takes exactly the one-stream path's rounds (tested).
- Prewarm adds verify windows of every width 1..32 (argmax, every key bucket); other shapes capture on first use.
Tests: tests/cuda/test_glm_moe_dsa_multi_dflash.py (drafter == solo; 2/3/4 streams DFlash2 + auto + MTP mixed,
greedy + seeded sampled, staggered, one prompt past index_topk, windows of 32 rows; graphs; engine parallel=3).
Deferred: DCP with several streams, paged caches, measuring the 32-row RoCE cost on the cluster.
