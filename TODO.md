# DeepSeek-V4.1-Flash EXL3 on 2× DGX Spark — TensorFold port

Branch `dsv41-cuda` on top of upstream `ashhart/TensorFold` 0.6.2 (remote `upstream`; rebased 2026-10-02).
Target: `Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw` (mul1, 196 GiB, 39 shards) + Engram shards 47/48 of
`deepseek-ai/DeepSeek-V4.1-Flash`, TP=2 over CX7 on `aiai` (rank 0, 10.42.0.1) and `aiai2` (rank 1).

Baseline to beat (vLLM recipe on the same pair, DSpark k=3): decode 31.6 tok/s ×1, 23 tok/s serial,
aggregate 113.7 at ×6; prefill ~710 tok/s measured at 2K–32K (2026-10-01; ~1,000 quoted to 128k).
Per-token weight floor ≈ 3.7 GB/rank → ~17 ms, so serial ≈ 45–55 tok/s is the ceiling.
Where we are (2026-10-05, compose 16 × 614400, fp4 KV, carveout): serial ~37.8 tok/s, C1 98.0 (fp8 105.1; ~4% behind
since CUDA decode attention), hard prose 37.0, 16 clients 123.5 tok/s aggregate (fp8 128.0), PP8192 1,508,
100K prompt 2,286 tok/s; shared pool 6.52M tokens (fp8 2.38M).

Clean-room rule: the MiaAI-Lab vLLM recipe overlay and coolbho3k's `display_kv.c` (DeepSeek-v4.1-Flash-2x-DGX-Spark)
are **AGPL-3.0**. Read vLLM (Apache-2.0) and DeepSeek's reference (MIT) freely; take only ideas from AGPL code and
copy none of it into this Apache-2.0 tree.

## Phase 0 — groundwork

- [x] Clone upstream, branch `dsv41-cuda`, this TODO
- [x] NVIDIA container `nvcr.io/nvidia/pytorch:26.07-py3` on both nodes; install TensorFold; build kernels on sm_121
  (dev container `tf-dev` on aiai: repo at /tf, models at /models, Engram at /engram-src)
- [x] Toolchain smoke: `python -m tensorfold.cuda.exl3.inspect` — all 47,900 groups readable (`notes/dsv41/inspect.txt`)
- [x] Toolchain smoke: `tests/cuda/test_exl3_*` on GB10 — 114 passed, 54 skipped (need other checkpoints)
- [x] **DRM scanout carveout allocator, independent re-implementation of coolbho3k's display_kv idea**
  (`tensorfold/cuda/carveout.py`, pure ctypes, Apache-2.0, no AGPL code)
  - [x] DRM dumb buffer create/map, `cudaHostRegister(DEVICEMAP)`, torch tensor view, process-lifetime owner
  - [x] Opt-in via `TF_CARVEOUT=1`, size `TF_CARVEOUT_BYTES` (default 1792 MiB), card `TF_DRM_CARD`
  - [x] Probe CLI: `python -m tensorfold.cuda.carveout probe [--cuda]`
  - [x] Verify on GB10 with vLLM stopped: 1792 MiB, 0 MiB host RAM, round trip ok, 59 vs 122 GB/s (`notes/dsv41/BENCH.md`)
  - [x] Carveout holds the V4.1 compressed-KV pools (`TF_CARVEOUT=1`, largest source first; indexer keys stay in
        ordinary memory): prefill speed and long parity unchanged, admission credits the 1.75 GiB
- [x] Capacity: host reserve configurable (upstream 0.6.2: `TENSORFOLD_MEMORY_RESERVE_GIB`, >= 2 GiB; default max(4 GiB, 10 %)),
      carveout bytes counted as room outside MemAvailable
- [ ] Container flags doc: `--device /dev/dri/card0`, `nvidia_drm modeset=1 fbdev=0`, no display in use
      (`make up` already refuses with modeset off; the doc is still missing)

## Phase 1 — family skeleton + reference

- [x] `families/deepseek_v41`: MODEL_TYPES, config parsing (text_config), EXL3 check, `cuda_engine` stub
- [x] Architecture spec from vLLM reference: `notes/dsv41/ARCH.md` (open questions in §13)
- [x] Weight-streaming floor per rank: 482 µs/layer, ~49 tok/s serial ceiling (`tools/dsv41_bench_layer.py`, BENCH.md)
- [ ] Checkpoint map: EXL3 groups vs plain tensors per layer; per-rank split plan (experts by id, attention by head)
- [ ] Per-rank weight budget (must stay ≤ ~99.5 GiB/rank like vLLM)
- [x] Engram hashing/token map/bucket layout (`families/deepseek_v41/engram.py`; multipliers, primes, 99,092 ids verified)
- [x] Single-GPU layer-streaming reference forward, T ≤ 512 (`families/deepseek_v41/reference.py`)
- [x] Goldens: vLLM `prompt_logprobs` for 7 prompts (`tools/dsv41_golden.py`) → `notes/dsv41/golden.json`
- [x] Reference vs goldens: 94.2% top-1 / NLL 1.400 vs 1.383 on 365 tokens after dropping q per-head RMS
      (found with vLLM activation dumps: a private vLLM recipe checkout, eager mode,
      hook in `overlay/patch_memory_log.py`; `tools/dsv41_dump_diff.py`). vLLM itself is noisy on short
      prompts (eager vs CUDA graphs differ by up to 0.6 mean |Δlogprob|).
- [x] Chat template: the checkpoint's `chat_template.jinja` (V4.1 encoder port; DSML tool calls, reasoning_effort)

## Spec findings that change the plan (ARCH.md)

- Not an encoder/decoder: all layers causal. Only layers 2, 8, 14 (ratio 2) and 20 (ratio 1) own long-range KV;
  the rest read their source's cache. ~1.8 KB/token long-range KV (fp8) → 600k context ≈ 1.1 GB. KV is cheap.
- Top-k computed once per index source (2, 8, 14, 20, 24, 28, 32, 36) and reused by following layers.
- Below 512 tokens nothing is dropped: indexer/candidates can be deferred for first parity.
- Engram tables are 101 GB/layer fp8 in the original shards; 24 random 256 B rows/token/layer.
- The served checkpoint is the in-place abliterated variant (`ABLIT_META.json`, wo_b of layers 10–35).

## Phase 2 — serial TP=2 engine (go/no-go)

- [x] Embedding, hyper-connections (hc_mult 4, 20 Sinkhorn iters) on CUDA
- [x] Attention: q/kv low-rank, 64 heads × 512 shared KV, window 128 + sinks, RoPE/YaRN, grouped low-rank output
- [x] CSA2 compressors (ratios 2 / 1 per layer), compressed pools, cross-layer KV sharing (`kv_source_layer_ids`)
- [x] Indexer (32×128, top-512, bf16 keys) shared by `index_source_layer_ids`; rings for window/raw caches;
      long-context parity to 16K tokens (NLL equal to vLLM; BENCH.md)
- [x] Candidate blocks (layer 20, 2048×8): parity to 40K tokens (NLL 1.614 vs 1.613)
- [x] Position-keyed sampling (`tensorfold.cuda.sampling.sample_rows`), DSpark acceptance against keyed samples
- [x] MoE: sqrt-softplus router, noaux_tc top-6 of 384, shared expert; routed via `cuda/exl3/experts`
- [x] Engram layers 1/14: n-gram hashing, FP8 e4m3 rows, pread row store (page cache / NVMe)
- [ ] Mapped-table accounting in capacity
- [x] 2-rank split + NCCL rank-order reduction; lm_head (6-bit)
- [x] First serial TP=2 engine (`cuda/serial.py`, `tools/dsv41_serial_run.py`): eager PyTorch, 96.1 GiB/rank,
      365-token prefill 94.5% top-1 vs reference (NLL 1.420 vs 1.400), coherent greedy text; decode 7.3 tok/s,
      prefill 237 tok/s (2026-09-30)
- [ ] Engine vs reference agreement should be ~99%: layer-diff the engine against reference dumps
- [x] Fused HC pre/post Triton kernels (`cuda/hc.py`, tests/cuda/test_dsv41_hc.py) + one-row decode CUDA graph:
      decode 7.3 → 18.4 tok/s; profile 38 ms GPU/step: weights ~22, small torch ops ~7, NCCL 2.7, HC 1.7, host ~5
- [x] Fused MQA attention / table RoPE / RMSNorm kernels (`cuda/kernels.py`, tests/cuda/test_dsv41_kernels.py)
- [x] Concurrent Engram preads + GPU dequant: decode **26.9 tok/s** serial (vLLM serial 23) — BENCH.md
- [x] Native Engram reader (pthreads pread), 3 decode graphs with overlapped reads, argmax in graph,
      rank-order partial sums fused into HC post, fused router (fp16 mm → fp32, no TF32), attention chunk skip,
      Engram wkv split: **33.5 tok/s** serial (vLLM 23 serial / 31.6 DSpark)
- [x] Small linears in parallel on side streams (q / window KV / compressor, 8 wo_a slices, shared expert)
- [x] Shared expert folded into the grouped expert call (measured slower; off by default)
- [x] **Go/no-go**: serial 34–35 tok/s vs vLLM 23 at matching quality (target ≥ 40 still open)

## Phase 3 — drafting, long context, prefill

- [x] DSpark on CUDA (`cuda/dspark.py`): taps = entry streams of layers 37–39 (V4.1 semantics), Markov head,
      draft + (N+1)-row verify graphs. Acceptance equals vLLM's on the same prompts; ~1.3× vLLM tok/s (BENCH.md)
- [x] Exact verify: bit-identical to one-row decode (Triton router matmul + Engram gate); DSpark == serial tokens
- [x] Adaptive draft length (`DraftPolicy`, k = 0..N by expected tokens/ms): 1.39–1.49× vLLM on the 3 cases
- [x] Rank-deterministic draft policy (rank 0 decides k each round; ranks chose different k from own clocks → deadlock)
- [x] DSpark == serial tokens greedy and sampled (temperature 0.8) on all cases
- [x] Round costs from real tokens (capture zeros understated multi-row verify); policy picks k by true costs
- [x] Split Engram reads across ranks (bit-identical; 16 clients 114.9 -> 117.1 tok/s)
- [x] Shared expert folded into the grouped call: implemented, measured slower (35.0 vs 36.0 serial), off (`TF_FOLD_SHARED=1`)
- [ ] Serial decode >= 40 tok/s (now ~37.8 with x3ld + fast top-k + RoCE, GPU-bound 96 %): EXL3 decode GEMV toward ~240 GB/s (now ~215 large,
      110-180 small, 199 routed), fuse input rotation into the linear kernel, HC pre/post into one, router + route
- [x] Decode graphs at narrow key widths on one graph memory pool (`TF_DSV41_WIDTHS`, default one 64K width;
      4.05 GiB for 32 graphs -> 0.27 GiB for 128): 16 clients at 16 × 614400 101.2 -> 122.5 tok/s (2026-10-03)
- [ ] Eager == graph checks per width; more narrow widths once graph driver memory allows (see Open)
- [x] Prefill 330 → 486 tok/s: 1,024-row chunks, EXL3 prompt GEMM for dense linears, prompt grouped expert kernel
      (`cuda/experts_prompt.cu`), bf16 prompt partials
- [x] Expert prompt kernel v2 (smem-staged activations, 2 member tiles/decode, 8 warps), 2,048-row chunks,
      last-row-only prompt logits: prefill **622 tok/s**
- [x] Prefill 1,252 tok/s steady (whole prompts 1,376 at 8K, 1,160 at 32K) vs vLLM ~710 (BENCH.md)
- [x] Long prompts to 600k: 4 sessions x 614,400 tokens admitted (fp8 caches), a 592,960-token prompt answered
- [x] KV format ~1.8 KB/token (fp8); carveout-backed pools
- [x] `--parallel N` shared rounds (see Phase 4)

## Phase 4 — serving

- [x] `cuda/engine.py` `Dsv41Engine`: `tensorfold serve --tp 2` on both nodes (rank 1 `follow()`s rank 0's
      requests: header + prompt over NCCL, rank 0's stop on the per-round agreement), warm-up before serving
- [x] OpenAI chat/completions through `cuda/server.App`: reasoning split, streaming, stop strings, seeded sampling,
      `draft: false` == drafted output, client disconnect stops within a round
- [x] DSML tool calls: V4.1 writes `<｜DSML｜ calls>` (space, no `tool_`); server + CUDA reply parsers read both forms
- [x] Launcher `tools/dsv41_serve2.sh [--port P] [--context N]` (dual-link NCCL env, Engram dir)
- [x] Prompt reuse: the live caches serve a prompt that extends them (common prefix >= 64 tokens, ring-safe);
      a follow-up turn of 8.5K tokens: 8,503 cached, 0.6 s instead of 8.0 s
- [x] Shared prompt-state pool `tensorfold/cuda/kv_pool.py` (model-agnostic: LCP match, LRU in a byte budget, rank-
      deterministic) + V4.1 adapter (`save_prefix` / `load_prefix`: per-position caches + rings' last window); budget
      from free memory (7 GiB ~ 2.35M tokens at context 40,960); switching conversations resumes in < 1 s, retries too
- [ ] Move GLM-5 / Nemotron-H snapshots onto `kv_pool` (see Open)
- [x] Concurrent decoding (`--parallel N`): stream slots, stream-aware decode graphs, MultiDecoder + shared Scheduler,
      cost-based draft allocation; 16 clients: 115.5 tok/s aggregate (vLLM recipe baseline 113.7 at x6); outputs ==
      sequential
- [x] Up to 32 rows a round + 15 MB decode rings a slot: 32 clients 143.3 tok/s aggregate
- [x] Batched DSpark drafting across streams (8 clients 82.5 -> 88.9 tok/s)
- [x] DSpark up to 5 drafts (reasoning 71.6, code 57.6 tok/s single stream)
- [x] Copy drafts, single stream (edit requests 80 -> 117 tok/s; after MiaAI-Lab's GLM recipe); [ ] in concurrent rounds
- [x] RoCE all-gather (`tensorfold.cuda.rdma`, from b12x RoCEnante via MiaAI-Lab 0006; exact): serial +1.7%;
      default since 28e8205 (see below); [ ] two rails
- [x] `--context` admission before any cache exists (both ranks' free memory; refuses with the largest that fits;
      default 40,960 shrinks to fit); expandable-segments allocator (prompt transients 3.6 -> 1.6 GiB at 38K)
- [x] Structured output: `response_format` json_schema / json_object, guided_choice / regex / grammar (xgrammar),
      masks on verify rows (DSpark drafts cut to the grammar's prefix), rank 1 compiles the same grammar
- [x] Compose project in place of the vLLM recipe: `deploy/dsv41-tp2` (image with deps baked in, `make swap-in` / `swap-out`, API key)
- [x] vLLM-compatible `/tokenize`; `/v1/completions` and `/tokenize` take token-id prompts
- [x] Draft allocator looks ahead (a stream's next 1..cap drafts at once): recipe bench C3 57 -> 112, C4 93 -> 132 tok/s
- [x] Hard prose at 4 × 614K 30.6 -> 33.2 tok/s: indexer scores only visible keys, Engram cached rows inline
      (`preadv2 RWF_NOWAIT`), hot pool helpers, no readahead on the tables

## Phase 5 — shared KV pool and kept prompts (2026-10-02 .. 10-03)

- [x] Shared KV pool (`--parallel > 1`): every stream's compressed entries and indexer keys in one arena (first-fit
      2048-aligned extents, design after MiaAI-Lab's GLM recipe), per-slot base table in the decode graphs, rank 0
      sends placements in ADMIT, NoRoom holds requests; concurrent == sequential
- [x] Kept prompts in the pool's free rows (extents outlive their stream, window rings in a bank of
      `TF_DSV41_KEPT_ENTRIES`, now 32); resume by takeover or copy; LRU eviction of whole extents (EVICT ops);
      the copy-out PrefixPool is off with the shared pool; 24-request suite == a no-keep server
- [x] Growth and yields: admission reserves prompt + `TF_DSV41_GROW_AHEAD` (4096); extents grow / move / evict kept
      states before each round (GROW/MOVE/EVICT ahead of ROUND); a stuck stream waits, the newest background stream
      yields and replays exactly; foreground requests ask `yield_for`; outputs == an unconstrained server
- [x] Decode graphs' driver memory (~40 MB a graph outside the allocator) charged in the pool size
      (`GRAPH_EXEC_BYTES`), `TF_DSV41_MEMLOG=1` per startup stage: pool 2.03M -> 3.08M tokens at 16 × 614400 (fp8)
      with 3.4-3.6 GiB left
- [x] Release after a long prompt only below `TF_DSV41_RELEASE_BELOW_GIB` (2.5) available (handing ~1 GiB back per
      prompt fragmented memory: 8K prompts 650-1000 -> 1550-1830 tok/s); glibc heap trimmed
- [x] Chunk-invariant prefill (`Linear.prompt_mode`, no chunk <= 32 rows): rows bit-identical whatever the chunking
      (chunk-test tails 1..1060, chunks 33..2048); kept prompts resume at any position their window covers
- [x] Exact bounded tail (layers 21-39 only over a prompt's last `tail_min` rows; idea from jayleaton's CED replay,
      exact form ours; `TF_DSV41_BOUNDED_TAIL=0` off): prefill 8K 1283 -> 1676, 16K 1279 -> 1995,
      32K 1249 -> 2165 tok/s; tail-test at 12K/30K/39K bit-equal
- [x] THP guards (`NUMPY_MADVISE_HUGEPAGE=0`, `MIMALLOC_ALLOW_THP=0`) in the compose environment
- [x] Defragmented starts: `make up` drops caches + `vm.compact_memory` on both nodes (fragmented host > 35 min load
      vs ~100 s clean; engine ready 59 s instead of ~88 s); `[boot]` timeline line
- [x] NVMe kept tier (`TF_DSV41_DISK`, `kvdisk.py` after Jay Leaton's sessdisk, MIT): evicted kept states spilled
      both ranks in step, restored on admission (32K: ~45 MB read in fp4 instead of ~30 s prefill), PERSIST on
      SIGTERM, key intersection at start; `KV_DISK_DIR` / `STOP_GRACE_S` 150 s in compose

## Phase 6 — speed, boot, serving hardening (2026-10-04)

- [x] Fast top-k: bounded radix-select indexer top-k for decode/verify rows (`topk.py`, after Jay Leaton's dtopk,
      our code; `TF_DSV41_FAST_TOPK=0` off): 16 clients 116.8 -> 123.0, serial 36.2 -> 36.8 tok/s; same entries
- [x] x3ld routed-expert load path (Jay Leaton's x3ld.cu, MIT; bit-identical Z), default on (4,2): 16 clients
      122.8 -> 128.2, 4 clients 79.8 -> 81.7, serial 36.9 -> 37.8 tok/s; bulk L2 prefetch sites opt-in
      (`TF_L2_PREFETCH=bulk`)
- [x] Fast boot: prepared per-rank weight folders (`TF_DSV41_PREPARED`, O_DIRECT parallel readers, per-chunk SHA-256,
      `make prepare`), cached Engram token map and calibration, lighter warm-up with `--parallel`
- [x] `make prebuild` compiles CUDA extensions (and the RoCE proxy, x3ld) before any model loads
      (compiling beside a loaded 4 × 600K model OOM-killed the server)
- [x] DSML parser: one lenient parser for streamed and whole replies (unclosed think, V4 spellings, cut invokes,
      `parallel_tool_calls=false`; adapted from Jay Leaton's dsml.py, MIT)
- [x] Tool grammar (`TF_DSV41_TOOL_GRAMMAR=required`, on in compose): required / named / strict tools as xgrammar's
      deepseek_v4_1 structural tag; streamed == whole, required == auto when auto calls first, alone == beside 3 streams
- [x] Health: `/health` reports fatal / stalled (`TF_STALL_S`), `TF_HEALTH=strict` answers 503
- [x] Watchdog: systemd user timer (`watchdog.sh`, `make watch-install`), lease / lock / stop marker, idle 1-token
      probe, heals with `WATCH_HEAL=1` (default 0: alert only)
- [x] Soak / stress / structured harnesses (`tools/dsv41_soak.py`, `dsv41_stress.py`, `dsv41_structured.py`,
      `make soak|stress|structured`, key only from the environment); `tools/dsv41_clients.py` N clients × T s
- [x] RoCE all-gathers by default (`TF_COMM=nccl` opts out; both ranks vote, any failure keeps NCCL): hard prose
      35.8 -> 37.7, C1 97.2 -> 97.7, C4 149.5 -> 154.0, 16 clients 127.3 -> 128.0 tok/s
- [x] Attribution (2740427): THIRD_PARTY_NOTICES.md, NOTICE and headers credit Jay Leaton, MiaAI-Lab GLM recipe,
      b12x RoCEnante, coolbho3k (ideas only), vLLM, DeepSeek, ExLlamaV3, xgrammar (FlashInfer added with mqa_fp4)
- [x] Draft PR ashhart/TensorFold#342 (head urtho:dsv41-cuda, draft) opened 2026-10-04; **on hold**: no pushes, no
      ready-for-review, no comments until the user pushes (e1baa98 and later are local only)

## Phase 7 — FP4 KV (2026-10-04 .. 10-05)

- [x] Torch port of V4.1's KV quantizers (NVFP4 / MXFP4 / FP8 ue8m0) against a NumPy oracle
- [x] `QRows` byte planes; `Fp4Rows` (NVFP4 entries 288 B, MXFP4 keys 68 B) + RoPE/quantize kernel, byte-equal on 1M rows
- [x] Attention and indexer read packed fp4 bit-equal to their bf16 dequant; prompt top-k ties to the lower index
- [x] `TF_DSV41_KV=bf16|fp8|fp4` (+ `IQ_FP4`, `SWA_FP8`, `COMP_BF16`); ranks agree on the format and knobs;
      fp8 1845 B a token, fp4 890
- [x] Invariant tools `--views-test`, `--resume-test`, `--tf-compare`, `--needle`, `--decode-bench`: all pass in fp4
- [x] Tie-keyed prompt top-k in one Triton pass (`_tie_pick`): fp4 prefill 100K 2085 vs fp8 2160, 128K 1993 vs 2133,
      600K 1270 vs 1343-1467 tok/s (was 256K 788 vs 1921)
- [x] Quality: teacher-forced vs bf16 over 24.5K positions: fp4 top-1 96.9%, dNLL +0.0016; fp8 97.8%, +0.0003;
      needles at 128K and 590K pass
- [x] **fp4 default** (11418ba, 2026-10-04): pool at 16 × 614400 2.38M -> 5.93M tokens (6.52M on the compose server);
      A/B fp4 / fp8: C1 98.0 / 105.1, hard 37.0 / 38.4, PP8192 1508 / 1596, 100K 2286 / 2428, 16 clients 123.5
- [x] CUDA decode attention for fp4 (`mqa_fp4.cu`, after FlashInfer's Cake DSv4.1 decode, Apache-2.0; default on,
      `TF_DSV41_CUDA_MQA=0` Triton): attention a step 0.33-0.36 ms (Triton fp4 1.56-1.60, fp8 0.86-0.90) at 4K-600K;
      verify R=4 0.51-0.57 ms; teacher-forced top-1 96.86%, dNLL +0.0018 (within noise)
- [x] Shared-tile fp4 indexer (`_index_scores_tile`, `TF_DSV41_SHARED_IK`, default on): MXFP4 key tile decoded once
      per stream for all its rows, tiles past visible keys skipped; torch.equal to the per-row kernel

## Open (2026-10-05)

- [x] RoCE GID index robustness (70abae0): no fixed index; NCCL (`NCCL_IB_ADDR_FAMILY=AF_INET`, RoCE v2, 10.42.0.0/15)
      and the RoCE all-gather find the IPv4 RoCE v2 GID per device (a reboot under a live QP moved aiai2's to 4)
- [x] FP4 prefill gap (67d6b36, bit-identical): row-tiled segment scores, one tie-keyed pick a pass, prompt entries
      decoded once a chunk: PP8192 1649 (fp8 1596), 100K 2581 (fp8 2428); dev 500K 1762 (fp8 1619)
- [ ] fp4 indexer with many streams at full context: unmeasured (shared-tile kernel measured on unit cases only)
- [ ] C1 ~4% behind fp8 in fp4
- [ ] FP4 tensor-core indexer: parked (not bit-exact against the reference order)
- [x] Dev loop (notes/dsv41/DEV.md): one-load suites and tiers (`tools/dsv41_suite2.sh quick`: 168 s vs 499 s as
      fresh runs), prepared weights in dev runs (22-24 s loads), OOM guard (torch cap + per-run memwatch, start
      refusal), `make hot` / `make cold` (119 s, no image build)
- [ ] Slow first start after `make image` (C1 81, PP 1029, 100K 1276 on the first fp4 start; no compaction stalls):
      cause unknown
- [ ] Review-flagged test gaps: indexer equality test only at n_keys 2900; mqa_fp4 R=1 masking
- [ ] FP4 quality checks not run: paired MMLU, tool-call eval, 24+ prompts
- [ ] Serial decode >= 40 tok/s (now ~37.8; see Phase 3)
- [ ] Copy drafts in concurrent rounds
- [ ] RoCE two rails
- [ ] Triton kernels compiled in `make prebuild` (now compiled at first start)
- [ ] Exact-chunk TTFT mode
- [ ] More narrow decode widths (each width's graphs cost ~40 MB a graph of driver memory)
- [ ] L2 prefetch sweep (`TF_L2_PREFETCH=bulk` sites measured only together)
- [ ] Engine vs reference layer-diff (94.5% -> ~99% agreement)
- [ ] Eager == graph per width
- [ ] Mapped-table accounting in capacity
- [ ] Checkpoint map / per-rank weight budget (Phase 1)
- [ ] Container flags doc for the carveout (Phase 0)
- [ ] `kv_pool` for GLM-5 / Nemotron-H
- [ ] Housekeeping: agent worktrees (`attribution`, `item4-kvdisk`, `item5-serving` under `.claude/worktrees`),
      stray `uv.lock`, private path in `tools/dsv41_vllm_dump_patch.py`
- [ ] Watchdog: decide `WATCH_HEAL=1` (cluster heal tests in deploy README) and user linger
- [x] Rank 1 idled at ~95% GPU util spinning in the collective: CPU idle doorbell (2248e36, `TF_IDLE_DOORBELL`):
      0-1% / ~11 W idle, outputs and latency unchanged
- [x] Server MemAvailable after warm-up was 2.8-2.9 GiB: TF_DSV41_RESERVE_GIB 2.5 -> 3 (3.86 / 3.91 GiB; pool 6.52M -> 6.41M)
- [ ] Old `tf-dev-old-*` containers on both nodes can go (keep the `tf-dev-snapshot:*` images: tf-dev runs on them)

## Ops notes

- aiai / aiai2 are dev boxes until the port is done: `make down` / `up` / `restart` / `image` of the compose server
  (/home/docker/ai/vllm-serve/tensorfold-dsv41-TP2 on aiai; the worker gets the directory via `make sync`) is
  normal work. One GPU job at a time; `make down` before dev runs.
- GB10 memory: fragmented host memory makes loads and prefill slow (compact before start; watch `compact_stall`);
  each decode graph costs ~40 MB of driver memory outside torch's allocator (check MemAvailable, not reserved).
- vLLM service: a private vLLM recipe checkout (`make`/`recipe/start.sh stop|start`).
  OK to stop for GPU work; restart when done.
- Unified memory: a GB10 OOM once wedged both nodes (2026-09-11). Cap side jobs with
  `systemd-run --scope -p MemoryMax=…`; `--oom-score-adj 1000` on our containers.
