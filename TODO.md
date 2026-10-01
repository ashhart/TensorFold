# DeepSeek-V4.1-Flash EXL3 on 2× DGX Spark — TensorFold port

Branch `dsv41-cuda` on top of upstream `ashhart/TensorFold` 0.5.0 (remote `upstream`).
Target: `Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw` (mul1, 196 GiB, 39 shards) + Engram shards 47/48 of
`deepseek-ai/DeepSeek-V4.1-Flash`, TP=2 over CX7 on `aiai` (rank 0, 10.42.0.1) and `aiai2` (rank 1).

Baseline to beat (vLLM recipe on the same pair, DSpark k=3): decode 31.6 tok/s ×1, 23 tok/s serial,
aggregate 113.7 at ×6; prefill ~1,000 tok/s to 128k. Per-token weight floor ≈ 3.7 GB/rank → ~17 ms,
so serial ≈ 45–55 tok/s is the ceiling.

Clean-room rule: the vLLM recipe overlay and `display_kv.c` are **AGPL-3.0**. Read vLLM (Apache-2.0)
and DeepSeek's reference (MIT) freely; do not copy recipe overlay code into this MIT tree.

## Phase 0 — groundwork

- [x] Clone upstream, branch `dsv41-cuda`, this TODO
- [x] NVIDIA container `nvcr.io/nvidia/pytorch:26.07-py3` on both nodes; install TensorFold; build kernels on sm_121
  (dev container `tf-dev` on aiai: repo at /tf, models at /models, Engram at /engram-src)
- [x] Toolchain smoke: `python -m tensorfold.cuda.exl3.inspect` — all 47,900 groups readable (`notes/dsv41/inspect.txt`)
- [x] Toolchain smoke: `tests/cuda/test_exl3_*` on GB10 — 114 passed, 54 skipped (need other checkpoints)
- [x] **DRM scanout carveout allocator, clean-room rewrite** (`tensorfold/cuda/carveout.py`, pure ctypes, MIT)
  - [x] DRM dumb buffer create/map, `cudaHostRegister(DEVICEMAP)`, torch tensor view, process-lifetime owner
  - [x] Opt-in via `TF_CARVEOUT=1`, size `TF_CARVEOUT_BYTES` (default 1792 MiB), card `TF_DRM_CARD`
  - [x] Probe CLI: `python -m tensorfold.cuda.carveout probe [--cuda]`
  - [x] Verify on GB10 with vLLM stopped: 1792 MiB, 0 MiB host RAM, round trip ok, 59 vs 122 GB/s (`notes/dsv41/BENCH.md`)
  - [x] Carveout holds the V4.1 compressed-KV pools (`TF_CARVEOUT=1`, largest source first; indexer keys stay in
        ordinary memory): prefill speed and long parity unchanged, admission credits the 1.75 GiB
- [x] Capacity: host reserve configurable (`TF_HOST_RESERVE_GIB`; default stays max(4 GiB, 10 %)),
      carveout bytes counted as room outside MemAvailable
- [ ] Container flags doc: `--device /dev/dri/card0`, `nvidia_drm modeset=1 fbdev=0`, no display in use

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
      (found with vLLM activation dumps: private recipe copy `/home/docker/ai/vllm-serve/tf-dump`, eager mode,
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
- [ ] Shared expert folded into the grouped expert call
- [x] **Go/no-go**: serial 34–35 tok/s vs vLLM 23 at matching quality (target ≥ 40 still open)

## Phase 3 — drafting, long context, prefill

- [x] DSpark on CUDA (`cuda/dspark.py`): taps = entry streams of layers 37–39 (V4.1 semantics), Markov head,
      draft + (N+1)-row verify graphs. Acceptance equals vLLM's on the same prompts; ~1.3× vLLM tok/s (BENCH.md)
- [x] Exact verify: bit-identical to one-row decode (Triton router matmul + Engram gate); DSpark == serial tokens
- [x] Adaptive draft length (`DraftPolicy`, k = 0..N by expected tokens/ms): 1.39–1.49× vLLM on the 3 cases
- [x] Rank-deterministic draft policy (rank 0 decides k each round; ranks chose different k from own clocks → deadlock)
- [x] DSpark == serial tokens greedy and sampled (temperature 0.8) on all cases
- [x] Round costs from real tokens (capture zeros understated multi-row verify); policy picks k by true costs
- [ ] Verify cost: split Engram reads across ranks (skew shows up as NCCL wait); fold shared expert into grouped call
- [ ] CUDA graphs per window width; eager == graph checks
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
- [ ] Move GLM-5 / Nemotron-H snapshots onto `kv_pool`
- [x] Concurrent decoding (`--parallel N`): stream slots, stream-aware decode graphs, MultiDecoder + shared Scheduler,
      cost-based draft allocation; 16 clients: 115.5 tok/s aggregate (vLLM recipe baseline 113.7 at x6); outputs ==
      sequential
- [x] Up to 32 rows a round + 15 MB decode rings a slot: 32 clients 143.3 tok/s aggregate
- [x] Batched DSpark drafting across streams (8 clients 82.5 -> 88.9 tok/s)
- [x] `--context` admission before any cache exists (both ranks' free memory; refuses with the largest that fits;
      default 40,960 shrinks to fit); expandable-segments allocator (prompt transients 3.6 -> 1.6 GiB at 38K)
- [x] Structured output: `response_format` json_schema / json_object, guided_choice / regex / grammar (xgrammar),
      masks on verify rows (DSpark drafts cut to the grammar's prefix), rank 1 compiles the same grammar
- [x] Compose project in place of the vLLM recipe: `deploy/dsv41-tp2` (image with deps baked in, `make swap-in` / `swap-out`, API key)

## Ops notes

- vLLM service: `/home/docker/ai/vllm-serve/deepseek41flash-exl3-TP2` (`make`/`recipe/start.sh stop|start`).
  OK to stop for GPU work; restart when done.
- Unified memory: a GB10 OOM once wedged both nodes (2026-09-11). Cap side jobs with
  `systemd-run --scope -p MemoryMax=…`; `--oom-score-adj 1000` on our containers.
