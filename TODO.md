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
- [ ] NVIDIA container `nvcr.io/nvidia/pytorch:26.07-py3` on both nodes; install TensorFold; build kernels on sm_121
- [ ] Toolchain smoke: `python -m tensorfold.cuda.exl3.inspect` on the V4.1 checkpoint (headers only)
- [ ] Toolchain smoke: `tests/cuda/test_exl3_*` on GB10 (needs vLLM stopped)
- [x] **DRM scanout carveout allocator, clean-room rewrite** (`tensorfold/cuda/carveout.py`, pure ctypes, MIT)
  - [x] DRM dumb buffer create/map, `cudaHostRegister(DEVICEMAP)`, torch tensor view, process-lifetime owner
  - [x] Opt-in via `TF_CARVEOUT=1`, size `TF_CARVEOUT_BYTES` (default 1792 MiB), card `TF_DRM_CARD`
  - [x] Probe CLI: `python -m tensorfold.cuda.carveout probe [--cuda]`
  - [ ] Verify on GB10 with vLLM stopped: MemAvailable unchanged, GPU round trip, bandwidth vs cudaMalloc
  - [ ] Hand the carveout to the V4.1 compressed-KV pools only (59 vs 124 GB/s — keep hot buffers out)
- [x] Capacity: host reserve configurable (`TF_HOST_RESERVE_GIB`; default stays max(4 GiB, 10 %)),
      carveout bytes counted as room outside MemAvailable
- [ ] Container flags doc: `--device /dev/dri/card0`, `nvidia_drm modeset=1 fbdev=0`, no display in use

## Phase 1 — family skeleton + reference

- [x] `families/deepseek_v41`: MODEL_TYPES, config parsing (text_config), EXL3 check, `cuda_engine` stub
- [ ] Checkpoint map: EXL3 groups vs plain tensors per layer; per-rank split plan (experts by id, attention by head)
- [ ] Per-rank weight budget (must stay ≤ ~99.5 GiB/rank like vLLM)
- [ ] Reference activations: dump per-layer hidden states from the vLLM deployment for a few prompts
      (teacher-forced) → golden files for layer-by-layer parity
- [ ] Chat template / tokenizer: DeepSeek V4.1 encoder (`deepseek_v41` template, DSML tool calls, reasoning_effort)

## Phase 2 — serial TP=2 engine (go/no-go)

- [ ] Embedding, hyper-connections (hc_mult 4, 20 Sinkhorn iters) on CUDA
- [ ] Attention: q/kv low-rank, 64 heads × 512 shared KV, window 128 + sinks, RoPE/YaRN, grouped low-rank output
- [ ] CSA2 compressors (ratios 2 / 1 per layer), compressed pools, cross-layer KV sharing (`kv_source_layer_ids`)
- [ ] Indexer (32×128, top-512) shared by `index_source_layer_ids`; candidate blocks (layer 20, 2048×8)
- [ ] MoE: sqrt-softplus router, noaux_tc top-6 of 384, shared expert; routed via `cuda/exl3/experts`
- [ ] Engram layers 1/14: n-gram hashing, FP8 e4m3 rows (never uint8), file-backed row store (O_DIRECT/mmap),
      mapped-table accounting in capacity
- [ ] 2-rank split + NCCL rank-order reduction; lm_head (6-bit)
- [ ] **Go/no-go**: serial decode must clearly beat 23 tok/s (target ≥ 40) at matching quality

## Phase 3 — drafting, long context, prefill

- [ ] DSpark on CUDA (checkpoint `mtp.*`, 128 experts top-3, targets 37–39, block 5), exact verify windows
- [ ] CUDA graphs per window width; eager == graph checks
- [ ] Prefill kernels (EXL3 chunked prefill), long prompts to 600k
- [ ] KV format ≤ ~3.4 KiB/token; carveout-backed pools
- [ ] `--parallel N` shared rounds (optional; vLLM wins at width today)

## Ops notes

- vLLM service: `/home/docker/ai/vllm-serve/deepseek41flash-exl3-TP2` (`make`/`recipe/start.sh stop|start`).
  OK to stop for GPU work; restart when done.
- Unified memory: a GB10 OOM once wedged both nodes (2026-09-11). Cap side jobs with
  `systemd-run --scope -p MemoryMax=…`; `--oom-score-adj 1000` on our containers.
