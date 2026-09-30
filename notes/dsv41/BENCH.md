# DeepSeek-V4.1-Flash EXL3 — measured building blocks on GB10

## 2026-09-30 — display carveout (`python -m tensorfold.cuda.carveout probe [--cuda]`)

aiai, vLLM stopped, `nvcr.io/nvidia/pytorch:26.07-py3`, `--device /dev/dri/card0`:

| | |
|---|---|
| DRM only, 1792 MiB | write/read ok, MemAvailable moved 15 MiB |
| CUDA, 1792 MiB | GPU round trip ok, host RAM spent 0 MiB |
| copy into / out of carveout | 59 / 58 GB/s |
| ordinary → ordinary | 122 GB/s |
| 2048 MiB | `ENOMEM` from `DRM_IOCTL_MODE_CREATE_DUMB` (carveout ceiling between 1.75 and 2 GiB) |
| 1792 MiB while vLLM holds its carveout | `ENOMEM` (one owner per node) |

## 2026-09-30 — EXL3 weight streaming, one rank of TP=2 (`tools/dsv41_bench_layer.py`)

aiai rank 0, layers 3,9,15,21,27,33 (3-bit experts), CUDA graph of 6 layers back to back, fresh random
expert picks (6 of 384 per row) every replay. Split: routed and shared experts along the expert width
(1152 per rank), attention by heads (wq_b N/2, 4 of 8 wo_a groups, wo_b K/2), wq_a and wkv replicated.
Weights only — no attention math, norms, router, hyper-connections, Engram or collectives.

| Rows | attention (42 MB) | shared expert | routed experts | all | head (248 MB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 233 µs | 60 µs | 192 µs | **482 µs/layer** | 994 µs |
| 4 | 235 µs | 60 µs | 712 µs | 987 µs/layer | 1023 µs |
| 8 | 247 µs | 63 µs | 1344 µs | 1630 µs/layer | 1044 µs |

- One routed expert is 6.64 MB per rank (3-bit); 6 per row → 40 MB in 192 µs = 207 GB/s. Attention 180 GB/s.
- Serial floor: 40 × 482 µs + 1.0 ms ≈ **20.3 ms/token ≈ 49 tok/s** before attention math,
  collectives (~80 all-reduces) and Engram. vLLM serial today: 23 tok/s.
- 4-row verify window with random picks (worst case, up to 24 distinct experts): 40.5 ms/round; at vLLM's
  measured 3.19 tokens/round that bounds drafted decode near 79 tok/s (real windows share experts, so better).
- Routed experts dominate multi-row cost: expert overlap across draft rows is the lever for DSpark rounds.

## 2026-09-30 — serial TP=2 engine, decode progression (365-token prompt, greedy, `tools/dsv41_run2.sh`)

| step | decode tok/s | note |
|---|---:|---|
| eager PyTorch | 7.3 | 81 NCCL waits ~20 ms (ranks drift), ~3k tiny Sinkhorn/elementwise kernels |
| + fused HC Triton kernels + one-row CUDA graph | 18.4 | GPU 38 ms/step |
| + fused MQA attention, table RoPE, RMSNorm kernels | 19.7 | GPU 32 ms/step; host ~15 ms/token (Engram reads) |
| + concurrent Engram preads, dequant on GPU | **26.9** | cold row fetch 20–25 ms → 2–7 ms |

vLLM on the same pair: 23 tok/s serial, 31.6 with DSpark k=3. GPU/step now ~32 ms: EXL3 weights ~22,
NCCL 2.6 (81 × 32 µs), HC 1.6, attention 1.1, rot_in 0.6, rest small. Parity vs reference 95.1% top-1.
