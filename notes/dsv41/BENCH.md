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
| + split decode graphs (layer 0 / layers 1–13 / 14–39+head), Engram reads overlapped, native row reader | 29.9 | |
| + rank-order sums fused in HC post, fused router, attention chunk skip, split Engram wkv | 32.3 | |
| + fp16 router matmul with fp32 output (no TF32; parity 95.6%, mean \|Δlogprob\| 0.079) | 32.4 | |
| + layer-1 Engram read hidden behind layer 0 | **33.5** (157 tokens, story prompt) | ~29.8 ms/token |

Remaining per token (~29.8 ms): EXL3 weights ~20.5 ms (floor), NCCL ~2 ms (83 calls), HC 1.6 ms
(2 MB fp32 mix matrix per sublayer), attention 0.8, rot_in 0.6, small kernels ~1.5, host ~1–2 ms.
Next levers are structural (shared expert folded into the grouped expert call, batched small linears).

## 2026-09-30 — head-to-head vs vLLM (same prompt ids, greedy, 200-token cap; `tools/dsv41_vllm_accept.py` + `--cases`)

| prompt | vLLM DSpark k=3 (incl. ~0.1 s prefill) | TensorFold serial | TensorFold DSpark k=3 | ratio |
|---|---:|---:|---:|---:|
| story (creative) | 25.8 tok/s, 0.94 acc/round | 30.9 | **33.7**, 1.03 acc/round | 1.31× |
| reasoning (chat, thinking) | 42.0, 2.53 | 33.2 | **56.7**, 2.38 | 1.35× |
| code (chat, thinking) | 32.6, 1.53 | 33.1 | **42.9**, 1.74 | 1.32× |

- DSpark acceptance matches vLLM's on the same prompts; vLLM's production 2.19 average reflects predictable traffic.
- TensorFold's DSpark round costs ~2× a serial step (reasoning: 3.38 tokens/round at 56.7 tok/s = 60 ms/round).
- Outputs differ from vLLM's (kernel numerics, fp8 KV) and DSpark vs serial differ slightly: multi-row verify is not
  yet bit-identical to one-row decode (suspect cuBLAS router matmul algorithm by row count).

## 2026-09-30 — exact verification + adaptive draft length

Multi-row verify is now bit-identical to one-row decode (`tools/dsv41_exact.py`: every row of every window
equal, max |Δlogit| 0; DSpark token streams == serial on all cases). Culprits were row-count-dependent reductions:
cuBLAS router matmul → Triton `router_logits`; torch `.sum/.mean` in the Engram gate → Triton `engram_gate`.

| prompt | vLLM DSpark | TF serial | TF DSpark k=3 | TF DSpark adaptive (k histogram 0..3) |
|---|---:|---:|---:|---:|
| story | 25.8 | 33.6 | 35.1 | 36.0 ([18, 3, 26, 4]) |
| reasoning | 42.0 | 33.6 | 62.1 | 60.0 ([1, 1, 19, 31]) |
| code | 32.6 | 33.2 | 48.5 | 48.4 ([1, 1, 28, 51]) |

Round: draft ~5 ms, 4-row verify ~48 ms (GPU 46.7: routed experts 20.5 ms for ~16 distinct experts/layer,
NCCL 6.2 ms incl. rank skew from per-rank Engram reads).

## 2026-09-30 — long context: indexer top-512, rings (`--long-golden notes/dsv41/golden_long.json`, cap 16384)

| prompt | top-1 vs vLLM | NLL ours / vLLM | prefill |
|---:|---:|---:|---:|
| 1,024 | 97.8% | 1.241 / 1.231 | 227 tok/s |
| 2,048 | 95.9% | 1.567 / 1.560 | 330 tok/s |
| 4,096 | 96.0% | 1.625 / 1.622 | 331 tok/s |
| 8,192 | 96.2% | 1.676 / 1.675 | 329 tok/s |
| 16,000 | 95.8% | 1.660 / 1.660 | 315 tok/s |

Document: upstream README + four recipes (`notes/dsv41/long_doc.txt`). Index keys stored bf16 (vLLM: fp8).
Context limit 16,384 until layer 20's candidate blocks are implemented. Prefill ~330 tok/s vs vLLM ~1,000.

## 2026-09-30 — candidate blocks: parity to 40K (`golden_long_all.json`, cap 40960, 1024-row chunks)

| prompt | top-1 vs vLLM | NLL ours / vLLM | prefill |
|---:|---:|---:|---:|
| 1,024 | 97.8% | 1.241 / 1.231 | 102 tok/s (cold Engram) |
| 8,192 | 96.2% | 1.676 / 1.675 | 350 tok/s |
| 24,576 | 95.8% | 1.631 / 1.630 | 362 tok/s |
| 40,000 | 95.7% | 1.614 / 1.613 | 369 tok/s |

vLLM decode at 40K context (prefix-cached prompt): ~19 tok/s with DSpark (1.68 accepted/round).
1024-row chunks moved prefill only 330 → 369 tok/s: expert weight reads are not the prefill bottleneck.

## 2026-09-30 — prefill (one 1,024-row chunk at ~2K context, `--profile-prefill 1024`)

| step | chunk wall | tok/s |
|---|---:|---:|
| 128-row chunks (before) | — | ~330 |
| 1,024-row chunks, dense linears in 128-row decode pieces | 2,762 ms | 371 |
| + dense linears through the EXL3 prompt GEMM (W decoded once a call) | 2,426 ms | 422 |
| + expert member tables sized to the busiest expert | 2,283 ms | 449 |
| + prompt grouped expert kernel (one decode, 2 member tiles) + bf16 prompt partials over NCCL | 2,107 ms | **486** |

Long parity unchanged (8K: NLL 1.676 vs 1.675; 24K: 1.631 vs 1.630), prefill 450–490 tok/s on real prompts.
Remaining: the expert prompt kernel is 807 ms/chunk (~3× its weight-read floor): activations re-read per N block
and in-register decode; needs activations staged in shared memory / wider N tiles. vLLM prefills ~1,000 tok/s.

## 2026-09-30 — prompt expert kernel v2 + 2,048-row chunks (`--profile-prefill 2048`)

Kernel (layer 3, one rank's expert halves, 2,048 rows, `experts_prompt.cu`): upstream decode-kernel path n/a at this
size (grouping smem cap); v1 50.9 ms/layer; v2 (activations staged in shared memory per K slice, warps own N blocks)
45.4; **v2 + 2 member tiles a decode + 8 warps: 32.9 ms** (weight-read floor ≈ 12.6 ms).

| step (2,048-row chunk) | tok/s |
|---|---:|
| 1,024-row chunks, v1 kernel (before) | 486 |
| 2,048-row chunks, v2, torch-built member tables | 547 |
| + 2 member tiles, 8 warps, 3-deep weight prefetch | 588 |
| + prompt chunks compute only the last row's logits (no 2,048 × 129,280 gather) | **622** |

Parity unchanged (8K NLL 1.675 vs 1.675; 24K 1.631 vs 1.630). Remaining per chunk: experts 1.04 s, attention
0.36 s (per-row key loads), dense GEMMs 0.25 s, NCCL 0.19 s, expert epilogues 0.19 s, host ~0.4 s.

## 2026-10-01 — prefill past 1,000 tok/s (`--profile-prefill 2048`, steady-state 2,048-row chunks at 2–8K)

| step | tok/s |
|---|---:|
| previous (v2 expert kernel, 32-head MQA) | 785 |
| expert kernel v3: one K pass per program (Z written once) | 837 |
| v4: cp.async-staged activations, weight prefetch ring across slices, 16 warps | 888 |
| NCCL all-gather of the first row block overlapped with the second block's tail GEMMs | 893 |
| expert Z in scaled fp16 (acc / 64) + prompt epilogues (no fp32 per-member copy) | 939 |
| strided rot_in rows (no wo_a group copies), bf16 wo_b output | 946 |
| single-pass prompt MQA (no per-chunk fp32 partials), sink + inverse RoPE fused | **1,031** |
| + second PCIe half of the 200G link (`TF_DUAL=1`, NCCL 9.7 → 18.7 GB/s at 21 MB) | **1,058** |

Parity (`golden_long_all.json`): 1K NLL 1.245 / vLLM 1.231 (was 1.241), 8K 1.675 / 1.675, 24K 1.631 / 1.630,
40K 1.614 / 1.613. DSpark output == serial (greedy, 3 cases); decode unchanged with the dual link.
bf16 Z instead of fp16 cost 8K NLL +0.001 (rel 2.7e-3 vs 4.2e-4 on one layer), hence the scaled fp16.

Second link: `rocep1s0f0` (aiai) / `rocep1s0f1` (aiai2) needed IPv4 for a RoCE v2 GID; persistent in the
NetworkManager profile `cx7-companion-mtu` (10.43.0.1/24 on aiai `enp1s0f0np0`, 10.43.0.2/24 on aiai2
`enp1s0f1np1`, never-default, MTU 9000). vLLM stays pinned to the first half (`roceP2p1s0f0`/`...f1`).
Chunk profile now: experts 740 ms, dense GEMM 260, NCCL 164, MQA 143, _post 76, rot_in 58 + 57, unpack 45.

## 2026-10-01 — prefill past 1,250 tok/s (`TF_DUAL=1 --profile-prefill 2048`)

| step | tok/s |
|---|---:|
| start (dual link) | 1,058 |
| expert grid: an expert's n groups adjacent (activations shared in L2) | 1,119 |
| fused hc post + next pre for prompt chunks (streams not read back) | 1,144 |
| gate/up inputs rotated inside the expert kernel while staged (no rot_in copies; bit-identical) | 1,149–1,171 |
| busy experts' member groups adjacent (weights shared in L2; real routing: median 8, max ~950 rows) | 1,204 |
| expert configs (121 gate/up, 101 down) re-tuned on real routing | 1,230 |
| routed + shared added in the shared down GEMM epilogue, stored bf16 | **1,252–1,254** |

Parity (`golden_long_all.json`): 1K NLL 1.237 / vLLM 1.231, 8K 1.674 / 1.675, 24K 1.630 / 1.630,
40K 1.612 / 1.613. DSpark == serial. Chunk GPU busy 1.60 s: experts ~546 ms, dense GEMM 256, MQA 142,
NCCL ~115, hc post_pre ~100. Tried and dropped: folding both Hadamards into W for cuBLAS (slower), fused
rotation in the decode GEMV (neutral), 4 NCCL row blocks (neutral), caching fp16 dense W (7 GB, too little
headroom: ~7 GiB MemAvailable at cap 40960).

Decode (same day): MQA chunk 128 x 16 heads for <= 16 rows, side-stream parallel linears (q / window KV /
compressor, 8 wo_a slices, the shared expert beside the routed ones): 4-row verify 52 -> 47 ms, serial
31.5 -> 34-35 tok/s, DSpark reasoning 59 -> 63 tok/s. Experts at ~210 GB/s (at the DRAM limit). (A suspected ~12 ms of
host work a round turned out to be a mis-measured cost table; see below.)

## 2026-10-01 — final head-to-head vs vLLM (same document / prompts, both on the 2-node GB10 pair)

Prefill, whole prompt, fresh request (vLLM: `tools/dsv41_vllm_prefill.py`, random first token defeats the prefix
cache; TensorFold: `--prefill-bench`, dual link):

| prompt | vLLM | TensorFold | ratio |
|---:|---:|---:|---:|
| 2,048 | 715 | 768 | 1.07× |
| 8,192 | 711 | 1,376 | 1.94× |
| 16,000 | 715 | 1,349 | 1.89× |
| 32,000 | 707 | 1,160 | 1.64× |

Decode, DSpark k = 3, greedy, same prompt ids (vLLM incl. ~0.1 s prefill; outputs differ, so acceptance differs):

| prompt | vLLM | TensorFold | ratio |
|---|---:|---:|---:|
| story | 23.6 (0.79 acc) | 33.4–35 | ~1.45× |
| reasoning | 42.2 (2.43) | 62.5–64 | ~1.5× |
| code | 29.4 (1.25) | 48–52 | ~1.7× |
| serial (no drafts) | 23 (earlier) | 34–35 | ~1.5× |
| 40K context, DSpark | 19 (earlier) | 38 (earlier) | 2× |

## 2026-10-01 — decode round anatomy (`TF_ROUND_PROF=1`)

Host time a round is ~1 ms (agree 0.4, launches 0.4; Engram reads 2-3 ms each but overlapped with the graphs);
the GPU is busy the whole round: draft 4.6 ms + 4-row verify 45-47 ms. The "12 ms host overhead" was an artifact:
`round_costs` replayed the verify graphs with the capture's all-zero tokens, so every row routed to the same 6
experts and a 4-row verify looked like 33 ms instead of 47 (real rows reach ~17 distinct experts a layer). The
adaptive draft policy used those costs, so it over-chose large k.

Fixes: cost replays use the request's own recent tokens and run once before the decode clock (costs now
[29, 41, 46, 51] ms for k = 0..3, matching live replays); decode Engram reads with 16 threads instead of 64.
Result (DSpark k <= 3, greedy): story 35.9 tok/s (first case no longer pays the cost measurement), reasoning
62-63 -> 65, code ~49. DSpark == serial. The 6 ms a verified row is mostly the extra experts' weights (DRAM-bound).

## 2026-10-01 — segmented indexer top-k: contexts to the native 1,048,576

Prompt chunks no longer build the [rows, keys] fp32 score matrix: rows in passes of 512, keys in 16K segments, a
running top-512 merged per segment (layer 20's block maxima gathered per segment; later layers masked per segment).
Same entries as the full path (tests/cuda/test_dsv41_kernels.py). Admission: ~4.3 KB a context token (caches 3.2,
RoPE 0.8, block maxima 0.3) instead of ~28.5; `--context 1048576` admitted with ~7 GiB left idle.

| check | result |
|---|---|
| steady prefill (2,048-row chunks) | 1,255 tok/s (unchanged) |
| whole 32K prompt | 1,160 -> 1,321 tok/s |
| long parity 1K / 8K / 24K / 40K | NLL 1.237 / 1.674 / 1.630 / 1.612 (unchanged; vLLM 1.231 / 1.675 / 1.630 / 1.613) |
| 215,632-token prompt over HTTP (`--context 1048576`) | 214 s (1,009 tok/s), correct answer, >= 5.7 GiB free throughout |

Also: admission reads MemAvailable (CUDA's free figure on GB10 leaves out the reclaimable page cache full of weight
files right after loading, and read 0 on one rank).

## 2026-10-01 — concurrent decoding (`tensorfold serve --tp 2 --parallel N`)

Streams live in cache slots (stacked caches; prompt chunks run in a slot through views); decode graphs take rows of
any slots (each row's slot offsets its window ring, compressor ring, compressed entries, indexer keys and drafter
ring), so one forward verifies every stream's rows and the expert / dense weights a round reads serve them all.
Rows are row-invariant (<= 16 a round): every stream's tokens equal its sequential decoding (checked: 6 requests
concurrent == sequential). `cuda/multi.py` MultiDecoder behind the shared `tensorfold.cuda.scheduler.Scheduler`;
rank 0 sends admissions, prefill steps, per-stream draft counts and completions, rank 1 computes the same tokens.

Verify ms by rows (distinct tokens over slots): 1: 29, 4: 50, 8: 70, 12: 89, 16: 102. Independent tokens rarely
share experts (384, top-6), so a row costs ~6 ms whether it is a stream's token or a draft: drafts are allocated
per round by expected tokens per ms (per-stream acceptance, calibrated curve) and mostly vanish under load.

Sustained load (clients sending 256-token requests back to back, temperature 0.7, thinking off):

| setup | aggregate | per request |
|---|---:|---:|
| 1 client | 41 tok/s | 42 tok/s |
| `--parallel 4`, 4 clients (fixed 3 drafts) | 65 tok/s | 17 tok/s |
| `--parallel 8`, 8 clients (cost-based drafts) | 82.5 tok/s | 10.5 tok/s |
| `--parallel 16 --context 16384`, 16 clients | **115.5 tok/s** | 7.7 tok/s |

The decode Engram reads had run with 4 threads in the server (16 only in the runner): 16-stream rounds waited 16 ms
on the first table's rows; now 16 threads (4 a row for multi-row rounds, at most 64): 2.7 ms.
Next: more than 16 rows a round (row-invariant thresholds at 32), batched drafting for low concurrency.

## 2026-10-02 — carveout, 32-row rounds, small decode rings, batched drafting

- Display carveout (`TF_CARVEOUT=1`, 1,792 MiB, host RAM cost <= 90 MiB) holds the compressed-KV pools, largest
  source first; indexer keys stay in ordinary memory (scanned whole every decode step). Prefill 1,348 / 1,283 tok/s
  at 8K / 32K (unchanged), long parity unchanged.
- Decode rounds up to 32 rows (`TF_DSV41_DECODE_ROWS`, row-invariant kernels up to it; outputs == sequential).
  Verify ms by rows: 16: 98, 24: 132, 32: 154.
- A slot's rings: 230 MB -> 15 MB (decode rings of 256 rows; prompt chunks use one shared set of 4,096-row staging
  rings, the slot's 130-row window copied in before and out after each eager chunk). 32 slots at 16K context fit
  with 7.8 GiB left after capture.
- Batched DSpark drafting: one drafter pass for up to 10 drafting streams (graphs by stream count); the allocator
  charges the batched cost.
- Fixes found on the way: a kept draft equal to the end token did not end the stream (Stream.take checks only the
  last token): rounds are cut at it; fills resume past cached tokens (FILL carries rows); prompt fills are
  time-sliced while others decode (`TF_FILL_SHARE`); prompt snapshots are size-checked before copying.

| sustained load (256-token requests) | aggregate | per request |
|---|---:|---:|
| `--parallel 8`, 4 clients | 69.1 tok/s | 18.0 tok/s |
| `--parallel 8`, 8 clients | 88.9 tok/s | 11.9 tok/s |
| `--parallel 16 --context 16384`, 16 clients | 115–119 tok/s | 7.6 tok/s |
| `--parallel 32 --context 16384`, 32 clients | **143.3 tok/s** | 4.9 tok/s |

## 2026-10-02 — Engram read split, shared-expert fold, serial decode push

Serial (1-row) decode: 36.0 -> ~36 tok/s (27.6 ms a step); the GPU is busy 96 % of it (Nsight: 1.1 ms all-idle).
Exactness unchanged throughout (serial == DSpark == multi-stream, drafted == greedy, parity to 40K).

| change | result |
|---|---|
| Engram reads split across ranks (each half a token's rows, in-graph all-gather) | reads 3.7/3.4 -> 2.9/2.6 ms a round at 16 clients; 16 clients 114.9 -> 117.1 tok/s; bit-identical |
| shared expert folded into the grouped call (`TF_FOLD_SHARED=1`) | slower: 35.0 vs 36.0 tok/s serial, 113.8 vs 117.1 at 16 clients (its 5-bit blocks set the tail); off |
| shared expert removed entirely (timing only) | 36.0 -> 38 tok/s: the side stream already hides most of it, so a fold's ceiling is small |
| router logits split-K (8 slices) | 32.6 -> 20.5 us (router), 25.1 -> 7.9 us (indexer weights); ~+0.7 tok/s |
| decode step as one CUDA graph (pinned-flag waits for the rows) | back-to-back replays 28.7 -> 27.0 ms; real steps +0.3 tok/s (host sync already hid most switches) |
| persistent Engram read pool | 12 cached rows 342 -> 26 us (thread spawns); no serial change (reads were off the critical path) |
| L2 prefetch of next weights during all-gathers | slower (34.2 vs 35.3); off |
| NCCL size tuner (LL 1 row, Simple 2-4 rows) | standalone 2-4 row all-gathers 54/67/122 -> 23/27/29 us, but no change in the engine; removed |
| linear kernel at 2 blocks/SM (220 -> 128 registers) | no gain (wq_b 215 -> 202 GB/s); reverted |
| grouped expert tile settings | defaults already best (whole routed call 199 GB/s) |

Where the 27 ms goes: ~4.2 GB of weights a rank a step; plain reads reach ~247 GB/s on GB10, large EXL3 GEMVs
~215 GB/s, small ones (1.6-4 MB) 110-180 GB/s (fixed ~8-10 us each), the routed call 199 GB/s; 86 all-gathers at
~19 us pure latency (both GPUs equally fast). 40 tok/s (25 ms) needs the EXL3 decode kernels near ~240 GB/s plus
fusing the small kernels (input rotation, HC pre/post, router+route): a kernel rewrite, not a setting.

Follow-up (same day): sublayer timing (graph of one sublayer over layers 4-39, 1 row): attention 307-313 us a layer
(~42 MB: ~140 us above its bandwidth time), MoE 297-303 us (~53 MB: ~80 us above), HC pre 19 us (x2).
| change | result |
|---|---|
| wo_a slices as one grouped EXL3 launch (column groups in the linear kernel) | bit-identical; 35.6 -> 35.9 tok/s (noise level); kept |
| fused HC post+pre for decode rows | 34.5 vs 35.7 tok/s; off (`TF_FUSE_HC_DECODE`) |
| attention chunk 64 / 32 keys (20 / 40 programs for 1 row instead of 10) | no measurable change; default 128 (`TF_MQA_CHUNK`) |
| input rotation fused into the EXL3 linear kernel (`TF_EXL3_ROT_FUSE=1`) | bit-identical in unit tests; standalone +-1-2.6 us; no serial gain; a multi-row case fails ("invalid argument"), so off |
| L2 warming, corrected (GB10 ignores `prefetch.global.L2`: a 13 MB GEMV cold 129 us, after prefetch hints 110-124, after evict-last loads of every 32-byte sector 46-51, hot 44) | real warming but slower in decode: MoE weights during the attention all-gather 34.9, wo_a during the attention core 34.7, both 34.0, vs 36.1 tok/s; off |
| GPU-initiated RDMA (NVSHMEM 3.8 IBGDA) in place of NCCL's ~19 us proxied all-gathers | not possible on GB10: no DMA-BUF export, no nvidia_peermem, so IBGDA (and GPUDirect RDMA) cannot initialise |

## 2026-10-02 — DSpark 5 drafts, copy drafts

- DSpark up to 5 drafts a round (the checkpoint's `dspark_block_size`; was capped at 3): reasoning 65.5 -> 71.6 tok/s,
  code ~50 -> 57.6, story ~flat (35.1); DSpark == serial. Engine default now 5.
- Copy (prompt-lookup) drafts (`tensorfold/cuda/copy_drafts.py`, on by default, `TF_COPY_DRAFTS=0` off): when the last
  8 tokens occurred before, the tokens that followed are verified instead of DSpark's (up to 15, single stream; verify
  graphs to 16 rows). Edit-style requests, identical replies: rename a class in a 60-line file 79.9 -> 117.3 tok/s,
  add docstrings 76.5 -> 95.2. Standard cases: rarely fire, no change; copy == serial.
- Concurrent (`--parallel`) rounds do not use copy drafts yet.
