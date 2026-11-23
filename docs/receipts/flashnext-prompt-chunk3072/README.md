# Flash Next CUDA: a prompt pass holds 3,072 rows by default

A Flash Next prompt pass held 1,536 rows at most while another stream decoded (`cuda_batch_state.zig`'s `default_chunk`), so a long prompt ran in passes of 1,536 and each lane admitted 1,536 rows a pass. It holds 3,072 now: about half the passes for a long prompt, each holding twice the rows, and the same 3,072 rows a lane's prompt admission. A server started at one stream is untouched: it takes 8,192-row passes on both sides (`cuda_chunk.zig`'s solo rule). `TF_FLASHNEXT_CHUNK` still overrides the default, 1 to 16,384.

Result: cold prompts 6.4% to 9.8% faster to the first token idle and 12.2% to 17.2% beside three decoders (2k to 64k), `bench_cache`'s first cold request 7.2% faster; decode level at 1 to 8 streams; the one-stream control level; every exactness contract equal on both arms and every greedy reply equal across them.

## Qualified scope

TensorFold main `077998f7d9b14adebec3a2acc2caa47ff7ae6ec7` against main plus this change, commit `052e72a9c3eb6025b0f6a6095f06194fdc9a3313`, both built from source with Zig 0.17.0 and nvcc 13.3 in the `nvcr.io/nvidia/pytorch:26.07-py3` image (`zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast fatbins install native`). One DGX Spark (GB10, sm_121, memory shared with the host), kernel driver 580.178.04. Checkpoint `nvidia/Qwen3.8-Flash-Next-FP4-Local` at `b9ce18a6887e18c089fdea8509a33e63429a3300`.

Server per arm, through the serving recipe (`CHECKPOINT=nvfp4 KV_DTYPE=bf16 TENSORFOLD_CUDA_KERNELS=native VISION=0`); the container command line on the decode boots was:

```
tensorfold-native serve CHECKPOINT --name Qwen3.8-Flash-Next --host 127.0.0.1 --port 8080 --context 32768 --parallel 8 --max-tokens 32768 --temperature 1.0 --top-p 0.95 --top-k 20 --thinking --kv-cache bf16
```

with `--context 131072 --parallel 4` and `--prompt-cache-gib 0` for the cold prompts, `--parallel 1` for the control and `--parallel 8` for the pool check at that window. The recipe exports these variables into the container, the same set on every boot of both arms (each boot's own environment was recorded and compared):

```
TENSORFOLD_CUDA_KERNELS=native
TENSORFOLD_IMAGE_TOKENS=16384
TENSORFOLD_MAX_IMAGES=50
TENSORFOLD_MEMORY_RESERVE_GIB=9
TENSORFOLD_NO_UPDATE_CHECK=1
TENSORFOLD_VIDEO_TOKENS=16384
TENSORFOLD_VISION_PYTHONPATH=/opt/tensorfold/python
TENSORFOLD_VISION_WORKSPACE_MIB=2048
TF_FLASHNEXT_CONFIDENCE=-0.4
TF_FLASHNEXT_DRAFTS=6
TF_FLASHNEXT_DRAFT_AHEAD=1
TF_FLASHNEXT_GATHER_EARLY=1
TF_FLASHNEXT_GATHER_READ=pread
TF_FLASHNEXT_GDN_LANES=1
TF_FLASHNEXT_HEAD_PIECE=1536
TF_FLASHNEXT_HOST_DRAW_THREADS=7
TF_FLASHNEXT_KEEP_LATE=1
TF_FLASHNEXT_PREFILL_TAIL=512
TF_FLASHNEXT_PRODUCT_STREAMS=5
TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS=256
TF_FLASHNEXT_PROMPT_GEMM2=1
TF_FLASHNEXT_PROMPT_HC2=1
TF_FLASHNEXT_PROMPT_SHARED_DIRECT=1
TF_FLASHNEXT_SAMPLED_HEADS=1
TF_FLASHNEXT_SAMPLE_CAND=1
TF_FLASHNEXT_SHARED_SIDE=1
TF_FLASHNEXT_WARM_AHEAD=1
```

12 of them are read by neither tree (`TF_FLASHNEXT_DRAFT_AHEAD`, `TF_FLASHNEXT_GATHER_EARLY`, `TF_FLASHNEXT_GATHER_READ`, `TF_FLASHNEXT_HEAD_PIECE`, `TF_FLASHNEXT_HOST_DRAW_THREADS`, `TF_FLASHNEXT_PREFILL_TAIL`, `TF_FLASHNEXT_PRODUCT_STREAMS`, `TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS`, `TF_FLASHNEXT_PROMPT_GEMM2`, `TF_FLASHNEXT_PROMPT_HC2`, `TF_FLASHNEXT_PROMPT_SHARED_DIRECT`, `TF_FLASHNEXT_SAMPLE_CAND`), so they change nothing on either arm; the rest are read by both. The two arms differ in nothing but the image, and `TF_FLASHNEXT_CHUNK` is set on neither, so each serves its own default.

Order: `M1 B1 B2 M2`, a fresh server each, in every timed pass. A cell is slower when both branch runs are under both main runs (for a latency: over) and the mean gap is over 1%, faster when both are over both and the gap is over 1%, level otherwise; a flagged cell would have been re-measured once on fresh boots. No cell was flagged.

## Correctness

- B cache on concurrent: concurrent equals solo 156/0 (unverified 0); drafted equals `"draft": false` 56/0
- B cache on golden: concurrent equals solo 13/0 (unverified 0); drafted equals `"draft": false` 24/0
- B cache on drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B cache on turns: resumed from cached prompt tokens [13, 16, 16, 13, 24, 18, 3406]
- B cache off drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- B cache off fresh: 7 of 7 resumed replies equal the fresh ones
- M cache on concurrent: concurrent equals solo 156/0 (unverified 0); drafted equals `"draft": false` 56/0
- M cache on golden: concurrent equals solo 13/0 (unverified 0); drafted equals `"draft": false` 24/0
- M cache on drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- M cache on turns: resumed from cached prompt tokens [13, 16, 16, 13, 24, 18, 3406]
- M cache off drafts: 8 of 8 equal to `"draft": false` (basis token_ids)
- M cache off fresh: 7 of 7 resumed replies equal the fresh ones
- B vs M: replies equal over pass A's greedy cells (token_sha): 528 equal, 0 unequal

- the cold cells' token ids and reply text, the same prompt bytes sent to every arm: 8 of 8 cells at four slots and 4 of 4 at one slot equal on all four boots
- drafts off: 42 replies equal between the arms, 0 unequal

## Prompt speed

Cold prompts, `--prompt-cache-gib 0`, a 131,072-token window at four slots, so a 64k prompt is admitted beside three decoders (`load3`). Time to the first token, seconds, the median of three salted prompts, thinking off (`ttft.py --sizes 2k,8k,32k` and `--sizes 64k`, `--loads idle,load3 --reps 3 --max-tokens 16`), and `bench_cache.py --tokens 64 --repeats 2 --streams 2 --require-cold`'s first request.

| prompt | load | M1 | M2 | B1 | B2 | B vs M |
|---|---|---|---|---|---|---|
| cold2k-off | idle | 1.070 | 1.071 | 0.972 | 0.960 | -9.8% faster |
| cold2k-off | load3 | 1.273 | 1.262 | 1.053 | 1.046 | -17.2% faster |
| cold8k-off | idle | 4.035 | 3.986 | 3.664 | 3.685 | -8.4% faster |
| cold8k-off | load3 | 4.653 | 4.632 | 3.989 | 3.987 | -14.1% faster |
| cold32k-off | idle | 16.015 | 16.044 | 14.980 | 15.039 | -6.4% faster |
| cold32k-off | load3 | 18.249 | 18.420 | 16.114 | 16.094 | -12.2% faster |
| cold64k-off | idle | 33.058 | 33.175 | 30.665 | 30.682 | -7.4% faster |
| cold64k-off | load3 | 37.582 | 37.718 | 32.780 | 32.786 | -12.9% faster |
| first_ttft_s | idle | 1.241 | 1.221 | 1.142 | 1.143 | -7.2% faster |

Cold: 9 faster, 0 level, 0 slower, 0 unmeasured.

## The one-stream control

The same cells at one slot (idle only: a second stream cannot decode beside the measured prompt), where both arms take 8,192-row passes:

| prompt | load | M1 | M2 | B1 | B2 | B vs M |
|---|---|---|---|---|---|---|
| cold2k-off | idle | 0.979 | 0.968 | 0.969 | 0.966 | -0.6% level |
| cold8k-off | idle | 3.458 | 3.434 | 3.453 | 3.414 | -0.4% level |
| cold32k-off | idle | 14.144 | 14.150 | 14.124 | 14.100 | -0.2% level |
| cold64k-off | idle | 29.025 | 28.967 | 28.896 | 28.984 | -0.2% level |
| first_ttft_s | idle | 1.148 | 1.150 | 1.143 | 1.141 | -0.6% level |

Control: 0 faster, 5 level, 0 slower, 0 unmeasured.

## Decode speed

`bench_openai.py --tokens 256 --reps 3 --temperatures 0` (one stream) and `bench_concurrent.py --tokens 256 --reps 3 --levels 1,2,4,6,8 --alone --serial --temperatures 0`, prompt cache on, 32,768-token window, eight slots; aggregate tokens a second, the median of three reps.

| prompt | streams | M1 | M2 | B1 | B2 | B vs M | tokens a round, M / B |
|---|---|---|---|---|---|---|---|
| chat | 1 | 46.9 | 46.2 | 46.5 | 46.3 | -0.3% level | 2.81 / 2.81 |
| chat | 2 | 86.8 | 86.2 | 86.3 | 86.3 | -0.2% level | 2.51 / 2.51 |
| chat | 4 | 156.6 | 155.1 | 155.7 | 155.7 | -0.1% level | 2.51 / 2.51 |
| chat | 6 | 211.0 | 209.2 | 209.8 | 209.5 | -0.2% level | 2.51 / 2.51 |
| chat | 8 | 258.1 | 256.1 | 256.9 | 256.5 | -0.2% level | 2.51 / 2.51 |
| code | 1 | 73.1 | 71.9 | 72.3 | 72.5 | -0.1% level | 5.22 / 5.22 |
| code | 2 | 138.8 | 136.9 | 138.0 | 137.5 | -0.1% level | 4.83 / 4.83 |
| code | 4 | 244.9 | 240.8 | 242.7 | 241.6 | -0.3% level | 4.83 / 4.83 |
| code | 6 | 310.5 | 306.5 | 307.7 | 306.2 | -0.5% level | 4.49 / 4.49 |
| code | 8 | 374.5 | 370.5 | 370.8 | 371.7 | -0.3% level | 4.49 / 4.49 |
| fibonacci-raw | 1 | 73.4 | 72.2 | 72.7 | 72.6 | -0.2% level | 5.22 / 5.22 |
| gpu-chat-no-think | 1 | 47.0 | 46.3 | 46.5 | 46.5 | -0.3% level | 2.81 / 2.81 |

Decode: 0 faster, 12 level, 0 slower, 0 unmeasured. The server's acceptance ratio is identical on every boot (0.6242, 0.6884).

## Memory

The pass's device scratch at the default grows with the rows (`Scratch.bytes(default_chunk, .{ .head_rows = default_chunk })`: 2,753,048,104 bytes at 3,072 rows against 1,397,558,248 at 1,536). The engine reserves its scratch at load, before it reads its room, so each arm's own startup line carries the cost:

| window / slots | arm | model GiB | a stream | streams at once | GiB left after the 9.0 GiB reserve | pass rows |
|---|---|---|---|---|---|---|
| 32768 / 8 | main | 72.71 | 1.29 GiB each | 8 | 27.9 | 1536 |
| 32768 / 8 | branch | 73.50 | 1.32 GiB each | 8 | 27.1 | 3072 |
| 131072 / 4 | main | 72.73 | 4.11 GiB each | 4 | 27.9 | 1536 |
| 131072 / 4 | branch | 73.52 | 4.14 GiB each | 4 | 27.0 | 3072 |
| 131072 / 1 | main | 76.16 | 4.23 GiB each | 1 | 24.3 | 8192 |
| 131072 / 1 | branch | 76.16 | 4.23 GiB each | 1 | 24.4 | 8192 |
| 131072 / 8 | main | - | 8 need 32.86 GiB | refused, names 6 | 27.85 | - |
| 131072 / 8 | branch | - | 8 need 33.10 GiB | refused, names 6 | 27.01 | - |

At 131,072 tokens and eight slots both arms refuse `--parallel 8` by name and name six streams; at every other setting both admit the same streams.

The lines verbatim:

- 32768/8 main: CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 72.71 GiB; 8 streams at once, 1.29 GiB each at a 32768-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, a 16.0 GiB prompt cache; own glue kernels
- 32768/8 branch: CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 73.50 GiB; 8 streams at once, 1.32 GiB each at a 32768-token window, of 27.1 GiB left after a 9.0 GiB reserve; prompts in passes of up to 3072 rows, a 16.0 GiB prompt cache; own glue kernels
- 131072/4 main: CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 72.73 GiB; 4 streams at once, 4.11 GiB each at a 131072-token window, of 27.9 GiB left after a 9.0 GiB reserve; prompts in passes of up to 1536 rows, no prompt cache; own glue kernels
- 131072/4 branch: CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 73.52 GiB; 4 streams at once, 4.14 GiB each at a 131072-token window, of 27.0 GiB left after a 9.0 GiB reserve; prompts in passes of up to 3072 rows, no prompt cache; own glue kernels
- 131072/1 main: CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 76.16 GiB; 1 stream at once, 4.23 GiB each at a 131072-token window, of 24.3 GiB left after a 9.0 GiB reserve; prompts in passes of up to 8192 rows, no prompt cache; own glue kernels
- 131072/1 branch: CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 76.16 GiB; 1 stream at once, 4.23 GiB each at a 131072-token window, of 24.4 GiB left after a 9.0 GiB reserve; prompts in passes of up to 8192 rows, no prompt cache; own glue kernels
- 131072/8 main: refused at start: --parallel 8 needs 32.86 GiB for its streams at a 131072-token window, and the CUDA memory budget leaves 27.85 GiB: serve --parallel 6, or lower --context
- 131072/8 branch: refused at start: --parallel 8 needs 33.10 GiB for its streams at a 131072-token window, and the CUDA memory budget leaves 27.01 GiB: serve --parallel 6, or lower --context
