# GLM-5.3-Flash's FP8 latent cache on CUDA: prefetched 8-byte code loads, byte-identical and faster

Base: `5c82ef28a6` (main). Head: the base plus this pull request's kernel commit, which changes
`zig/kernels/cuda/common/mla.cuh` and the two kernels' register records (`commit.txt` has the build, toolchain and both
binaries' sha256; one binary a tree, the same file on both ranks). Test date: 2026-11-15. Platform: two DGX Spark
(GB10, sm_121, 128 GB shared memory) at TP=2 over a direct RoCE link, Ubuntu 24.04.5, driver 580.178.04, Zig 0.17.0,
nvcc 13.5, NCCL 2.30.7. Checkpoint: `nvidia/GLM-5.3-Flash-FP4-Local` at `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`,
every launch with `--kv-cache fp8`. Drafts are the default, `MTP + copies`.

## Scope

With `--kv-cache fp8`, `tf_glm_chunks_16_fp8` (decode windows) and `tf_glm_chunks_32_fp8` (prompt chunks) turn each
32-key tile of E4M3 codes back into the BF16 shared tile that the BF16 path's tensor-core math reads. The base read the
codes one byte at a time after the previous tile's math, so the load never overlapped it (the BF16 path's `cp.async`
does). The head loads tile t + 2's codes as one 8-byte load a unit of eight, and its row scale, at the top of tile t,
holds them in registers through the tile's math, then decodes them with the same per-element arithmetic (E4M3 to half to
float, times the scale, rounded to BF16) in one 16-byte store a unit. The 32-head form already uses 101,120 of the
101,376 shared bytes a block can take, so the codes cannot be staged in shared memory; registers rise (sm_121: 64 to
126 and 130 to 159, no spills), and one block an SM is still the shared memory's limit. The BF16 path is unchanged.

## Same bytes

| Check | Result |
| --- | --- |
| Teacher-forced top-20 prompt logprobs, 30 sequences, 360,418 positions (`lpk-b`, `lpk-h`) | base and head rows byte-identical, and identical to `glm-kv-fp8-3a56d4eb`'s FP8 run (`rows-sha256.txt`) |
| Golden count, prose, code with the head: drafted = plain (`exk-h`) | PASS, ids `5aaa4528`, `0d8280fe`, `eba1e4d8`, the FP8 ids of `glm-kv-fp8-3a56d4eb` |
| Concurrent, 12 cells: replies = alone, alone = `"draft": false` | PASS, 84 and 30 equal, 0 not |
| Resume: repeated, `"draft": false` and two at once = cold | PASS, 4 of 4 from 6,144 cached tokens |
| Replies after each cold prompt, 4 launches (`kv-*`) | one token sha256 a prompt size across base and head |

`lpk-*` and `exk-h` use `tools/suite-lp.sh` and `tools/suite-ex.sh` as `glm-kv-fp8-3a56d4eb` does, with the corpus of
`corpus/` (rebuilt by `tools/mktexts.py` and `tools/mkcorpus.py`; the 213 MB row files are not committed).

## Speed

`tools/suite-kv.sh` at `--context 131072 --parallel 4 --prompt-cache-gib 0 --kv-cache fp8`, alternated base, head,
head, base on one pair of machines: `openai_aligned.py nodraft` (`"draft": false`, 256 tokens, 5 reps after a warm-up),
then `bench_cache.py --require-cold --tokens 256` at 2,048 to 65,536 tokens (the median of its first and two repeated
cold requests; the drafted decode is of the 256 tokens after the prompt). `python3 tools/kvload_compare.py . kv-b1 kv-h1
kv-h2 kv-b2` prints:

| Row | kv-b1 | kv-h1 | kv-h2 | kv-b2 | base median | head median | change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `"draft": false` decode, fibonacci-raw (tok/s) | 23.58 | 23.71 | 23.81 | 23.49 | 23.54 | 23.76 | +0.9% |
| `"draft": false` decode, gpu-chat-no-think (tok/s) | 23.51 | 23.76 | 23.77 | 23.50 | 23.51 | 23.77 | +1.1% |
| Cold prompt of 2,048 tokens (s) | 1.41 | 1.38 | 1.38 | 1.40 | 1.40 | 1.38 | -1.4% |
| Drafted decode after 2,048 (tok/s) | 36.7 | 36.2 | 36.7 | 35.3 | 36.0 | 36.4 | +1.3% |
| Cold prompt of 8,192 tokens (s) | 6.02 | 5.73 | 5.70 | 5.97 | 6.00 | 5.72 | -4.6% |
| Drafted decode after 8,192 (tok/s) | 41.4 | 42.0 | 42.1 | 41.9 | 41.7 | 42.0 | +0.9% |
| Cold prompt of 32,768 tokens (s) | 23.71 | 22.59 | 22.49 | 23.47 | 23.59 | 22.54 | -4.5% |
| Drafted decode after 32,768 (tok/s) | 43.8 | 45.0 | 44.1 | 44.7 | 44.2 | 44.6 | +0.7% |
| Cold prompt of 65,536 tokens (s) | 47.44 | 45.25 | 45.00 | 46.99 | 47.22 | 45.12 | -4.4% |
| Drafted decode after 65,536 (tok/s) | 40.3 | 41.1 | 41.0 | 41.6 | 40.9 | 41.1 | +0.3% |

From 8,192 tokens up, every head launch is faster than every base launch. The teacher-forced runs agree: 302 s for the
head against 306 s for the base (`lpk-*/check-lp.log`; each position also returns 20 logprobs). For scale,
`glm-kv-fp8-3a56d4eb` measured the BF16 cache's cold prompts at 5.69, 22.88 and 45.84 s at 8,192, 32,768 and 65,536
tokens on an earlier binary; the FP8 cache's are now 5.72, 22.54 and 45.12. Every launch's ranks exited 0 after SIGINT
with 0 bytes held after cleanup (`*/memory-rank*.json`).

## Kernel limits

`tf-cuda-test kernel-contract-limits` over fatbins of every image for sm_80, 86, 89, 120 and 121 (`limits/`): with nvcc
13.0, both kernels' records are current on all five SMs, and the run's 18 stale records are all other images'
(`flashnext_mtp`, `exl3_experts`, `qwen35`); with nvcc 13.3 likewise, its 834 stale records all other kernels' (in
`glm_glue`, `tf_glm_router_tile` and `tf_glm_top1_sums`). The shared MLA image's chunk kernels and GLM's BF16 chunk
kernels are current with both. `tools/res.sh` reads the same counts from `cuobjdump -res-usage` of `glm_glue` alone;
base against head, only the two FP8 chunk kernels change among its 240 kernel-and-SM entries
(`limits/glm_glue-nvcc-13.0-diff.txt`).

## Files

- `lpk-b/`, `lpk-h/`, `exk-h/`, `kv-b1/`, `kv-h1/`, `kv-h2/`, `kv-b2/`: each launch's raw results, the clients' logs,
  both ranks' server logs and memory receipts, and the host's free memory before the launch and at ready.
- `rows-sha256.txt`: the teacher-forced row files' sha256, with `glm-kv-fp8-3a56d4eb`'s FP8 run.
- `limits/`: the limits runs' lines for both kernels, the shared MLA image and the stale records; the base against head
  `cuobjdump` diff of `glm_glue`.
- `tools/`: `build.sh`, `fl-env.sh` (documentation addresses), `run.sh`, `rank.sh`, `stop.sh`, `chain.sh` and
  `queue.txt` (the seven launches in order; `kb` is the base tree, `kh` the head); `suite-lp.sh`, `suite-ex.sh`,
  `suite-kv.sh`; `mktexts.py`, `mkcorpus.py`; `kvload_compare.py`; `res.sh`.
- `corpus/`: the sequences' plan and the texts' and ids' sha256. `commit.txt`, `inputs.txt` (each client and fixture the
  suites read, by sha256 and origin), `runs.log`. `SHA256SUMS` covers every file here except itself.
