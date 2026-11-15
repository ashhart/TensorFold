# GLM-5.3-Flash's FP8 latent cache on CUDA: quality and exactness on FP4-Local at TP=2

Tested commit: `3a56d4eb92` (`commit.txt` has the build, toolchain and the binary's sha256; one binary for every launch,
on both ranks). Test date: 2026-11-15. Platform: two DGX Spark (GB10, sm_121, 128 GB shared memory) at TP=2 over a
direct RoCE link, Ubuntu 24.04.5, driver 580.178.04, Zig 0.17.0, nvcc 13.5, NCCL 2.30.7. Checkpoints:
`nvidia/GLM-5.3-Flash-FP4-Local` at `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` and, for scale,
`nvidia/GLM-5.3-Flash-NVFP4` at `423acf37583782c51c142d145aef733d72943d93`. Drafts are the default, `MTP + copies`.

## Scope

`--kv-cache fp8` keeps GLM's 512-wide latent rows as E4M3 codes with an FP32 scale a row; the rotary and index keys stay
BF16. It changes tokens against the BF16 cache (all three golden replies differ), so it is not a qualified mode; this
receipt is diagnostic evidence of how far it moves, on FP4-Local at TP=2, with every other setting default:

- Fidelity, teacher-forced: one binary and checkpoint, the BF16 cache against the FP8 cache, top-20 prompt logprobs at
  every position of 30 sequences (360,418 scored positions) from 2,048 to 65,536 tokens. A prompt pass writes its
  chunk's latent rows into the cache before any row attends (`cuda_dsa_block.zig`), so every key it reads, its own
  chunk's included, comes through the FP8 cache, as in decode.
- Control: the BF16 run twice gives the same bytes, so every difference below is the cache's.
- Scale: NVIDIA's NVFP4 checkpoint against FP4-Local, both with the BF16 cache, on the same sequences: the distance
  between the publisher's two FP4 versions of one model.
- Exactness, speed and memory with each cache: the same suites and settings, one launch each (`ex-*` and `sp-*`).

Not measured: a BF16-weights reference (the 1.1.5-era receipt's is not published), other checkpoints with the FP8 cache,
TP=1 or TP=3, other GPUs, Metal, and the combinations the engine refuses (FP8 on explicit pages, #3062/#3064; FP8 with
host KV). Nothing here changes the default, which stays BF16.

## Inputs

Two public text domains of TensorFold itself at `cc8f2889e7`: prose, every `docs/**/*.md` outside `docs/receipts`, and
code, `zig/src/**/*.zig` in path order (`tools/mktexts.py`). Each file goes through the server's `/tokenize` without
special tokens, and each domain's ids are cut into one sequence of 65,536 tokens, two of 32,768, four of 8,192 and eight
of 2,048, which share no token (`tools/mkcorpus.py`; `corpus/plan.json`, and `corpus/sha256.txt` for the texts and ids).
The checkpoint adds no prefix to a raw prompt, so each sequence starts inside a text. `tools/suite-lp.sh` sends each
sequence through `/v1/completions` with `prompt_logprobs: 20` and `max_tokens: 1` (the 1.1.5-era receipt's
`s5_prompt_lp.py`), at `--context 131072 --parallel 1 --prompt-cache-gib 0`.

## Fidelity

`tools/kv8_compare.py` with the BF16-cache run as the reference (`lp/cmp-fp8.json`, `lp/cmp-nvfp4.json`):

| Against FP4-Local with the BF16 cache | FP8 cache | NVFP4, BF16 cache (scale) |
| --- | --- | --- |
| Mean NLL, all 360,418 positions (BF16 cache: 1.40910) | 1.40831 (−0.06%) | 1.40912 (+0.001%) |
| Best token the same | 95.49% (96.27% within ties) | 92.23% (93.01%) |
| KL over the reference's top 20, mean / p99 | 0.0162 / 0.151 | 0.0432 / 0.408 |
| Mean \|Δ logprob\| of the given token | 0.092 | 0.163 |
| Given token near sure (p > 0.99) on one side, below even odds on the other: BF16-cache sure / other sure | 52 / 64 of 127,647 | 149 / 137 |

By domain and position (FP8 cache):

| Rows | Positions | Best token the same | Mean NLL, BF16 → FP8 cache | KL mean / p99 |
| --- | ---: | ---: | --- | --- |
| Prose | 180,209 | 93.22% | 2.1942 → 2.1937 (−0.03%) | 0.0246 / 0.198 |
| Code | 180,209 | 97.76% | 0.6240 → 0.6229 (−0.17%) | 0.0078 / 0.114 |
| Positions 1 to 2,047 | 61,410 | 93.96% | 1.7511 → 1.7479 (−0.18%) | 0.0194 / 0.206 |
| 2,048 to 8,191 | 86,016 | 95.19% | 1.4816 → 1.4801 (−0.10%) | 0.0126 / 0.129 |
| 8,192 to 32,767 | 147,456 | 95.96% | 1.2811 → 1.2766 (−0.35%) | 0.0152 / 0.124 |
| 32,768 to 65,535 | 65,536 | 96.27% | 1.2815 → 1.2922 (+0.84%) | 0.0202 / 0.226 |

The NVFP4 scale shows the same last band at +0.68%. That band is one prose sequence; its rise sits at positions 48,128
to 54,271, changelog-style lists of near-identical pull request links, where a small change to any sum moves which
earlier link the model copies. There, as everywhere, the near-sure tokens go both ways (21 lost and 18 gained with the
FP8 cache). Code has none either way, and no position below 8,192 has any.

All 16,250 positions whose best token differs are in `lp/flips-fp8.jsonl` with both sides' margins. The reference's
margin between its best token and the FP8 cache's is at most 0.125 at half of them, 0.5 at 86% and 1.0 at 95%; the
largest, 16.8, is in the link lists above. Per-sequence NLL is in `cmp-fp8.json` (`sequences`): the largest rise is
+0.82% on a code sequence of 8,192 tokens, the largest fall −1.82% on a prose one of 2,048.

## Exactness with each cache

`tools/suite-ex.sh` at `--context 40960 --parallel 4`, every other setting default (`ex-bf16/`, `ex-fp8/`): the
1.1.5-era receipt's `golden_http.py` (three 512-token greedy prompts, drafted and with `"draft": false`),
`bench_concurrent.py --levels 1,2,4 --temperatures 0,1 --reps 3 --alone --serial`, and `resume_http.py` (a 6,594-token
chat, resumed four ways).

| Check | BF16 cache | FP8 cache |
| --- | --- | --- |
| Golden count, prose, code: drafted = plain | PASS, ids `ba28cfb7`, `dae7e2a5`, `db45a54e` | PASS, ids `5aaa4528`, `0d8280fe`, `eba1e4d8` |
| Tokens a round with drafts, count / prose / code | 4.20 / 1.92 / 3.01 | 4.83 / 2.01 / 3.10 |
| Concurrent, 12 cells: replies = alone, alone = `"draft": false` | PASS, 84 and 30 equal, 0 not | PASS, 84 and 30 equal, 0 not |
| Resume: repeated, `"draft": false` and two at once = cold | PASS, from 6,144 cached tokens | PASS, from 6,144 cached tokens |

The BF16 cache's golden ids are the 1.1.5-era receipt's (`../glm-fp4local-80517fc3`). The FP8 cache's replies leave them
after 7, 171 and 38 characters, inside the thinking text. The cache is lossy, so its replies are its own, and within
them drafts, concurrency and resumes change no token.

## Speed and memory against the BF16 cache

`tools/suite-sp.sh` at `--context 131072 --parallel 4 --prompt-cache-gib 0` (`sp-bf16/`, `sp-fp8/`): `bench_openai.py`
(drafted) and the 1.1.5-era receipt's `openai_aligned.py nodraft` (`"draft": false`), 256 tokens and 5 reps after a
warm-up; `bench_cache.py --require-cold --tokens 16` on token files cut from its `dc.json` (`inputs.txt`). One binary,
the BF16 launches an hour before the FP8 ones.

| Row | BF16 cache | FP8 cache |
| --- | --- | --- |
| Cache at a 131,072-token window, 4 streams (`sp-*` startup line) | 1,028 pages of 512 rows, 6.40 GiB, as streams need them | 1.08 GiB a stream, whole windows (`TF_KV_PAGES` keeps FP8 off pages) |
| Cache bytes a token on each rank, 12 latent caches (`cacheBytes` in `cuda_latent.zig`) | 13,056 = 12 × (1,024 latent + 64 index), the recipe's | 6,960 = 12 × (512 codes + 4 scale + 64 index) (−47%); the startup line gives (1.08 GiB − 231 MiB) / 131,072 ≈ 7,000 |
| Fixed state a stream, whatever its length | 231 MiB (startup line) | the same buffers |
| A whole stream at a 131,072-token window | about 1.82 GiB | 1.08 GiB (−41%) |
| `"draft": false` decode, fibonacci-raw / gpu-chat-no-think (tok/s) | 23.9 / 23.9 | 23.5 / 23.6 |
| Drafted decode, greedy (tok/s) | 42.3 / 34.4 | 44.7 / 36.5 (different replies, so different drafts) |
| Cold prompt of 8,192 / 32,768 / 65,536 tokens (s) | 5.69 / 22.88 / 45.84 | 5.99 / 23.59 / 47.27 (3 to 5% slower) |
| Drafted aggregate at 4 streams, code / chat, greedy (tok/s) | 113.8 / 85.4 | 110.5 / 86.7 |
| Process-owned CUDA buffers at load end, each rank, 40,960-token launch | 90.20 GiB | 90.47 GiB |

Every launch's ranks exited 0 after SIGINT with 0 bytes held after cleanup (`*/memory-rank*.json`).

## Files

- `lp/`: each teacher-forced launch (`bf16`, `fp8`, `bf16b` the control, `nvfp4` the scale) with its client log, both
  ranks' server logs and memory receipts; `rows-sha256.txt` for the four 213 MB row files, which are not committed;
  `cmp-fp8.json`, `cmp-nvfp4.json` and `flips-fp8.jsonl` from `kv8_compare.py`.
- `corpus/`: `plan.json` (each sequence's domain and length, and the ids' sha256) and `sha256.txt`.
- `ex-bf16/`, `ex-fp8/`, `sp-bf16/`, `sp-fp8/`: each cache's exactness and speed launch: the raw results, the clients'
  logs, both ranks' server logs and memory receipts, and the host's free memory before the launch and at ready. `python3
  tools/summarize.py . ex-bf16 sp-bf16 ex-fp8 sp-fp8` prints their tables.
- `tools/`: `build.sh`; `fl-env.sh` (the ranks' environment, with documentation addresses), `run.sh` (one launch: rank 1
  on the peer, rank 0 here, the suite, then both stopped by SIGINT), `rank.sh` (a rank under
  `tools/memory_receipts.py`), `stop.sh`, `chain.sh` and `queue.txt` (the eight launches in order; `ex-head` and
  `sp-head` there are `ex-bf16/` and `sp-bf16/` here); `suite-ex.sh`, `suite-sp.sh`, `suite-lp.sh`; `mktexts.py`,
  `mkcorpus.py`; `kv8_compare.py` (with a control that two equal runs give zero differences); `summarize.py`.
- `commit.txt`: the commit, build, toolchain and binary. `inputs.txt`: the sha256 of each client and fixture the suites
  read and where it comes from (the tested commit's `tools/`, the 1.1.5-era receipt's `tools/`, or the command that
  makes it).
- `runs.log`: each launch's start, ready, suite result and end. `SHA256SUMS` covers every file here except itself.

To rerun the comparison from the row files: `python3 tools/kv8_compare.py corpus/plan.json bf16.jsonl fp8.jsonl
cmp-fp8.json flips-fp8.jsonl`.
