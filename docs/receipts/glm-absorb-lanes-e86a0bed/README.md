# GLM-5.3-Flash's MLA absorb and expand without lane trees: prompt rows, same bits

Base: `e86a0bed1d` (main). Head: the base plus `1596a09cd0`, which changes `zig/kernels/cuda/glm_glue_latent.cuh` only
(`commit.txt` has the build, toolchain and both binaries' sha256; one binary a tree, the same file on both ranks).
Test date: 2026-11-16. Platform: two DGX Spark (GB10, sm_121, 128 GB) at TP=2 over a direct RoCE link, driver
580.178.04, Zig 0.17.0, nvcc 13.5, NCCL 2.30.7. Checkpoint: `nvidia/GLM-5.3-Flash-FP4-Local` at
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, BF16 latent cache, drafts the default `MTP + copies`.

## Scope

`tf_glm_absorb_q` (q . wk into the 512-wide latent) and `tf_glm_expand_v` (latent . wv into 256-wide v heads) sum
each output as 8-fma chains spread over lanes, then reduce them through shuffle trees. In a prompt chunk that kept
the LSU pipe at 83% while the FMA pipe sat at 21% (Nsight Compute on one GB10). For prompt blocks (64 rows a block)
the head forms in one thread the partials main spreads over lanes, each the same fma chain in the same order, and
adds them in main's tree shape before main's last shuffles; the q and o rows come through shared memory 4 rows a
pass with 16-byte loads. Decode windows (blocks under 64 rows) run main's loops unchanged: their 16-byte weight loads
are what a decode window's cold kv_b read needs. Every output keeps its bits.

## Same bits

| Check | Result |
| --- | --- |
| Kernel, main against head, random inputs, rows 1 to 70, 100, 127 to 129, 1000, 2047, 2048, 4096 (`bits/latent_eq.out`) | 78 sizes, both kernels byte-identical |
| The same after the decode loops became main's kernel bodies token for token (`bits/latent_eq_main_loops.out`) | 78 sizes, both kernels byte-identical |
| Replies after each cold prompt, 4 launches (`fa-*`) | one token sha256 a prompt size across base and head |
| Golden count, prose, code: drafted = plain (`fe-h`, `fe-b`) | PASS on both, the same token sha256 `ba28cfb7`, `dae7e2a5`, `db45a54e` |
| Concurrent, 12 cells: replies = alone, alone = `"draft": false` (`fe-*/check-conc.log`) | PASS on both, 0 unequal |
| Resume: repeated, `"draft": false` and two at once = cold (`fe-*/check-resume.log`) | PASS on both |

## Speed

`tools/suite-ab.sh` at `--context 65536 --parallel 4 --prompt-cache-gib 0`, alternated base, head, head, base on one
pair of machines: `openai_aligned.py nodraft` (`"draft": false`, 128 tokens, 3 reps after a warm-up), then
`bench_cache.py --require-cold --tokens 16` at 8,192 and 32,768 tokens (the median of its first and two repeated cold
requests). `python3 tools/ab_compare.py . fa-b1 fa-h1 fa-h2 fa-b2` prints:

| row | fa-b1 | fa-h1 | fa-h2 | fa-b2 | base med | head med | change |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode fibonacci-raw tok/s | 23.917 | 24.502 | 24.555 | 23.873 | 23.895 | 24.529 | +2.65% |
| decode gpu-chat-no-think tok/s | 23.926 | 24.512 | 24.498 | 23.786 | 23.856 | 24.505 | +2.72% |
| cold 8192 s | 5.662 | 5.547 | 5.472 | 5.657 | 5.659 | 5.510 | -2.65% |
| cold 32768 s | 22.247 | 21.998 | 21.598 | 22.237 | 22.242 | 21.798 | -1.99% |

Every head launch beats every base launch in every row. The decode rows run main's loops on both sides, so their gain
is not claimed as this change's. In isolation (`bits/latent_cold.out`, L2 flushed before each call) decode-sized
blocks take the same time on both sides, and a 2,048-row prompt block's pair of calls drops from 9.97 to 5.72 ms.

## Kernel contract

Both kernels are registered (`zig/tests/cuda/contract_glm_latent.zig`): gates 1 to 4 against float64 at decode and
prompt shapes, a next-head control for each that gate 3 must report red, and records for nvcc 13.0 and 13.3 on
sm_80/86/89/120/121 (2,560 and 4,224 static shared bytes, bound 128; `expand_v` 96 registers on all five, `absorb_q`
111, 113, 113, 96 and 96 since its decode loop became main's kernel body token for token, the same with 13.0 and
13.3). On GB10 with the runner built by nvcc 13.3 (`contract-13.3.txt`, before that change): `(filtered) SUMMARY 2
kernels: 2 pass gates 1-4, 0 fail, 0 known findings; 2 controls: 2 red as planted, 0 missed; 0 limit problems`, exit
0; rerun after it, the same summary with the live limits current (96 registers on sm_121).

## Files

- `fa-b1/`, `fa-h1/`, `fa-h2/`, `fa-b2/`: the speed launches' results, client logs and both ranks' server logs.
- `fe-h/`, `fe-b/`: the exactness launches (`tools/suite-ex.sh`, `--context 40960 --parallel 4`).
- `bits/`: `latent_eq.out` and `latent_cold.out` from `tools/latent_eq.cu` and `tools/latent_cold.cu` (main's kernels
  from the base's header against the head's, renamed by `tools/mk_new_latent.py`).
- `tools/`: `env.sh` (documentation addresses), `rank.sh`, `up.sh`, `down.sh`, `abba.sh`, `build.sh`, `suite-ab.sh`,
  `suite-ex.sh`, `ab_compare.py`, `mkprompts.py`, and the bit tests.
- `contract-13.3.txt`, `commit.txt`, `inputs.txt` (each client and fixture by sha256 and origin). `SHA256SUMS` covers
  every file here except itself.
