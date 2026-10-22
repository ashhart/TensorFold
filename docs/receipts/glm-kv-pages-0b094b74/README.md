# GLM-5.3-Flash on CUDA: KV pages from the core pool, receipt

Tested implementation: the binaries built from `0b094b74…`, commit 11 of #1163 (commit.txt has the tree and the
commits): commits 1 to 7 are the change, 8 to 10 add the kernel contract's tests, gate 5's records and two host tests,
and 11 fixes a bug the first TP=3 run found (below). After it, `cbe29802` puts a comment on one line and this
receipt's commit adds this folder: neither changes engine or kernel source. The kernel contract ran on `3b41fd7c`
(commit 10); commit 11 changes no kernel. TP=3's ranks ran commit 11's engine source with two local commits for the
ring, which are not part of #1163 (`tp3/ring.patch`): rank 0's NCCL id handed to the other ranks in a file, so that a
ring of /32 links bootstraps (#1178 is the upstream way), and a heat ceiling (`TF_HEAT_CEILING`: a prompt chunk also
waits while its expected rise would pass the ceiling), which changes when chunks run, not what they compute.
Base: main `f8d254c5…` (after 1.1.2); main's binary for the replay and the speed A/B was built the same way
(`bin-f8d254c5`).
Test dates: 2026-10-22 and 10-23. Platform: MSI EdgeXpert (GB10, DGX Spark-compatible; sm_121, 128 GB shared
memory), two at TP=2 over a direct RoCE link, three at TP=3 over a ring of direct links; driver 580.178.04. Ranks in
Ubuntu 24.04 containers pinned to the efficiency cores (`--cpuset-cpus=0-4,10-14`), NCCL 2.30.7.
Toolchain: Zig 0.17.0, nvcc 13.3 (V13.3.73) for sm_121; gate 5's records also with nvcc 13.0 (V13.0.88), both for
sm_80, 86, 89, 120 and 121. binaries.txt has every binary's sha256.
Checkpoint: `nvidia/GLM-5.3-Flash-NVFP4` at `423acf37583782c51c142d145aef733d72943d93`.

## Scope

This qualifies GLM's streams on KV pages from the core pool at TP=2 and TP=3: the paged streams against their
whole-window twins byte for byte, the ranks' agreement on one pool, served exactness on pages (concurrent = alone,
drafted = `"draft": false`, resumed = fresh, images), the replies against main's, a long prompt beside short ones,
admission when the pool is short and when more requests arrive at once than there are paged sequences, speed against
main at TP=2, the kernel contract with gate 5's records, the no-GPU CI and the mutations. It does not qualify FP8
caches on pages (refused by name), DFlash2 drafts (no weights here), Metal or MLX.

## The bug commit 11 fixes

The first TP=3 run, on `b7fecb3d` (`tp3/found-b7fecb3d/r6all/`), served `TF_KV_PAGES=all` at `--parallel 8`. With
`all` the engine's own sequence stays idle, so eight lanes have seven paged sequences. Eight requests that arrive
together are admitted before any of them opens (the pool's check then sees every paged sequence free), and the last to
open took slot 8: binding it made a whole-window sequence, 12.88 GiB at a 1M window, before the pages refused it with
`LaneMemory`. Rank 0's host fell from 11 GiB free to 4 GiB in two seconds, its memory guard stopped the server
(`hostwatch-rank0.log`, `KILLED`), the other ranks ended on the closed line, and 53 of the 60 alone comparisons and 40
of the 72 `"draft": false` ones failed; none was unequal. TP=2's `all` (`--parallel 4`, a 128k window) took the same
path and made 1.77 GiB for slot 4, inside its reserve, so its replies were all equal. Commit 11 takes a stream's pages
before its sequence is bound: a slot past the paged sequences waits for one and nothing is made for it. Every served
run in this folder but that one ran commit 11.

## Results

| Check | Result |
| --- | --- |
| `glm-paged-check`, TP=2, first five layers (`tp2/check/`) | Rank 0: 7 prompts at the pages' edges (511 to 5,000 tokens) and 91 window rows equal their whole-window twins' logits, tokens and drafts byte for byte, the pages taken out of order (12 jumps) and NaN before use, the pool idle after; rank 1 made rank 0's pool and mapped its pages through every step |
| `glm-paged-check`, one GPU (`single/glm-paged-check.log`) | The same 7 prompts and 91 rows, byte for byte |
| Synthetic checkpoint, `TF_GLM_LAYERS=2` (`single/reg/`) | `glm-ref`, `glm-synth`, `glm-lanes-cover`, `glm-lanes-concurrent`, `glm-shared-check` and `glm-paged-check` pass, 0 failed |
| TP=2 `all` (`tp2/serve/r2all`) | 3 streams on 504 pages of 512 rows (3.14 GiB), 180 MiB fixed each, the engine's own sequence idle; `bench_concurrent.py`: 12 cells, every reply equal to its alone run (28/28) and to `"draft": false` (30/30); the reuse record's 44 requests all cached as planned; every idle census `free 504 + store 0 == total 504` |
| TP=2 `on` (`tp2/serve/r2on`) | 509 pages (3.17 GiB) beside the engine's own sequence: 28/28 and 30/30, 44 requests as planned, idle `free 509 + store 0 == total 509` |
| The `on` record replayed on this head's windows and on main (`tp2/serve/r2auto`, `r2main`, `tp2/compare/`) | `auto` keeps windows at TP=2 ("4 streams at once, 1.77 GiB each at a 131072-token window"), as main; both replays against the `on` and the `all` records: 43/43 replies equal and as cached as planned in all four, drafted == plain 23/23, no resumed reply slower than its fresh one |
| Images on pages (`tp2/serve/r2vis`) | 305 pages; `media_http.py`: drafted, repeated, `"draft": false` and three at once give the same tokens (prompt 2,050); `media_resume_http.py`: resumed past the image (6,144 of 7,605 tokens cached) = cold, as are `"draft": false` and two at once; idle `free 305 + store 0 == total 305` |
| Speed at TP=2 against main (`tp2/speed/`) | Serves in turn: this head, main, this head, main, this head with `all`. Decode, the 32k chat and cold prompts of 1k to 64k: this head's faster serve over main's slower 0.976 to 1.001, `all` 0.977 to 1.005; at 16k and 32k main's own two serves differ by 1.8% and 1.1%. The same tokens in all five |
| `glm-paged-check`, TP=3, first five layers (`tp3/check/`) | Rank 0: the same 7 prompts and 91 window rows equal their twins byte for byte, the pages out of order (12 jumps) and NaN before use, the pool idle after; ranks 1 and 2 made rank 0's pool and mapped its pages through every step |
| TP=3, pages by default, a 1M window (`tp3/serve/r6auto`) | `TF_KV_PAGES` unset pages at three ranks: "8 streams at once, the engine's own at a 1048576-token window (12.88 GiB, in the model's), 7 more taking 1284 pages of 512 rows (7.99 GiB) as they need them, 131 MiB fixed each, of 24.9 GiB left after a 12.2 GiB reserve"; `bench_concurrent.py`: 16 cells, every reply equal to its alone run (60/60) and to `"draft": false` (72/72); the reuse record's 44 requests all cached as planned; cold prompts of 1k to 64k tokens 0.73 s to 43.3 s (`cold.json`); seven 2k prompts while a 64k conversation held its rows on pages (63,488 of its 65,536 tokens resumed from the cold run's kept state): all seven equal their alone runs and none waited for another request to end (`long-beside-short.json`); every idle census `free 1284 + store 0 == total 1284` |
| TP=3 `all` (`tp3/serve/r6all`) | 1,237 pages of 512 rows (7.70 GiB), 131 MiB fixed each, the engine's own sequence idle, so seven streams at once; `bench_concurrent.py`: 16 cells, every reply equal to its alone run (60/60) and to `"draft": false` (72/72); in the cells of eight the eighth request waited for a paged sequence (their longest first token 12.0 to 21.2 s, against 1.7 to 2.1 s with `auto`); every idle census `free 1237 + store 0 == total 1237` |
| TP=3 `off`, the `auto` record replayed (`tp3/serve/r6off`) | A window a stream with `--parallel auto`: "2 streams at once, 12.88 GiB each at a 1048576-token window, of 24.7 GiB left after a 12.2 GiB reserve" (a named `--parallel 8` is refused); against the `auto` record (`compare-auto.log`): 43/43 replies equal and as cached as planned, drafted == plain 23/23, no resumed reply slower than its fresh one |
| Admission on a short pool (`tp3/serve/r6admit`) | `TF_KV_PAGES=all` with a 22.8 GiB prompt cache: 206 pages (1.28 GiB); `kvp_admit.py --fit` took 64k prompts, 129 pages each, so the pool holds one and not two. Three alone (cold, each kept in the prompt cache), then together: 3/3 equal their alone runs, each resumed 63,488 of its 65,536 tokens, and the second and the third started only after the one before ended; idle `free 206 + store 0 == total 206` |
| The run that found commit 11's bug (`tp3/found-b7fecb3d/r6all/`) | `TF_KV_PAGES=all` at TP=3 on `b7fecb3d` (above): the 8-stream cell's requests ended incomplete when rank 0 was stopped, and every later one could not connect; 7 of 60 alone and 32 of 72 `"draft": false` comparisons equal, 0 unequal |
| Kernel contract (`single/kernel-contract.log`, on `3b41fd7c`) | `SUMMARY 148 kernels: 148 pass gates 1-4, 0 fail, 1 known findings; 32 controls: 32 red as planted, 0 missed; 1018 limit problems` (the known finding: `tf_glm_scores` scores a NaN dot 0); main's (`single-main/kernel-contract.log`): 140 kernels, 140 pass, 0 known, 32 of 32 controls red, 1018; every limit problem line is main's |
| Gate 5's records (`gate5/`) | Fatbins for sm_80, 86, 89, 120 and 121 from nvcc 13.0.88 and 13.3.73: the six launches fit their limits on all five; on this GPU each runs with its 13.3 record's registers and shared memory (`PASS limits live`) |
| No-GPU CI under WSL2 on commit 11 (`ci/`) | `zig build test` 3,463/3,479 (15 skipped; the failure is upstream's `cuda_cost_identity` mtime test, WSL2's coarse file times), `test-kernel-contract` 72/72, `test-golden`, the server and pull security tests 20/20, `fuzz-cortheon` 14/14, the served Flash Next unittest (8), the GLM snapshot, KV pool and CUDA host tests 71/72 (the same failure); lean 439: the one problem past main's 438 is commit 11's two-line comment, which commit 12 puts on one line |
| Mutations (`ci/mutations/`) | Fifteen guards, each broken alone on a copy of commit 11: every one fails a test (`head.log`: the unchanged copy fails only `cuda_cost_identity`); the fifteenth binds a stream's sequence before its pages, as before commit 11 |

## Reproduction

Every serve: `tensorfold-native serve CKPT --name glm-tf --host 127.0.0.1 --port 8097 --max-tokens 32768
--reasoning-effort high --parallel P --context C`, drafts on, one rank a host (`TF_GLM_RANK`, `TF_GLM_WORLD`),
`TF_HEAT_HIGH=92 TF_HEAT_LOW=88 TF_HEAT_CEILING=93`. TP=2: `--context 131072 --parallel 4`; `r2all`, `r2on` and
`r2vis` add `--prompt-cache-gib 4` and `TF_KV_POISON=1` with `TF_KV_PAGES` `all` or `on`, `r2vis` `--vision`;
`r2auto` and `r2main` and every speed serve `--prompt-cache-gib 0`. TP=3: `--context 1048576 --parallel 8` and
`TF_KV_POISON=1`; `r6auto` with `TF_KV_PAGES` unset and the default prompt cache (16 GiB), `r6all` `TF_KV_PAGES=all`,
`r6admit` `TF_KV_PAGES=all` with a 22.8 GiB prompt cache (the `all` serve's room less its streams' fixed parts and 0.9 GiB), `r6off`
`TF_KV_PAGES=off --prompt-cache-gib 0 --parallel auto`; NCCL over the ring's direct links with `NCCL_DEBUG=INFO`, and
each TP=3 host's memory guard (v3's `hostwatch.sh`: the server stopped below 5 GiB free). The checks are the
repository's tools: `bench_concurrent.py --alone --serial --tokens 256 --reps 1`, `reuse_served.py` (`record --planned
--min-prompt 2048`, `replay`, `compare`, `cold`, `ids`), `bench_openai.py`, `bench_cache.py --require-cold`; here
`media_http.py`, `media_resume_http.py`, `kvp_admit.py` and `kvp_sum.py` (tools/; `kvp_sum.py conc` ends non-zero
when a request failed or a reply differed). `tf-cuda-test glm-paged-check CKPT RANK WORLD IDFILE` with
`TF_GLM_LAYERS=5`. runs/ has the chains' own output.

## Remaining coverage limits

- TP=3's ranks ran commit 11's engine source with `tp3/ring.patch` (above); TP=2's and one GPU's ran commit 11 alone.
- Ranks 1 and 2 of the TP=3 check print their PASS lines in the middle of NCCL's INFO lines (`NCCL_DEBUG=INFO`).
- Speed against main was measured at TP=2 only; TP=3's cold prompt times are in `tp3/serve/r6auto/cold.json`, with no
  main serve beside them.
- `--kv-cache fp8` keeps windows: `auto` does not page and `on` and `all` are refused (`Fp8OnPages`).
- DFlash2 (`--drafter`): no weights here; the drafts on pages are the MTP head's.
- Metal and MLX were not run (no Mac).

## Publication handling

CRLF line ends made LF and trailing blanks dropped; machine names, internal addresses, home and local paths, and the
work's session, plan, records, container and image names replaced with placeholders (`redaction-report.json` counts
them a file; host-a to host-c run TP=3's ranks 0 to 2, host-d and host-e TP=2's ranks 0 and 1); result values, token
ids and token hashes kept. The TP=3 server logs keep NCCL's INFO lines as they were written. The reuse records keep
their rows without the request bodies (3 MB a record): each row has its body's sha256 and size, and
`records-sha256.txt` the originals'. The speed runs' prompt files are not included: `reuse_served.py cold` and `ids`
make them again. SHA256SUMS pins every published file except itself.
