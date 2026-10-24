# GLM-5.3-Flash on CUDA: the engine's own sequence on KV pages, receipt

Tested implementation: the binaries built from `8e705835…`, commit 20: #1163's 14 commits and the first six of the
pull request that stacks on it (commit.txt has the tree and the commits). The receipt's commit adds this folder and
changes no source. TP=3's ranks ran commit 20's engine source with the two local commits for the ring that #1163's
receipt describes (`tp3/ring.patch`): rank 0's NCCL id handed to the other ranks in a file (#1178 is the upstream
way), and a heat ceiling (`TF_HEAT_CEILING`: a prompt chunk also waits while its expected rise would pass the
ceiling), which changes when chunks run, not what they compute.
Base: #1163's head `d7ff6494…`, on main `f8d254c5…` (after 1.1.2); main's binary for the replay and the speed A/B
was built the same way (`bin-f8d254c5`).
Test date: 2026-10-23. Platform: MSI EdgeXpert (GB10, DGX Spark-compatible; sm_121, 128 GB shared memory), two at
TP=2 over a direct RoCE link, three at TP=3 over a ring of direct links; driver 580.178.04. Ranks in Ubuntu 24.04
containers pinned to the efficiency cores (`--cpuset-cpus=0-4,10-14`), NCCL 2.30.7.
Toolchain: Zig 0.17.0, nvcc 13.3 (V13.3.73) for sm_121. binaries.txt has every binary's sha256.
Checkpoint: `nvidia/GLM-5.3-Flash-NVFP4` at `423acf37583782c51c142d145aef733d72943d93`.

## Scope

This qualifies the engine's own sequence (slot 0) on KV pages at TP=2 and TP=3: slot 0 on pages from its init
against its whole-window twin byte for byte, the ranks' agreement on `TF_KV_PAGES`, served exactness with every
stream on pages (concurrent = alone, drafted = `"draft": false`, resumed = fresh, images), the replies against main's,
the floor (one window of pages planned before the rest) holding a long request while short ones share the pool,
admission when the pool is down to its floor, speed against main at TP=2, the no-GPU CI and the mutations. The
kernels are #1163's, unchanged, so the kernel contract and gate 5's records are in #1163's receipt. It does not
qualify FP8 caches on pages (refused by name, as in #1163), DFlash2 drafts (no weights here), Metal or MLX.

## What changed for these runs

- `TF_KV_PAGES=all` is now `on`: every stream, the engine's own among them, is on pages. So `r2all` and `r2on` serve
  the same configuration twice, and `r6all` is the concurrency check at a 96k window with all eight streams on pages.
- Every rank resolves `TF_KV_PAGES` and the ranks agree on it; the chains pass the same environment to every rank.
- The floor is shown two ways: `r6admit` serves a 96k window with the pool down to its floor and a page a stream
  (64k requests: one fits, the next waits), and `r6auto` serves seven 2k prompts beside a 64k conversation at a 1M
  window. No request was sent at 1M tokens (below).

## Results

| Check | Result |
| --- | --- |
| `glm-paged-check`, TP=2, first five layers (`tp2/check/`) | Rank 0: on the engine's own sequence, on pages from its init, 7 prompts at the pages' edges (511 to 5,000 tokens) and 91 window rows equal their whole-window twins' logits, tokens and drafts byte for byte, the pages out of order (12 jumps) and NaN before use, the pool idle after; rank 1 made rank 0's pool and mapped its pages through every step |
| `glm-paged-check`, one GPU (`single/glm-paged-check.log`) | The same 7 prompts and 91 rows on the engine's own sequence, byte for byte |
| Synthetic checkpoint, `TF_GLM_LAYERS=2` (`single/reg/`) | `glm-ref`, `glm-synth`, `glm-lanes-cover`, `glm-lanes-concurrent`, `glm-shared-check` and `glm-paged-check` pass, 0 failed |
| TP=2 `all` and `on` (`tp2/serve/r2all`, `r2on`) | The same configuration twice: "4 streams at once, the engine's own and 3 more taking 803 pages of 512 rows (5.00 GiB) as they need them (enough for one at a 131072-token window), 180 MiB fixed each, of 9.5 GiB left after a 12.2 GiB reserve" (757 pages under `on`, whose serve had 9.2 GiB left); #1163's had 504 pages with the engine's own sequence idle under `all`, and 509 beside its window under `on`. `bench_concurrent.py`: 12 cells each, every reply equal to its alone run (28/28) and to `"draft": false` (30/30); each reuse record's 44 requests all cached as planned; every idle census `free 803 + store 0 == total 803` (757 under `on`) |
| The `on` record replayed on windows and on main (`tp2/serve/r2auto`, `r2main`, `tp2/compare/`) | `auto` keeps windows at TP=2 ("4 streams at once, 1.77 GiB each at a 131072-token window"), as main does; both replays against the `on` and the `all` records: 43/43 replies equal and as cached as planned in all four, drafted == plain on the same messages 23/23, no resumed reply slower than its fresh one |
| Images on pages (`tp2/serve/r2vis`) | 561 pages (3.49 GiB; 340 MiB fixed each with `--vision`); `media_http.py`: drafted, repeated, `"draft": false` and three at once give the same tokens (prompt 2,050); `media_resume_http.py`: resumed past the image (6,144 of 7,603 tokens cached) = cold, as are `"draft": false` and two at once; idle `free 561 + store 0 == total 561` |
| Speed at TP=2 against main (`tp2/speed/`) | Serves in turn: this head (windows, as `auto` keeps them at two ranks), main, this head, main, then this head with `all` (every stream on pages, the engine's own among them). Decode, the 32k chat and cold prompts of 1k to 64k: this head's faster serve over main's slower 0.990 to 1.000, `all` 0.994 to 1.005; at 64k main's own two serves differ by 0.7%. The same tokens in all five |
| `glm-paged-check`, TP=3, first five layers (`tp3/check/`) | Rank 0: on the engine's own sequence, on pages from its init, 7 prompts at the pages' edges (511 to 5,000 tokens) and 91 window rows equal their whole-window twins' logits, tokens and drafts byte for byte, the pages out of order (12 jumps) and NaN before use, the pool idle after; ranks 1 and 2 made rank 0's pool and mapped its pages through every step |
| TP=3, pages by default, a 1M window (`tp3/serve/r6auto`) | `TF_KV_PAGES` unset pages at three ranks: "8 streams at once, the engine's own and 7 more taking 3355 pages of 512 rows (20.89 GiB) as they need them (enough for one at a 1048576-token window), 131 MiB fixed each, of 37.8 GiB left after a 12.2 GiB reserve" (#1163's: the engine's own 12.88 GiB window in the model's and 1,284 pages for the other seven); `bench_concurrent.py`: 16 cells, every reply equal to its alone run (60/60) and to `"draft": false` (72/72); the reuse record's 44 requests all cached as planned; cold prompts of 1k to 64k tokens 0.73 s to 43.4 s (`cold.json`); seven 2k prompts while a 64k conversation held its rows on pages (63,488 of its 65,536 tokens resumed from the cold run's kept state): all seven equal their alone runs and none waited for another request to end (`long-beside-short.json`); every idle census `free 3355 + store 0 == total 3355` |
| TP=3 `all`, a 96k window (`tp3/serve/r6all`) | `all` is `on`: "8 streams at once, the engine's own and 7 more taking 1544 pages of 512 rows (9.61 GiB) as they need them (enough for one at a 98304-token window)", the bound of eight windows' pages; 16 cells, 60/60 and 72/72; in the cells of eight no request waited (the longest first token 1.69 to 2.15 s; #1163's `all`, with the engine's own sequence idle, had the eighth wait, 12.0 to 21.2 s); every idle census `free 1544 + store 0 == total 1544` |
| TP=3 `off`, the `auto` record replayed (`tp3/serve/r6off`) | A window a stream with `--parallel auto`: "2 streams at once, 12.88 GiB each at a 1048576-token window, of 24.4 GiB left after a 12.2 GiB reserve"; against the `auto` record (`compare-auto.log`): 43/43 replies equal and as cached as planned, drafted == plain 23/23, no resumed reply slower than its fresh one |
| The pool at its floor (`tp3/serve/r6admit`) | `TF_KV_PAGES=all` at a 96k window with a 37.2 GiB prompt cache asked: the plan trims the cache to 35.9 GiB to keep the floor, so the pool is the floor and a page a stream, 200 pages (1.25 GiB); `kvp_admit.py --fit` took 64k prompts, 129 pages each, so the pool holds one and not two. Three alone (cold, each kept in the prompt cache), then together: 3/3 equal their alone runs, each resumed 63,488 of its 65,536 tokens, and they ran one at a time, each starting after the one before ended; idle `free 200 + store 0 == total 200` |
| Builds on a GB10 (`tp2/build.log`, `tp2/build/`, `tp3/build.log`) | Commit 20, nvcc 13.3.73 `-Dsm=121`: `zig fmt --check` on the changed files, the build, the native server, `test-cuda-host` (155/155 steps, 2,516/2,522 tests, 6 skipped), `test-kv-pool` (17/17) and `test` (276/276 steps, 2,023/2,037 tests, 14 skipped) pass. Their `failed command` lines are zig's build runner showing the stderr of tests that print on purpose (a comparator's FAIL line for a planted NaN, a rank's stop message), and `lean rc=1` is the lean list's exit with main's 438 problems, none new. The ring's binaries came from the TP=3 tree's first build, commit 20 with `tp3/ring.patch` (`tp3/build.log`), whose steps pass the same way; the TP=2 chain built the same commit again later (`runs/tp2-chain.txt`, its binary another sha), and the ring ran the first |
| No-GPU CI under WSL2 on commit 20 (`ci/`) | `zig build test` 3,470/3,486 (15 skipped; the failure is upstream's `cuda_cost_identity` mtime test, WSL2's coarse file times), `test-kernel-contract` 72/72, `test-golden`, the server and pull security tests 20/20, `fuzz-cortheon` 14/14, the served Flash Next unittest (8), the GLM snapshot, KV pool and CUDA host tests 72/73 (the same failure); lean 438, as on main |
| Mutations (`ci/mutations/`) | Thirty guards, each broken alone on a copy of commit 20: #1163's fifteen and fifteen for the engine's own pages (the floor, its bound, `holds`, the flag to `plan` and `holds`, `window_rows`, `Plan.size`, slot 0 made and placed on pages, a pool refused beside a window, admission by the pool alone, the ranks' agreement and fingerprint). 29 fail a test; #1163's `bytes-on-paged-seq` no longer applies, as its line became `Plan.size`, which two of the new fifteen break. The unchanged copy fails no test (`head.log`) |

## Reproduction

Every serve: `tensorfold-native serve CKPT --name glm-tf --host 127.0.0.1 --port 8097 --max-tokens 32768
--reasoning-effort high --parallel P --context C`, drafts on, one rank a host (`TF_GLM_RANK`, `TF_GLM_WORLD`),
`TF_HEAT_HIGH=92 TF_HEAT_LOW=88 TF_HEAT_CEILING=93`, the same `TF_KV_PAGES` and `TF_KV_POISON` on every rank.
TP=2: `--context 131072 --parallel 4`; `r2all`, `r2on` and `r2vis` add `--prompt-cache-gib 4` and `TF_KV_POISON=1`
with `TF_KV_PAGES` `all` or `on`, `r2vis` `--vision`; `r2auto` and `r2main` and every speed serve
`--prompt-cache-gib 0`. TP=3: `--parallel 8` and `TF_KV_POISON=1`; `r6auto` `--context 1048576` with `TF_KV_PAGES`
unset and the default prompt cache (16 GiB); `r6all` `--context 98304 TF_KV_PAGES=all`; `r6admit` `--context 98304
TF_KV_PAGES=all` with a 37.2 GiB prompt cache asked (the `all` serve's room less its streams' fixed parts),
which the plan trims to keep the floor; `r6off` `--context 1048576 TF_KV_PAGES=off --prompt-cache-gib 0 --parallel
auto`; NCCL over the ring's direct links with `NCCL_DEBUG=INFO`, and each TP=3 host's memory guard (v3's
`hostwatch.sh`: the server stopped below 5 GiB free). The checks are the repository's tools:
`bench_concurrent.py --alone --serial --tokens 256 --reps 1`, `reuse_served.py` (`record --planned --min-prompt
2048`, `replay`, `compare`, `cold`, `ids`), `bench_openai.py`, `bench_cache.py --require-cold`; here `media_http.py`,
`media_resume_http.py`, `kvp_admit.py` and `kvp_sum.py` (tools/; `kvp_sum.py conc` ends non-zero when a request
failed or a reply differed). `tf-cuda-test glm-paged-check CKPT RANK WORLD IDFILE` with `TF_GLM_LAYERS=5`. runs/ has
the chains' own output.

## Remaining coverage limits

- No request reached a 1M-token window's length: the floor's pages for one such request are in each 1M startup line
  ("enough for one at a 1048576-token window"), and `r6admit` shows the floor holding a 64k request at a 96k window.
- TP=3's ranks ran commit 20's engine source with `tp3/ring.patch` (above); TP=2's and one GPU's ran commit 20 alone.
- Ranks 1 and 2 of the TP=3 check print their PASS lines in the middle of NCCL's INFO lines (`NCCL_DEBUG=INFO`).
- Speed against main was measured at TP=2 only.
- `--kv-cache fp8` keeps windows: `auto` does not page and `on` and `all` are refused (`Fp8OnPages`).
- DFlash2 (`--drafter`): no weights here; the drafts on pages are the MTP head's.
- Metal and MLX were not run (no Mac).

## Publication handling

CRLF line ends made LF, trailing blanks dropped and an indent with a space before a tab made spaces (as `git diff
--check` asks); machine names, internal addresses, home and local paths, and the work's session, plan, records,
container and image names replaced with placeholders (`redaction-report.json` counts them a file; host-a to host-c
run TP=3's ranks 0 to 2, host-d and host-e TP=2's ranks 0 and 1); result values, token ids and token hashes kept.
The TP=3 server logs keep NCCL's INFO lines as they were written. The reuse records keep their rows without the
request bodies: each row has its body's sha256 and size, and `records-sha256.txt` the originals'. The speed runs'
prompt files are not included: `reuse_served.py cold` and `ids` make them again. SHA256SUMS pins every published
file except itself.
