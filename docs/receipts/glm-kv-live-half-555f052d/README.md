# GLM-5.3-Flash on CUDA: the live-half keep (`TF_GLM_KV_LIVE_HALF=1`) on the rebased head, served, receipt

Tested implementation: the binary built from `555f052d9`, upstream `main` at `ca854daf3` plus the change, after
#2162's L2 prefetch went on by default on GB10 (commit.txt has the commit and its tree). The base is the binary built
from `ca854daf3` itself. The receipt's commit adds this folder and changes no source.
Test date: 2026-11-07 (18:32:28 to 19:02:13 UTC, the nodes' production ring stopped and restored). Platform: two GB10
(sm_121, 128 GB shared memory) at TP=2 over a direct RoCE link, ranks pinned to CPUs 0-4,10-14, `--parallel 1`,
context 65,536, the prompt cache in pinned host memory (`TF_GLM_HOST_CACHE=1`, `--prompt-cache-gib 16`), drafts on
(MTP + copies) in every arm.
Toolchain: Zig 0.17.0, nvcc 13.0 for sm_121; binaries.txt has both binaries' sha256 (each equal on both ranks).
Checkpoint: `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (EXL3/TR3 4bpw, the recipe's checkpoint;
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` re-hosts it byte for byte), revision `a5fee929`.

## Scope

The same nine-turn greedy conversation as `../glm-kv-live-half-d3e06e76/` (six main turns, a seventh that keeps a
raw state, an eighth that restores it and a ninth that restores a later keep), drafts on, in four arms:

- `on`: the branch's binary, `TF_GLM_KV_LIVE_HALF=1`;
- `off`: the branch's binary, the variable unset;
- `fresh`: the branch's binary, the variable unset, `--prompt-cache-gib 0` (nothing kept, every turn prefilled whole);
- `main`: upstream `main`'s binary (`ca854daf3`), nothing set.

It does not qualify the change at other TP sizes, under concurrent streams, on paged sequences (`TF_KV_PAGES`), with
the page-sharing store (`TF_GLM_KV_SHARE`) or with `--kv-cache fp8`, as before.

## Results

Replies: identical in every turn across the four arms (compare.json), `fresh` included.

| turn | kind | prompt | cached on / off / fresh / main | TTFT s on / off / fresh / main | reply sha256 |
|---|---|---|---|---|---|
| 1 | main | 3102 | 0 / 0 / 0 / 0 | 5.91 / 5.88 / 4.51 / 5.82 | `e2ca39adac07` |
| 2 | main | 5322 | 2048 / 2048 / 0 / 2048 | 5.96 / 5.85 / 7.73 / 5.52 | `4be3fc8fb869` |
| 3 | main | 7544 | 4096 / 4096 / 0 / 4096 | 5.90 / 5.78 / 9.68 / 5.48 | `74622e0b72f3` |
| 4 | main | 9775 | 6144 / 6144 / 0 / 6144 | 6.38 / 6.34 / 13.19 / 6.01 | `adf747f9672f` |
| 5 | main | 11999 | 8192 / 8192 / 0 / 8192 | 6.67 / 6.59 / 14.76 / 6.09 | `66156c629643` |
| 6 | main | 14218 | 10240 / 10240 / 0 / 10240 | 6.77 / 6.67 / 17.96 / 6.18 | `7b52308dca65` |
| 7 | raw-keep | 16440 | 12288 / 12288 / 0 / 12288 | 6.31 / 6.32 / 20.72 / 5.72 | `33271190574e` |
| 8 | raw | 18661 | 16384 / 16384 / 0 / 16384 | 4.69 / 4.70 / 23.93 / 4.41 | `a4d24ccdbb04` |
| 9 | partial | 20881 | 18432 / 18432 / 0 / 18432 | 4.85 / 4.67 / 25.93 / 4.49 | `9cef66832f4c` |

`cached_tokens`: `on` equals `off` and `main` in every turn; `fresh` resumes nothing.

Drafts (records/*/turns.json: drafted / accepted a turn, 48 tokens a reply): identical in all four arms, 38/25, 40/28,
40/28, 36/31, 39/29, 42/29, 37/32, 38/25, 38/32. Fewer tokens accepted than drafted in every turn, so commit's
partial-accept replay ran throughout, on the live half and on the whole head alike.

Kept-state sizes, the engine's own accounting (rank 0's `prompt cache: ... states, N MiB` lines in
records/*/server-r0.log; MiB a state, the difference of consecutive lines):

| keep after turn | on | off | main | off - on |
|---|---|---|---|---|
| 1 | 126 | 194 | 194 | 68 |
| 2 | 164 | 232 | 232 | 68 |
| 3 | 190 | 258 | 258 | 68 |
| 4 | 214 | 282 | 282 | 68 |
| 5 | 240 | 308 | 308 | 68 |
| 6 | 266 | 334 | 334 | 68 |
| 7 | 316 | 384 | 384 | 68 |
| 8 | 342 | 410 | 410 | 68 |
| 9 | 368 | 436 | 436 | 68 |
| the nine states | 2226 | 2838 | 2838 | 612 |

Every `on` keep is 68 MiB smaller than its `off` twin: 34 KDA layers x 32 heads a rank x 128 x 128 x 4 bytes =
71,303,168 bytes = 68 MiB exactly, the dead half. `off` equals `main` state for state, so with the variable unset
the kept plan is main's. The totals line up to the byte as far as rank 0's MiB lines resolve (71,303,168 is a whole
number of MiB, and the nine cumulative totals differ by 68 x n MiB exactly).

Rank 1 runs no prompt cache of its own and prints no sizes: it follows rank 0's plan, whose head split is the same on
both ranks at TP=2 (32 KDA heads each), so its dead half is the same 71,303,168 bytes. That is from the plan, not
measured on rank 1. A `/metrics` sampler in validate2.sh read `tensorfold:prompt_cache_bytes`, which reports the
whole budget held with its spare buffers (14,938,013,696 bytes, constant), so it gives no per-state size and its
samples are not kept.

Time to first token: `on` within one run's noise of `off` and `main` in every turn (one run an arm). `fresh` grows
with the prompt (4.5 s to 25.9 s), as nothing resumes.

## Host tests

`zig build test-cuda-host` on the build node: 7 steps fail under the build runner at upstream `main` (`ca854daf3`) and
the same 7 on the branch, none added. Each of the branch's 7 failing test binaries, run directly, passes (2,552 tests,
0 failed), among them the GLM snapshot tests (21 of 21). The complete `zig build test` suite is the CI's.

## Reproduction

```
# on rank 0, the production ring stopped and restored by the script (both ranks run the same binaries, binaries.txt)
bash validate2.sh      # records/on, off, fresh, main; compare.json; production restored
```

Each arm: `start2.sh "PROBE_MODEL=$MODEL PROBE_BINARY=<binary> PROBE_CONTEXT=65536 TF_GLM_HOST_CACHE=1
[TF_GLM_KV_LIVE_HALF=1 | PROBE_PROMPT_CACHE_GIB=0]"` (launch2.sh on both ranks: `tensorfold-native serve $MODEL
--backend cuda --port 18080 --context 65536 --parallel 1 --prompt-cache-gib 16|0 --keep-warm 0 --idle-unload 0`), then
`python3 lazy_receipt.py run --config <arm> --lazy-ms 0 --pause 3` (the runner and its `served.py` are the fork's
`tools/kvprobe` scripts as run, one docstring line excepted; Python 3.11 or newer, the standard library), `collect`, and
`lazy_receipt.compare` over the four record folders. Placeholders in the scripts and logs: `RANK0_RING_IP`,
`RANK1_RING_IP`, `node0`, `node1`, `$MODEL`, `$HOME`, `$WORK` (the scripts' working directory), `$SERVE_DIR` (the serving stack the run stops and restores), `$NCCL_LIB_DIR` (the NCCL runtime's directory) and `$GPU_LOCK` (a
lock file that keeps other GPU jobs off the node) (redaction-report.json).

## Records

- `records/on/`, `records/off/`, `records/fresh/`, `records/main/`: `turns.json` (per-turn timings, draft counts and
  reply hashes), `log-summary.json`, the two ranks' server logs.
- `compare.json`: the four-way comparison. `validate2.log`: the run's console.
- `validate2.sh`, `start2.sh`, `stop2.sh`, `launch2.sh`, `lazy_receipt.py`, `served.py`: the scripts as run.
- Token ids and texts are not in the records.

## Publication handling

Private test on the team's two nodes; the records hold timings, hashes, counts and MiB only, scrubbed of the home
directory, the ring addresses, the node names and the local model path (redaction-report.json).
