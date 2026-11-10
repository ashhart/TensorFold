# GLM-5.3-Flash on CUDA: the live-half keep (`TF_GLM_KV_LIVE_HALF=1`), served, receipt

Tested implementation: the binaries built from `d3e06e76…`, upstream `main` at `ad98b602c` plus the change (commit.txt
has the commit and its tree). The receipt's commit adds this folder and a recipe paragraph and changes no source.
Rebased: onto upstream main 8abb30f6f as 0119e4893, then onto 29d6e8b64 as f43f279d2; the rebased tree differs from
the tested one by the fingerprint reconciliation only (layout 20 as assigned in the review, kv_live_half placed after
upstream's kda_pad), the changelog line's position and the recipe bullet's fresh context (upstream's KV-pages paragraph
now sits above it)
Test date: 2026-11-03 (06:13:22 to 06:39:57 UTC, the nodes' production ring stopped and restored). Platform: two GB10
(sm_121, 128 GB shared memory) at TP=2 over a direct RoCE link, ranks pinned to CPUs 0-4,10-14, `--parallel 1`,
context 65,536, the prompt cache in pinned host memory (`TF_GLM_HOST_CACHE=1`, `--prompt-cache-gib 16`).
Toolchain: Zig 0.17.0, nvcc 13.0 for sm_121; binaries.txt has the binary's sha256 (equal on both ranks).
Checkpoint: Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit (EXL3 4bpw), revision `57edefd2`.

## Scope

A kept prompt state carries both halves of the double-buffered KDA recurrent state (`rec`, fp32
[2][layers][heads][128][128], 142.6 MB a rank for GLM-5.3-Flash at TP=2). The dead half is the state before the last
prompt chunk or decode window: every chunk and every committed window reads `recAt(cur)`, writes `recAt(1 - cur)`
and flips the layer's bit, and a keep waits for an empty window, so nothing reads the dead half after a restore.
With `TF_GLM_KV_LIVE_HALF=1` the snapshot plan carries the arena head in two parts around the dead half, a restore
builds its plan from the kept state's own parity and leaves the slot's dead half as it was, and the ranks'
fingerprint carries the flag (layout 16 in the tested binary, 20 on the branch as the maintainer assigned, after upstream took 16 for `kda_pad`; ranks that disagree stop before loading).

This qualifies the change served at TP=2: the whole head (`both`, the flag at 0) against the live half (`live`),
with drafts off and then on (`both-drafts`, `live-drafts`: the MTP head and copies), the same binary, one growing
greedy conversation of nine turns each (six main turns, a seventh that keeps a raw state, an eighth that restores it
and a ninth that restores a later keep), replies byte for byte, the prompt cache's own per-state sizes, draft
statistics and time to first token. It does not qualify the change at other TP sizes, under concurrent streams, on
paged sequences (`TF_KV_PAGES`), with the page-sharing store (`TF_GLM_KV_SHARE`) or with `--kv-cache fp8`; the head's
split is the same code for them, untested here.

## Results

Replies: identical in every turn across the four configurations (compare.json; the hashes also equal the fork's
earlier runs of this conversation on an older base, which are not part of this receipt).

| turn | kind | prompt | cached | TTFT s both / live / both-drafts / live-drafts | total s both / live / both-drafts / live-drafts | reply sha256 |
|---|---|---|---|---|---|---|
| 1 | main | 3102 | 0 | 5.80 / 5.78 / 5.28 / 5.24 | 11.62 / 11.50 / 8.40 / 8.39 | `ea582398230c` |
| 2 | main | 5322 | 2048 | 5.57 / 5.52 / 5.56 / 5.70 | 11.35 / 11.24 / 8.43 / 8.53 | `25d3cf4c723b` |
| 3 | main | 7544 | 4096 | 5.63 / 5.65 / 5.59 / 5.55 | 11.56 / 11.40 / 9.25 / 9.01 | `4bbd2f8df0b6` |
| 4 | main | 9775 | 6144 | 6.15 / 6.08 / 6.14 / 6.13 | 11.93 / 11.73 / 10.08 / 9.96 | `0dbcac0b9d7a` |
| 5 | main | 11999 | 8192 | 6.19 / 6.20 / 6.29 / 6.32 | 11.89 / 11.94 / 9.64 / 9.54 | `73e33772a797` |
| 6 | main | 14218 | 10240 | 6.27 / 6.22 / 6.13 / 6.22 | 11.98 / 11.93 / 10.25 / 10.15 | `1c313dc54dee` |
| 7 | raw-keep | 16440 | 12288 | 7.18 / 6.97 / 6.97 / 7.11 | 13.03 / 12.72 / 9.61 / 9.70 | `450d3bec5a69` |
| 8 | raw | 18661 | 16384 | 4.47 / 4.43 / 4.44 / 4.43 | 10.39 / 10.21 / 9.07 / 8.78 | `6d615174db62` |
| 9 | partial | 20881 | 18432 | 4.53 / 4.44 / 4.46 / 4.55 | 10.30 / 10.28 / 7.84 / 7.69 | `921fc4b0e3f2` |

Drafts (records/*/turns.json: drafted / accepted / rounds a turn, 48 tokens a reply): identical between `both-drafts`
and `live-drafts` in every turn, 45/30/17, 39/31/16, 41/26/21, 41/24/23, 42/29/18, 43/24/24, 38/33/14, 35/20/28,
44/29/18; with drafts off, 47 rounds a turn. Fewer tokens accepted than drafted in every turn means commit's
partial-accept replay (`keep < pending`: it reads `recAt(cur)` and the window's saved rows and writes `recAt(1 - cur)`)
ran throughout both drafts-on runs, on the whole head and on the live half alike.

Kept-state sizes, the engine's own accounting (rank 0's `prompt cache: ... states, N MiB` lines in records/*/server-r0.log,
MiB a state; rank 1 prints none, its plans have the same shape):

| keep after turn | both | live | both-drafts | live-drafts |
|---|---|---|---|---|
| 1 | 190 | 122 | 194 | 126 |
| 2 | 226 | 158 | 232 | 164 |
| 3 | 248 | 180 | 258 | 190 |
| 4 | 272 | 204 | 282 | 214 |
| 5 | 296 | 228 | 308 | 240 |
| 6 | 318 | 250 | 334 | 266 |
| 7 | 366 | 298 | 384 | 316 |
| 8 | 388 | 320 | 410 | 342 |
| 9 | 412 | 344 | 436 | 368 |
| the nine states | 2716 | 2104 | 2838 | 2226 |

Every live keep is 68 MiB smaller than its whole-head twin: 34 KDA layers x 32 heads a rank x 128 x 128 x 4 bytes =
71,303,168 bytes, the dead half. Over the conversation's nine states the prompt cache holds 22.5 % fewer MiB with
drafts off and 21.6 % fewer with drafts on (whose states carry the MTP layer's caches, 4 to 24 MiB more each).

Reading: the dead half was the one part of a kept state that carried bytes no restore reads; leaving it out of the
plan removes 68 MiB a rank from every keep's pages with every reply, every draft statistic and every restore
unchanged, drafts off and on. Time to first token is within one run's noise of the whole head in every turn
(differences under 0.25 s with no consistent sign; one run per configuration), as expected of a copy that shrinks
by 68 MiB at pinned-memory bandwidth. The flag stays off by default.

## Reproduction

```
# on rank 0, the production ring stopped by the script (both ranks run the same binary, binaries.txt)
bash validate-pr.sh          # records/both, live, both-drafts, live-drafts; compare.json; production restored
```

Each configuration: `start-ranks-pr.sh "PROBE_CONTEXT=65536 TF_GLM_HOST_CACHE=1 TF_GLM_KV_LIVE_HALF=0|1" [--no-drafts]`
(launch-pr.sh on both ranks: `tensorfold-native serve $MODEL --backend cuda --port 18080 --context 65536 --parallel 1
--prompt-cache-gib 16 --keep-warm 0 --idle-unload 0`), then `python3 lazy_receipt.py run --config <label> --lazy-ms 0
--pause 3` (the runner and its `served.py` are the fork's `tools/kvprobe` scripts as run; Python 3.11 or newer, the
standard library), `collect`, and `lazy_receipt.compare` over the four record folders. Placeholders in the scripts
and logs: `RANK0_RING_IP`, `RANK1_RING_IP`, `node0`, `node1`, `$MODEL`, `$HOME`, `$WORK` (the scripts' working directory), `$SERVE_DIR` (the serving stack the run stops and restores), `$NCCL_LIB_DIR` (the NCCL
runtime's directory) and `$GPU_LOCK` (a lock file that keeps other GPU jobs off the node) (redaction-report.json).

## Remaining coverage limits

- TP=2, one stream, no paged sequences, no page-sharing store, BF16 latent cache only.
- Rank 1's log prints no prompt-cache sizes; rank 0's are reported.
- Host tests: `zig build test-cuda-host` passed on the build node at upstream `main` and on the branch (`zig fmt
  --check` clean on the changed files); a failure planted in the new layout test was reported by name (one of 2,289
  tests) and the restored tree passed again. The complete `zig build test` suite is the CI's.

## Records

- `records/both/`, `records/live/`, `records/both-drafts/`, `records/live-drafts/`: `turns.json` (per-turn timings,
  draft counts and reply hashes), `log-summary.json`, the two ranks' server logs.
- `compare.json`: the four-way comparison. `validate-pr.log`: the run's console.
- `validate-pr.sh`, `start-ranks-pr.sh`, `stop-ranks.sh`, `launch-pr.sh`, `lazy_receipt.py`, `served.py`: the scripts
  as run.
- Token ids and texts are not in the records.

## Publication handling

Private test on the team's two nodes; the records hold timings, hashes, counts and MiB only, scrubbed of the home
directory, the ring addresses, the node names and the local model path (redaction-report.json).
