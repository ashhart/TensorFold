# Receipt: full GLM-5.3 conversation states on the core's disk tier (slice (c))

Full GLM-5.3 (`glm_moe_dsa`) at TP=3 keeps a chat's kept states on disk across a restart with
`--prompt-cache-disk`, and a resumed turn equals the cache-off reference. This is slice (c) of
#3063 on top of the merged (b), against #1411's disk tier.

- Tested implementation: `6ec94e2a7` (the binary in `binaries.txt`). Code commit on this branch: `91e4975cf3`, rebased onto main `c464c9128b`. Four of the five files match the tested commit; `cuda.zig` also has main's env-reader migration (`2b76e782fb`), which is not on the disk path. `commit.txt`.
- Checkpoint `Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold` revision `2d747d0e30…faa83`, all 146
  files' sha256 in `checkpoint-sha256.txt`, identical on the three hosts.
- Servers and the test as written in `launch.md`; BF16 cache, `--context 32768 --parallel 2`,
  drafts on.

## Result (2026-11-27, `logs/learn.log`)

| Step | Value |
|---|---|
| turn 1 on disk | 22,528 tokens kept: 88 new blocks, 2,120 MiB this state (2,046 MiB shared rows), written in 13.738 s |
| restart, turn 2 read | 2,120 MiB in 2.729 s; 2.1 GiB on disk |
| turn 2 resume | `prompt 23307 cached 22528` |
| turn 1 disk vs fresh | equal (sha `37287f2c420f…`) |
| turn 2 disk vs fresh | equal (sha `af785ef933c5…`); cached 22528 vs 0 |
| verdict | PASS |

A kept state's image is the state's own bytes: the 576-wide latent row on the 78 latent caches
and the pooled keys of the 21 indexed layers only (a shared indexer layer's cache holds latent
rows alone), so the read-back needs no recomputation and the resumed reply is token-identical to
the fresh one.

## Host side (on the PR head)

- Focused GLM suite: `25/25 steps succeeded; 100/100 tests passed` (`-Dfixed-cpu -j4`), on the tested commit and again on the previous rebase (`9fd37df44`), whose five files matched the tested commit. Not repeated after the rebase onto `c464c9128b`.
- `check-native-cuda-host compile-cuda-tests -Dtarget=aarch64-linux-gnu`: `21/21 steps succeeded`.
- New host test: the image against the snapshot plan at four positions and every head length;
  rows by position are the 78 caches' latent rows and the 21 indexed layers' pooled keys.
