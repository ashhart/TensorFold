# XPU Qwen3.8-27B: a lone DFlash2 stream picks its tree's width from the lane rule

Receipts for the XPU lone stream (`--parallel 1` with the DFlash2 drafter). Main verified a fixed 15-node tree every
round; the head asks the core's lone-window pricing (`lone_costs`, with `allocateDrafts`) how many nodes to verify,
from the stream's own measured rounds, a context band each. The price of a width is a line through the widths sampled
so far (lone XPU tree only; the engine keeps main's pricing). A stream files its calibration when it ends
(`dflash2.retain`), so the next request of its band starts priced. `XPU_TREE` stays a cap. The width only sets how
many drafts are verified: the walk keeps the target's own pick, so every reply is the same.
Measured on 2026-11-22, from 09:45 to 12:12 local time, against `main` at `7fb9fc661`.

## Builds and hardware

- Bases: `main` at `7fb9fc661`; the head's code at `c36b7391e` (the branch on `7fb9fc661` through merge `059bf7f06`,
  plus the retain fix). The table's head binary was built from a tree that differs from `c36b7391e` by the wording of
  one doc comment. No local patch: `git status` was clean in the trees, and `zig/src/xpu/rt.zig` is main's.
- Zig 0.17.0. Binaries: `zig build -j4 -Dxpu xpu-native -Doptimize=ReleaseFast -Dxpu-dpas=true -Dngen-include=<nGEN>
  --prefix <dir>` in each tree.
- `tensorfold-native` SHA-256, main: `ab8032c551ab51547a65c5a1cd3c6aec19ae68da2cd07cf88daf2b178be0239c`
  (MD5 `b57528c19275044fc67303c834baeb8b`); head: `5b196084989fe0d141d2a342db1cb1db904b17a28ca8d0f2fbad2b712085b3e4`
  (MD5 `2b0ab16ab515b81d4457ff12bf303c6a`).
- Card: Intel Arc Pro B70 (BMG-G31, PCI id `0xe223`), the host's card 0 (`ZE_AFFINITY_MASK=0`), on a PCIe 5.0 x16 slot
  wired to the CPU. The host has a second B70, which no run used. One card only.
- Host: Intel Core Ultra 7 270K Plus, 30 GiB RAM, Ubuntu 26.04.1, Linux 7.0.0-43-generic.
- Driver (`dpkg -l`): `intel-igc-core-2` 2.41.5, `intel-igc-opencl-2` 2.41.5, `intel-ocloc` 26.35.39758.21,
  `intel-opencl-icd` 26.35.39758.21, `libze-intel-gpu1` 26.35.39758.21, `libze1` 1.32.0, `libigdgmm12` 22.10.2.
- Checkpoints: `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` and its `Qwen3.8-27B-DFlash2` drafter, local copies.
  Hugging Face revisions from each copy's download metadata: target `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`,
  drafter `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.

## Protocol

One fresh `tensorfold-native serve` per pass (`--parallel 1 --context 132096 --prompt-cache-gib 0`, bf16 KV, the
DFlash2 drafter). Main runs fixed trees through `XPU_TREE` (15, 7, 3); the head runs with no `XPU_TREE`. Passes, in
`scripts/driver.sh`'s order: main 15, main 3, main 7, main 3 again (58,543 and 121,022 tokens only), main 15 again
(34 tokens only), the head (`scripts/run_pass.sh`, with timeouts). `scripts/bench_tree.py` runs code prompts of 34,
28,946, 58,543 and 121,022 tokens (the first lines of `corp_code`, see Reproduce), greedy temperature 0, 256 new
tokens, thinking off, one warm-up request and three measured ones per cell. The first passes of main 15, main 3 and the
head also run a seeded sampled cell (temperature 1.0, top_k 20, top_p 0.95, seed 7) at 34 and 121,022 tokens. Every
request records the server's `tensorfold.token_sha256` and its `speculative` object. A cell's speed is the median of
its three measured requests; the warm-up request (rep 0) is excluded and listed below.

## Exactness: same tokens at every width

`analyse.out` (from `scripts/analyse.py runs`): each cell's requests carry one `token_sha256`, across main's fixed trees
of 15, 7 and 3 nodes and the head's chosen widths (the plain decode mode was not run here).

| prompt tokens | greedy requests | distinct hashes | seeded requests (main 15, main 3, head) | distinct hashes |
|---|---|---|---|---|
| 34 | 20 | 1 | 12 | 1 |
| 28,946 | 16 | 1 | | |
| 58,543 | 20 | 1 | | |
| 121,022 | 20 | 1 | 12 | 1 |

`CELLS WITH MORE THAN ONE HASH: 0`.

## Speed

Decode tok/s, served, one stream: the median of a pass's three measured requests, the mean of the variant's passes
(each pass's median in brackets when a cell has two). The head's nodes a round (`drafted / rounds`) in the last column.

| prompt tokens | main, 15 nodes | main, 7 | main, 3 | head (default) | head, nodes a round |
|---|---|---|---|---|---|
| 34 | 140.45 (140.46, 140.43) | 132.10 | 91.45 | 140.37 | 12.99 |
| 28,946 | 36.34 | 47.51 | 52.82 | 53.01 | 4.13 |
| 58,543 | 25.71 | 36.75 | 43.83 (43.83, 43.83) | 44.67 | 3.38 |
| 121,022 | 13.30 | 21.43 | 27.54 (27.54, 27.53) | 29.52 | 2.32 |

Seeded sampling, one pass each:

| prompt tokens | main, 15 nodes | main, 3 | head (default) | head, nodes a round |
|---|---|---|---|---|
| 34 | 153.98 | 102.15 | 159.10 | 12.00 |
| 121,022 | 14.09 | 27.46 | 28.92 | 2.51 |

The head against the best fixed tree (unrounded means; seeded: the best of main 15 and main 3):

| prompt tokens | best fixed | head / best fixed | head / best fixed, seeded |
|---|---|---|---|
| 34 | 15 nodes, 140.45 | 0.999 | 1.033 |
| 28,946 | 3 nodes, 52.82 | 1.003 | |
| 58,543 | 3 nodes, 43.83 | 1.019 | |
| 121,022 | 3 nodes, 27.54 | 1.072 | 1.053 |

- The three requests of a main cell differ by 0.1 % or less; the head's by 2.2 to 5.0 % (greedy) and up to 3.3 %
  (seeded). Main's repeat passes (3 at 58,543 and 121,022 tokens, 15 at 34) reproduce the first by 0.1 % or less.
- At 34 tokens the head matches the 15-node tree. The best fixed tree is 3 nodes from 28,946 tokens on, and the head's
  lead over it grows with the context.
- The first request of a context band on a fresh server is still cold, as designed: the band's prices are not sampled
  yet (rep 0 below: 121.46 tok/s at 34 tokens, 48.26 at 28,946, 24.34 at 121,022). Later requests of the band start
  from the calibration the previous one filed.

The head's greedy requests, one by one (rep 0 is the warm-up, which also runs width trials):

| prompt tokens | rep | nodes a round | tok/s |
|---|---|---|---|
| 34 | 0 | 11.49 | 121.46 |
| 34 | 1 | 13.49 | 141.51 |
| 34 | 2 | 13.02 | 140.37 |
| 34 | 3 | 12.45 | 135.38 |
| 28,946 | 0 | 5.02 | 48.26 |
| 28,946 | 1 | 4.14 | 53.60 |
| 28,946 | 2 | 4.28 | 52.43 |
| 28,946 | 3 | 3.98 | 53.01 |
| 58,543 | 0 | 4.88 | 39.97 |
| 58,543 | 1 | 3.52 | 43.14 |
| 58,543 | 2 | 3.19 | 45.33 |
| 58,543 | 3 | 3.42 | 44.67 |
| 121,022 | 0 | 3.79 | 24.34 |
| 121,022 | 1 | 2.49 | 28.65 |
| 121,022 | 2 | 2.24 | 29.52 |
| 121,022 | 3 | 2.22 | 30.13 |

## First request on a fresh server (review of 2026-11-24)

The review asked for the first request of a fresh server, with no earlier request, at 34 and 7,588 prompt tokens
(28,946 and 121,022 added), against fixed trees of 15, 7 and 3 nodes. This section supersedes the remarks above about
the first request of a band (rep 0 of the warm-up runs); the sections above are unchanged.

**Policy.** A band whose calibration nothing has filed yet (`lone_kept` empty) keeps the 15-node tree through the
request, with no width trial, when the prompt is under 2,048 tokens (the drafter's window plus one,
`cold_context` in `xpu_dflash2.zig`); its 15-node rounds are the calibration the request files when it ends. From 2,048
prompt tokens on, a cold band probes widths as before. A band that has a calibration is priced from it as before. So a
fresh server's first short request draws exactly the fixed 15-node tree, and a deep first request still adapts.

**Builds.** Head: `599d8a56e` (this branch, with main `ef0ecb9a6` merged), SHA-256
`97a02ea19447c3d178af6bf8d7dc260360e310e7e09c009cd83d32973eee33c8`. Main: `ef0ecb9a6`, SHA-256
`d78347faa767d6b4072a14f7d39d752a5e3035b1f4fc4f1a8b38009cbc0155f4`, run with `XPU_TREE` 15, 7 and 3. Same build flags,
Zig, card (the host's card 0, PCIe 5.0 x16), checkpoints and server flags as above; no local patch. The host's other
card ran an unrelated server during the run. Measured on 2026-11-24, 12:57 to 13:52 local time.

**Protocol.** 48 cells: 4 prompts x 4 arms (head, main 15, main 7, main 3) x 3 fresh servers. A cell is one fresh
`tensorfold-native serve` (`--parallel 1 --context 132096 --prompt-cache-gib 0`), one request (greedy, 256 new tokens,
thinking off, the request of `scripts/bench_tree.py`), then the server is stopped and the page cache dropped.
The request is the server's first: no warm-up, no calibration request. The prompts are cells 0, 8192, 32768 and
131072 of `bench_tree.py` (`prompt_sha256` `cf916455e20cf240`, `829dddadac0f329f`, `0bd40816987651da`,
`60986a50b4990e56`; the
7,588- and 121,022-token ones are also those of the `xpu-qwen27-attn-exp` receipt). Within a prompt the arm order is
head, 15, 7, 3 for server 1, reversed for server 2, forward again for server 3; the prompts run in the order of the
tables below. Speed is decode tok/s as `bench_tree.py` defines it ((tokens - 1) over the time from the first token to
the end); nodes a round is `drafted / rounds`; TTFT is the time to the first content chunk.

#### 34 prompt tokens

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 128.98, 126.98, 126.71 | 126.98 | 127.56 | 15.00 | 0.09 | 1 |
| fixed 15 | 3 | 127.59, 128.82, 126.73 | 127.59 | 127.71 | 15.00 | 0.09 | 1 |
| fixed 7 | 3 | 120.21, 120.20, 119.87 | 120.20 | 120.09 | 7.00 | 0.09 | 1 |
| fixed 3 | 3 | 85.11, 85.16, 84.77 | 85.11 | 85.01 | 3.00 | 0.09 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 15, 127.59): 0.995 on the medians (BELOW); the slowest adaptive server is 0.993 of it.
Adaptive / fixed 15 (head against main's default): 0.995.

#### 7,588 prompt tokens

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 65.61, 65.87, 65.53 | 65.61 | 65.67 | 7.75 | 4.01 | 1 |
| fixed 15 | 3 | 59.14, 59.14, 59.22 | 59.14 | 59.17 | 15.00 | 4.01 | 1 |
| fixed 7 | 3 | 67.28, 67.29, 67.32 | 67.29 | 67.30 | 7.00 | 4.01 | 1 |
| fixed 3 | 3 | 57.98, 58.02, 58.01 | 58.01 | 58.01 | 3.00 | 4.01 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 7, 67.29): 0.975 on the medians (BELOW); the slowest adaptive server is 0.974 of it.
Adaptive / fixed 15 (head against main's default): 1.109.

#### 28,946 prompt tokens

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 47.86, 47.81, 47.82 | 47.82 | 47.83 | 5.12 | 16.95 | 1 |
| fixed 15 | 3 | 35.39, 35.37, 35.36 | 35.37 | 35.37 | 15.00 | 16.98 | 1 |
| fixed 7 | 3 | 45.74, 45.73, 45.72 | 45.73 | 45.73 | 7.00 | 16.96 | 1 |
| fixed 3 | 3 | 50.37, 50.33, 50.33 | 50.33 | 50.34 | 3.00 | 16.96 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 3, 50.33): 0.950 on the medians (BELOW); the slowest adaptive server is 0.950 of it.
Adaptive / fixed 15 (head against main's default): 1.352.

#### 121,022 prompt tokens

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 23.60, 23.56, 23.55 | 23.56 | 23.57 | 3.53 | 101.26 | 1 |
| fixed 15 | 3 | 13.12, 13.12, 13.13 | 13.12 | 13.13 | 15.00 | 101.27 | 1 |
| fixed 7 | 3 | 20.92, 20.92, 20.92 | 20.92 | 20.92 | 7.00 | 101.29 | 1 |
| fixed 3 | 3 | 26.87, 26.89, 26.89 | 26.89 | 26.88 | 3.00 | 101.29 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 3, 26.89): 0.876 on the medians (BELOW); the slowest adaptive server is 0.876 of it.
Adaptive / fixed 15 (head against main's default): 1.795.

**Reading.**
- Every reply has one `token_sha256` per prompt across all 12 cells of the prompt: exactness holds at every width.
- 34 tokens: the head draws 15.00 nodes a round in all three servers, as fixed 15 does; the two are the same
  computation, and their 0.995 is the spread of a first request (the head's servers span 126.71 to 128.98 tok/s,
  fixed 15 alone 126.73 to 128.82).
- 7,588 tokens: the head is 1.109 times fixed 15 and 0.975 times the best fixed tree (7 nodes). Its first request probes
  widths (7.55 to 7.92 nodes a round) that the fixed 7 does not pay for.
- 28,946 and 121,022 tokens: the head is 1.352 and 1.795 times fixed 15, and 0.950 and 0.876 times the best fixed tree
  (3 nodes). The first request pays for its width trials (5.12 and 3.53 nodes a round against 3); the warmed head of the
  Speed section beats the best fixed tree at these depths.
- So the first-request criterion "at or above the best fixed tree" is met at 34 tokens up to the spread, and is not met
  from 7,588 tokens on; a cold band's probing is the price of learning its widths.

## First request with a context prior (2026-11-24, second series)

The first series above shows why a prior was added: a cold band's first request paid for its width trials and ended
0.975, 0.950 and 0.876 of the best fixed tree at 7,588, 28,946 and 121,022 prompt tokens (0.995 at 34). This series
measures the change that answers it.

**Policy.** A band with no filed calibration (`lone_kept` empty) draws a fixed number of nodes, chosen once from the
prompt length when the stream starts, through the whole request, with no width trial (`prior` in `xpu_dflash2.zig`):

| prompt tokens | nodes | the edge |
|---|---|---|
| under 2,048 | 15 (the grown tree) | `cold_context`, the drafter's window plus one |
| 2,048 to 8,191 | 7 | band 0 of `lone_costs.band` |
| 8,192 and more | 3 | bands 1 and up |

The request's rounds are timed and filed as before, so the band's next request is priced from them, probes widths and
adapts as in the sections above. A smaller `XPU_TREE` cap wins over the prior (the tree is grown at the cap and cut to
at most its nodes). A tree of w nodes is the first w pops of the 15-node tree, so a prior of 7 or 3 draws what
`XPU_TREE=7` or `XPU_TREE=3` draws. The numbers come from the first series and the earlier sections' best fixed trees
(15 at 34 tokens, 7 at 7,588, 3 at 28,946 and 121,022). Not measured: where the best fixed tree changes from 7 to 3
between 8,192 and 28,946 tokens, and the 2,048 edge between 15 and 7 (34 and 7,588 tokens are the only short cells); the
prior is a decision on the data here, not a fit.

**Builds.** Head: `48e6f4d89` (this branch with main `683c66fe3` merged), SHA-256
`df12ffcbc9a85322d2718faad60ca4c4e0a5d29f3913e75e1d0fc0b9bdacb3af`. Main: `683c66fe3`, SHA-256
`f88d5603bb8e4bc2f13a7bb3a1d173da4117cedf3b8679a9affdc415c10c1fca`, run with `XPU_TREE` 15, 7 and 3. Same build flags,
Zig, card (the host's card 0, PCIe 5.0 x16), checkpoints and server flags as above; no local patch; the host's other
card idle. Measured on 2026-11-24, 14:11 to 15:05 local time.

**Protocol.** The first series' protocol, unchanged: 48 cells, 4 prompts x 4 arms x 3 fresh servers, one request each,
the server's first, no warm-up, alternating arm order. The driver is `scripts/cold/series.sh` as published here, with
its environment variables set at launch. A smoke on the host's other card before the series gave 15.00, 7.00 and 3.00
nodes a round at 34, 7,588 and 28,946 tokens.

#### 34 prompt tokens

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 127.06, 126.93, 126.69 | 126.93 | 126.89 | 15.00 | 0.09 | 1 |
| fixed 15 | 3 | 127.15, 127.21, 126.96 | 127.15 | 127.11 | 15.00 | 0.09 | 1 |
| fixed 7 | 3 | 119.90, 119.84, 119.95 | 119.90 | 119.89 | 7.00 | 0.09 | 1 |
| fixed 3 | 3 | 84.84, 84.92, 84.84 | 84.84 | 84.87 | 3.00 | 0.09 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 15, 127.15): 0.998 on the medians (BELOW); the slowest adaptive server is 0.996 of it.
Adaptive / fixed 15 (head against main's default): 0.998.

#### 7,588 prompt tokens

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 67.11, 67.16, 67.16 | 67.16 | 67.14 | 7.00 | 4.01 | 1 |
| fixed 15 | 3 | 59.15, 59.16, 59.21 | 59.16 | 59.17 | 15.00 | 4.01 | 1 |
| fixed 7 | 3 | 67.31, 67.21, 67.28 | 67.28 | 67.26 | 7.00 | 4.01 | 1 |
| fixed 3 | 3 | 58.03, 57.98, 57.98 | 57.98 | 58.00 | 3.00 | 4.01 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 7, 67.28): 0.998 on the medians (BELOW); the slowest adaptive server is 0.998 of it.
Adaptive / fixed 15 (head against main's default): 1.135.

#### 28,946 prompt tokens

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 50.08, 50.09, 50.09 | 50.09 | 50.09 | 3.00 | 16.94 | 1 |
| fixed 15 | 3 | 35.42, 35.40, 35.38 | 35.40 | 35.40 | 15.00 | 16.96 | 1 |
| fixed 7 | 3 | 45.77, 45.75, 45.73 | 45.75 | 45.75 | 7.00 | 16.97 | 1 |
| fixed 3 | 3 | 50.32, 50.33, 50.31 | 50.32 | 50.32 | 3.00 | 16.96 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 3, 50.32): 0.995 on the medians (BELOW); the slowest adaptive server is 0.995 of it.
Adaptive / fixed 15 (head against main's default): 1.415.

#### 121,022 prompt tokens

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 26.85, 26.86, 26.86 | 26.86 | 26.85 | 3.00 | 101.30 | 1 |
| fixed 15 | 3 | 13.13, 13.13, 13.13 | 13.13 | 13.13 | 15.00 | 101.24 | 1 |
| fixed 7 | 3 | 20.93, 20.92, 20.93 | 20.93 | 20.93 | 7.00 | 101.28 | 1 |
| fixed 3 | 3 | 26.89, 26.89, 26.89 | 26.89 | 26.89 | 3.00 | 101.25 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 3, 26.89): 0.999 on the medians (BELOW); the slowest adaptive server is 0.998 of it.
Adaptive / fixed 15 (head against main's default): 2.045.

**Reading.**
- One `token_sha256` per prompt across its 12 cells, as in the first series.
- The head draws exactly 15.00, 7.00, 3.00 and 3.00 nodes a round in every server, the same trees the best fixed
  trees draw, and 45, 77, 95 and 95 rounds, so the first request is the best fixed tree's computation.
- The head's first request is 0.998, 0.998, 0.995 and 0.999 of the best fixed tree on the medians (first series: 0.995,
  0.975, 0.950, 0.876), and 0.998, 1.135, 1.415 and 2.045 of fixed 15. It is not at or above the best fixed tree on any
  prompt: it stays 0.1 to 0.5 % under it, on every server, with the same nodes and rounds. Equal trees point to the
  lone path's per-round work (the width choice, the timing, the synchronize after each round) rather than to the
  tree; this series does not isolate it. Between servers the head's tok/s spread is under 0.7 %, the fixed trees'
  under 0.5 %.

## First request with a context prior, grown to the prior (2026-11-25, third series)

The second series left the head 0.1 to 0.5 % under the best fixed tree with the same nodes and rounds. Per-round host
timers (a local patch, not committed) on fresh servers at 28,946 tokens, 95 rounds, in ms a round:

| part of the lone round | before | after |
|---|---:|---:|
| grow (drafter forward and host tree) | 7.22, 7.24 | 6.99, 6.99 |
| width choice | 0.0003 | 0.0002 |
| settle (verify, commit, taps) | 45.94, 45.95 | 45.97, 45.98 |
| synchronize after settle, for the round's time | 0.33 | 0.33 |
| observe and price (`observeTree`, `measured`) | under 0.006 | under 0.006 |

The cost was in grow: a cold band grew the host's best-first tree to all 15 nodes every round, then cut it to the
prior's 3. `14ed3f117` grows only `min(prior, cap)` nodes while the band is cold. The tree is the same prefix in pop
order, so the drawn tree and the tokens do not change. Dropping the synchronize instead was slower (50.06 and 50.12
tok/s, against 50.17 and 50.14): its 0.33 ms moved into the next wait, so it stays. The same head binary forced down the
fixed path ran 50.39 and 50.35, main's `XPU_TREE=3` 50.38 and 50.37.

**Builds.** Head: `dca272507`, SHA-256 `2eca3a73bf6769712ba6ae4e686a6660f4c6b4358de606b1fb6fa8265580a370`. Main:
`683c66fe3`, the second series' binary. Same flags, card, checkpoints, server flags and protocol as the second series
(48 cells, `scripts/cold/series.sh`). Measured on 2026-11-25, 09:52 to 10:46 local time.

| prompt | adaptive, servers 1..3 | best fixed tree, servers 1..3 | adaptive / best fixed | adaptive / fixed 15 |
|---:|---|---|---:|---:|
| 34 | 127.13, 127.04, 126.84 | 15: 127.25, 126.78, 127.24 | 0.998 | 0.998 |
| 7,588 | 67.31, 67.22, 67.28 | 7: 67.34, 67.29, 67.30 | 1.000 | 1.137 |
| 28,946 | 50.31, 50.36, 50.37 | 3: 50.26, 50.34, 50.38 | 1.000 | 1.422 |
| 121,022 | 26.89, 26.90, 26.90 | 3: 26.92, 26.91, 26.91 | 0.999 | 2.046 |

**Reading.**
- One `token_sha256` per prompt across its 12 cells; 15.00, 7.00, 3.00 and 3.00 nodes a round; 45, 77, 95 and 95 rounds.
- At 34, 7,588 and 28,946 tokens the adaptive and best fixed servers' ranges overlap. At 121,022 they do not: the
  adaptive servers run 26.89 to 26.90 tok/s, the fixed 3 servers 26.91 to 26.92, 0.05 % apart.
- Not changed: once a band is calibrated, the width choice needs the 15 grown nodes' chances, so a calibrated band at a
  narrow width still grows 15 nodes a round (about 0.2 ms, 0.4 % of a 28,946-token round). Growing only up to the last
  width would limit which widths can be reached; it is left out. Full tables: `runs/cold-grow/summary.md`.

## First request, the first grow at the prior (2026-11-27, fourth series)

The third series' head still grew the full cap on a cold stream's very first round: `width()`, which sets the band's
prior, ran after `D.grow`. `3d7343730` seeds the prior first (`seed()`), so the first round grows 15, 7 or 3 nodes as
the prompt's band says; a host test drives `lone()` through a fake drafter and checks the first and second grow, and a
calibrated band's next stream growing the cap. Same main binary (`683c66fe3`), card, checkpoints, server flags and
protocol as the second and third series (48 cells, `scripts/cold/series.sh`); head SHA-256 `a1684d1b4c1f...` (built at
`74e3c0786`, the same code as `3d7343730`, whose test file adds the calibrated-band case). Measured on 2026-11-27,
11:19 to 12:10 local time. Full tables: `runs/cold-seed/summary.md`.

| prompt | adaptive, servers 1..3 | best fixed tree, servers 1..3 | adaptive / best fixed |
|---:|---|---|---:|
| 34 | 127.01, 126.94, 126.70 | 15: 127.18, 127.18, 127.12 | 0.998 |
| 7,588 | 67.24, 67.23, 67.16 | 7: 67.32, 67.33, 67.33 | 0.999 |
| 28,946 | 50.35, 50.32, 50.33 | 3: 50.35, 50.34, 50.37 | 1.000 |
| 121,022 | 26.90, 26.86, 26.88 | 3: 26.88, 26.89, 26.88 | 1.000 |

One `token_sha256` per prompt across its 12 cells. At 28,946 and 121,022 tokens the adaptive and best fixed servers'
ranges now overlap (121,022: 26.8615 to 26.8975 against 26.8754 to 26.8869). At 34 and 7,588 they do not, although the
34-token path is unchanged from the third series (its prior is the cap, 15), where they overlapped: the gap is the size
of the spread between series.

## Spread at 34 and 7,588 tokens (2026-11-27, six servers an arm)

To tell that gap from the spread, the same two binaries ran the 34- and 7,588-token prompts on six fresh servers an arm:
`scripts/cold/series.sh` with `ARMS=(A F15 F7)`, `PROMPTS=(short long)` and the even rounds' order `F7 F15 A`
(36 cells, `SERVERS=6`), 16:06 to 16:30 local time. Full tables: `runs/cold-noise/summary.md`.

| prompt | adaptive: median (min to max), standard deviation | best fixed tree | mean, adaptive / best fixed |
|---:|---|---|---:|
| 34 | 127.01 (126.68 to 127.20), 0.18 | 15: 127.12 (127.04 to 127.21), 0.07 | 0.9989 |
| 7,588 | 67.29 (67.22 to 67.32), 0.04 | 7: 67.28 (67.18 to 67.35), 0.06 | 1.0000 |

- One `token_sha256` per prompt; 15.00 and 7.00 nodes a round for the adaptive servers.
- At 7,588 tokens the two arms are equal; the fourth series' gap there was spread.
- At 34 tokens the ranges overlap and the adaptive mean is 0.11 % lower, not significant at six servers (Welch t = 1.8,
  about 6.5 degrees of freedom, p about 0.12). A per-round cost of that size on the first request is neither shown nor
  ruled out; it is the remaining cold-request tradeoff.

## Tests

Host tests on the final head (`5ce953e90`): `zig build test-shared-lane-regressions-host` passes 513 with 1 skip, and
`zig build test-draft-xpu-host` passes 23; on a trial merge into main `044703688`, 523 with 1 skip and 23. The prior's
edges (2,047 to 2,049 and 8,191 to 8,193) and prompts of several lengths in one band (9K, 16,383 and 12K in band 1,
then 28K in the next band; 3,000, 8,191 and 2,049 in band 0) have host tests in `xpu_dflash2_lone_test.zig`.
No kernel file changed; the device tests and
the kernel contract of the first version of this receipt were not rerun on this head.

## Reproduce

`scripts/build_corp_code.sh inputs/corp_code.txt` rebuilds the prompts' text from the Ubuntu 26.04 python3.14 standard
library (expected SHA-256 `31b2f903b4a2ea01728927e53436d924994c93cf3114c7b7f35e58d906961e30`). Then
`BINMAIN=... BINHEAD=... MODEL=... DRAFTER=... OUT=... scripts/driver.sh` runs the passes and `scripts/analyse.py OUT`
prints the tables.

## Files

`runs/`: every request of the table's passes (`<pass>-<variant>.jsonl`, the seeded ones `-sampled.jsonl`; a variant
ending in `b` is a repeat pass of the same variant, `fix` is the head). `analyse.out`: the tables.
`scripts/`: the drivers. `runs/cold/`, `runs/cold-prior/`, `runs/cold-grow/`, `runs/cold-seed/`: the four
first-request series, and `runs/cold-noise/` the six-server spread check (`raw/`: one JSON line per cell, `summary.md`:
its tables, `binaries.sha256`); `scripts/cold/`: its driver (`series.sh`, with
`--dry-run`), client, prompts and summary.
