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

## Tests

Host tests on the head's code: `zig build test-shared-lane-regressions-host` passes 449 with 1 skip, and
`zig build test-draft-xpu-host` passes 21. No kernel file changed; the device tests and the kernel contract of the
first version of this receipt were not rerun on this head.

## Reproduce

`scripts/build_corp_code.sh inputs/corp_code.txt` rebuilds the prompts' text from the Ubuntu 26.04 python3.14 standard
library (expected SHA-256 `31b2f903b4a2ea01728927e53436d924994c93cf3114c7b7f35e58d906961e30`). Then
`BINMAIN=... BINHEAD=... MODEL=... DRAFTER=... OUT=... scripts/driver.sh` runs the passes and `scripts/analyse.py OUT`
prints the tables.

## Files

`runs/`: every request of the table's passes (`<pass>-<variant>.jsonl`, the seeded ones `-sampled.jsonl`; a variant
ending in `b` is a repeat pass of the same variant, `fix` is the head), nothing else. `analyse.out`: the tables.
`scripts/`: the drivers.
