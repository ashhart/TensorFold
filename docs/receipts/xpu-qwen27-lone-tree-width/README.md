# XPU Qwen3.8-27B: a lone DFlash2 stream picks its tree's width from the lane rule

Receipts for the XPU lone stream (`--parallel 1` with the DFlash2 drafter). Main verified a fixed 15-node tree every
round; the branch asks the core's lone-window pricing (`lone_costs`, with `allocateDrafts`) how many nodes to verify,
from the stream's own measured rounds, a context band each. `XPU_TREE` stays a cap. The width only sets how many
drafts are verified: the walk keeps the target's own pick, so every reply is the same.
Measured on 2026-11-11, from 20:44 to 23:28 local time, against `main` at `52c8f51e7`.

## Builds and hardware

- Branch: `02e26dd51` (one commit on `main` at `52c8f51e7`). Both binaries carry the same local VRAM-guard patch to
  `zig/src/xpu/rt.zig`, which is not part of this change and not committed; the runs also set that patch's card
  variable, left out of the scripts.
- Zig 0.17.0. Binaries: `zig build xpu-native -j2 -Doptimize=ReleaseFast -Dxpu-dpas=true -Dngen-include=<nGEN>` in each
  tree. Test binaries: `zig build xpu-qwen27-tests -j1 ...` and `zig build tf-xpu-contract-run -Dxpu-contract=true ...`.
- `tensorfold-native` SHA-256, base: `ccb634311d5c9c681a6233afec82bd8577f7549b82a9ee869007cba80e7c7750`
  (MD5 `db90abd15e89fe3ca13a93149ce96b38`); branch:
  `74020974b7fb10efea8009f6730900699e8ccb03a7a2b44daadb1ae18296c29d` (MD5 `99a96ad478ea8949dfda7fad44e90525`).
- Card: Intel Arc Pro B70 (BMG-G31, PCI id `0xe223`), the host's card 0 (`ZE_AFFINITY_MASK=0`). The host has a second
  B70, which the table did not use (it served development runs only). One card only.
- Host: Intel Core Ultra 7 270K Plus, 30 GiB RAM, Ubuntu 26.04.1, Linux 7.0.0-43-generic.
- Driver (`dpkg -l`): `intel-igc-core-2` 2.41.5, `intel-igc-opencl-2` 2.41.5, `intel-ocloc` 26.35.39758.21,
  `intel-opencl-icd` 26.35.39758.21, `libze-intel-gpu1` 26.35.39758.21, `libze1` 1.32.0, `libigdgmm12` 22.10.2.
- Checkpoints: `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` and its `Qwen3.8-27B-DFlash2` drafter, local copies.
  Hugging Face revisions from each copy's download metadata: target `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`,
  drafter `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.

## Protocol

Eight server passes, each a fresh `tensorfold-native serve` (`--parallel 1 --context 132096 --prompt-cache-gib 0`,
bf16 KV, the DFlash2 drafter), in this order: main with `XPU_TREE=15`, main `7`, main `3`, the branch (no `XPU_TREE`),
the branch again, main `3`, main `7`, main `15` (`scripts/driver.sh`, `scripts/run_pass.sh`). One pass runs
`scripts/bench_tree.py`: code prompts of 34, 28,946, 58,543 and 121,022 tokens (the prompt tokens of the cells named
0, 32768, 65536 and 131072 in the scripts; the first lines of `corp_code`, see Reproduce), greedy temperature 0, 256 new
tokens, thinking off, one warm-up request and three measured ones per cell. The two main tree-15 passes also run the
plain mode (`"draft": false`). Passes 1, 3 and 4 also run a seeded sampled cell (temperature 1.0, top_k 20,
top_p 0.95, seed 7) at 34 and 121,022 tokens. Every request records the server's `tensorfold.token_sha256` and its
`speculative` object. The branch's warm-up request (rep 0) also pays the width trials of a band it has not measured:
it is excluded from the speeds, and listed below.

## Exactness: same tokens at every width

`analyse.out` (from `scripts/analyse.py runs`): each cell's requests carry one `token_sha256`, across the fixed trees
of 15, 7 and 3 nodes, the branch's chosen widths and plain decode.

| prompt tokens | greedy requests (draft and plain) | distinct hashes | sampled requests (main 15, main 3, branch) | distinct hashes |
|---|---|---|---|---|
| 34 | 40 | 1 | 12 | 1 |
| 28,946 | 40 | 1 | | |
| 58,543 | 40 | 1 | | |
| 121,022 | 40 | 1 | 12 | 1 |

`CELLS WITH MORE THAN ONE HASH: 0`.

## Speed

Decode tok/s, served, one stream: the median of a pass's three measured requests, the mean of the variant's two passes
(each pass's median in brackets). The two passes of a variant differ by 0.2 tok/s or less, except the branch's
(0.2 to 1.0).

| prompt tokens | main, 15 nodes | main, 7 | main, 3 | branch (default) | plain decode |
|---|---|---|---|---|---|
| 34 | 141.23 (141.33, 141.13) | 132.50 | 91.63 | 136.14 (136.30, 135.99) | 33.42 |
| 28,946 | 29.46 (29.47, 29.45) | 40.24 | 45.21 | 46.42 (46.92, 45.91) | 25.21 |
| 58,543 | 19.74 (19.74, 19.74) | 28.63 | 34.31 | 34.84 (35.03, 34.65) | 19.85 |
| 121,022 | 10.63 (10.63, 10.63) | 16.63 | 20.48 | 21.64 (21.75, 21.52) | 13.82 |

The branch against the best fixed tree (unrounded means):

| prompt tokens | best fixed | branch / best fixed | branch / main default (15) | branch / plain |
|---|---|---|---|---|
| 34 | 15 nodes, 141.23 | 0.964 | 0.964 | 4.07 |
| 28,946 | 3 nodes, 45.21 | 1.027 | 1.576 | 1.84 |
| 58,543 | 3 nodes, 34.31 | 1.015 | 1.765 | 1.76 |
| 121,022 | 3 nodes, 20.48 | 1.056 | 2.036 | 1.57 |

**At 34 prompt tokens the branch is 3.6 % below the best fixed tree**: it does not reach the target of "at or above the
best fixed tree at each depth" there. Its measured requests verify 11.3 to 12.0 nodes a round (drafted / rounds) where
the best fixed tree verifies 15, and accept 4.36 drafts a round where 15 nodes accept 4.67; the cause is measured below, in a separate
instrumented run. At the three deep points it is above every fixed tree.

### Why 34 tokens stays below

A separate run of the branch with a temporary per-round log (price table, chosen width, measured time, drafted and
accepted; 34 tokens, one warm-up and 3 requests, 190 rounds, `diag/`) shows the cause. That run is not the table's: its
mean width is 13.4 where the table's is 11.66, and it reads 133 to 139 tok/s.

- A round costs about the same at 14 nodes as at 15. 14 nodes: 40.42 ms, sigma 1.37 ms, n = 156 rounds; the 15-node
  fixed tree: 40.1 ms a round. One round varies by 3.4 %, and the difference of two single timings by 4.8 %, a price
  being one timing blended at 0.2.
- The table keeps prices of widths the rule no longer picks, from the warm-up request's first rounds (15 nodes at 50.6
  and 48.5 ms there): 44.6 and 44.9 ms at 11 and 12 nodes while 14 is at 39.8, and 43.0 at 15 (one trial).
- The chances model is not at fault: 5.46 tokens a round predicted at the chosen width, 5.49 observed (184 rounds).
- The trials are not the cause either: 2 of the 190 rounds.
- The gap is rounds: 47 to 48 against 45 for 15 fixed nodes, at the same ms a round.

Three rules that take the widest width priced within a band of the best were tried on the same card, greedy, 1 warm-up
and 3 requests a depth (`exp/`, raw requests and bench logs; `exp/scatter-rule.patch` is the scatter rule, the 5 %
rule's build was not kept). "Base" is the branch's own rule, same binary, same session. The 5 % is a band on tokens a
ms; 2 sigma and 4 sigma are bands of that many scatters of the stream's own times (residual of their line over the
drafts count) on the price. Median of the requests, tok/s, and mean drafts a round in brackets:

| rule | 34 tokens | 28,946 | 121,022 |
|---|---|---|---|
| base | 135.82 (13.24) | 46.61 (4.04) | 22.00 (2.62), 2 requests, pass cut |
| 5 % of rate | 138.05 (14.14) | 41.28 (5.96) | 21.96 (2.98) |
| 2 sigma | 135.11 (12.81) | 47.06 (3.80) | 22.00 (2.54) |
| 4 sigma | 137.92 (14.65) | 42.42 (5.69) | 20.28 (3.47) |

The best fixed tree at 34 tokens is 141.23 (the table's, mean of two passes' medians; its passes read 141.3 each). None
of the rules gets there (0.977 for 5 %, 0.974 for 4 sigma), and the two that widen 34 tokens' trees lose against base at
28,946 tokens (0.886 and 0.910); 2 sigma changes neither depth. No such rule is in the branch. All hashes equal the
table's in these runs (34: `5d721308`, 28,946: `1d59abb4`, 121,022: `db07bcbf`, first 8 hex).

Drafts verified a round (`drafted / rounds`) and drafts accepted a round, mean of the measured requests:

| prompt tokens | 15 nodes | 7 | 3 | branch |
|---|---|---|---|---|
| 34 | 15.00 / 4.67 | 7.00 / 3.90 | 3.00 / 2.23 | 11.66 / 4.36 |
| 28,946 | 15.00 / 2.69 | 7.00 / 2.27 | 3.00 / 1.69 | 3.92 / 2.01 |
| 58,543 | 15.00 / 3.16 | 7.00 / 2.58 | 3.00 / 1.89 | 3.33 / 2.05 |
| 121,022 | 15.00 / 3.16 | 7.00 / 2.62 | 3.00 / 1.77 | 2.59 / 1.80 |

The branch's requests, one by one (rep 0 is the warm-up, which also runs width trials; at 121,022 tokens its 4.2 nodes
a round against 2.6 once the band is measured, and 18.0 tok/s against 21.4 to 22.4; the later requests still carry
the occasional trial round):

| prompt tokens | pass | rep | nodes a round | tok/s |
|---|---|---|---|---|
| 34 | 4 | 0 | 10.64 | 120.26 |
| 34 | 4 | 1 | 11.31 | 136.30 |
| 34 | 4 | 2 | 11.77 | 139.54 |
| 34 | 4 | 3 | 11.89 | 136.27 |
| 28,946 | 4 | 0 | 5.39 | 40.60 |
| 28,946 | 4 | 1 | 3.87 | 46.44 |
| 28,946 | 4 | 2 | 3.76 | 47.42 |
| 28,946 | 4 | 3 | 3.89 | 46.92 |
| 58,543 | 4 | 0 | 4.94 | 30.67 |
| 58,543 | 4 | 1 | 3.32 | 34.27 |
| 58,543 | 4 | 2 | 3.29 | 35.13 |
| 58,543 | 4 | 3 | 3.31 | 35.03 |
| 121,022 | 4 | 0 | 4.22 | 18.24 |
| 121,022 | 4 | 1 | 2.73 | 21.67 |
| 121,022 | 4 | 2 | 2.57 | 21.75 |
| 121,022 | 4 | 3 | 2.58 | 22.02 |
| 34 | 5 | 0 | 9.82 | 121.20 |
| 34 | 5 | 1 | 11.33 | 135.99 |
| 34 | 5 | 2 | 12.00 | 135.19 |
| 34 | 5 | 3 | 11.67 | 136.72 |
| 28,946 | 5 | 0 | 5.62 | 39.42 |
| 28,946 | 5 | 1 | 3.97 | 45.82 |
| 28,946 | 5 | 2 | 4.00 | 46.01 |
| 28,946 | 5 | 3 | 4.02 | 45.91 |
| 58,543 | 5 | 0 | 5.28 | 30.04 |
| 58,543 | 5 | 1 | 3.55 | 34.29 |
| 58,543 | 5 | 2 | 3.25 | 34.87 |
| 58,543 | 5 | 3 | 3.26 | 34.65 |
| 121,022 | 5 | 0 | 4.22 | 18.00 |
| 121,022 | 5 | 1 | 2.66 | 21.36 |
| 121,022 | 5 | 2 | 2.52 | 21.52 |
| 121,022 | 5 | 3 | 2.49 | 22.35 |

## Tests

Host: `zig build test` exits 0 on the branch; the new tests are under the filter `draft sources:` (`zig build
test-draft-xpu-host`) and `lone price:` (`lone_costs_test.zig`, whose CUDA tests are unchanged):

- the width follows the context band's measured rounds: a synthetic stream in a short band converges on the width
  that maximizes expected tokens a ms, a stream in the 131,072-token band on the narrow one (the exact brute-force
  optimum of each synthetic world: 12 and 4 nodes), each band's next stream starts on what its band kept;
- `XPU_TREE` caps (`priced(cap)`), and under 2 nodes the tree stays fixed and nothing is priced;
- `grown`/`follow` at widths 1 to 15, greedy and seeded: the list of drafts at w nodes is the first w of the list at 15
  (tokens, parents and own chances), and the accepted tokens of a 120-token reply equal the target's own continuation
  at every fixed width and in a mixed schedule;
- a lone round without `plain_guard` is priced from the stream's own table, leaves the shared rounds' table alone, and
  its probes start at 2 rows.

Three device tests built by `xpu-qwen27-tests` ran on card 0 on the branch: 3 of 3 passed (`tests/`).

The kernel contract (`zig build -Dxpu-contract=true tf-xpu-contract-run`, `tests/contract-run.out`); no kernel file
changed:

```text
SUMMARY bmg-g31 0xe223 9 kernels: 9 pass gates 1-4, 0 fail, 0 known findings; 5 controls: 5 red as planted, 0 missed; 0 limit problems; gates 5508 ms, limits 1 ms
```

## Reproduce

`scripts/build_corp_code.sh inputs/corp_code.txt` rebuilds the prompts' text from the Ubuntu 26.04 python3.14 standard
library (expected SHA-256 `31b2f903b4a2ea01728927e53436d924994c93cf3114c7b7f35e58d906961e30`). Then
`BINMAIN=... BINPR=... MODEL=... DRAFTER=... OUT=... scripts/driver.sh` runs the eight passes and
`scripts/analyse.py OUT` prints the tables; `scripts/run-tests.sh` runs the device tests.

## Files

`runs/`: every request of the eight passes (`<pass>-<variant>.jsonl`, the sampled ones `-sampled.jsonl`), nothing
else. `analyse.out`: the tables. `tests/`: the device tests and the
contract run. `scripts/`: the drivers.
