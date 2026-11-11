# XPU Qwen3.8-27B sampled rows: the top_k on the card

Receipts for the XPU Qwen3.8-27B drawing a sampled row from a top-k kept on the card (`topk_part` and `topk_merge` in
`zig/kernels/xpu/qwen27/glue_quant.cl`). Before, every sampled row copied its 248,320 fp16 logits to the host, waited on the
card, and scanned them on one CPU thread (`pickRow` in `zig/src/native/xpu.zig`): once a round for a lone stream's DFlash2 tree
path and once a row in shared rounds. Now the forward that produced the logits also keeps, for the rows of its sampled
streams, the `top_k` largest values with their ids and a flag for any NaN or infinity in the row, in the copy and the
sync the greedy tokens already take. The draw is the same keyed rule over the same candidates. Measured on 2026-11-11,
from 10:28 to 11:27 local time, against `main` at `6b739dd5e`.

## What runs where

- On the card, after `argmax`, when a row of the forward belongs to a sampled stream with temperature above 0, no grammar
  mask, and `top_k` from 1 to 64: `topk_part` (one work-group per 1,024 ids of a row keeps its K best and flags any value
  that is not finite) and `topk_merge` (one work-group a row merges the up to 256 lists, by value and then the lower id,
  and ORs the flags). Values, ids and the flag go to one 528-byte row per logits row, copied beside the greedy tokens.
- On the host, the draw: the K values are widened to float64 and handed, with their ids, to the sampling rule `pickRow` already
  uses (`lanes.sampling.choose`). A flagged row fails as `NonfiniteLogits`, as `pickRow`'s scan of the whole row does.
- Unchanged, still on the host: `top_k` of 0 or above 64, temperature 0 with a sampling block, rows with a grammar mask, the
  prompt's last row (`first`), and any stream the card did not keep candidates for. Greedy streams read the device `argmax` as before.
- `verify`, `verifyTree` and `verifyMulti` no longer copy the rows' tokens and wait a second time: `forward` hands them back
  from the copy and the sync it already makes.

## Why the tokens cannot change

`pickRow` already draws a row with `top_k` up to 256 from its `top_k` largest values alone (`top_p` and `min_p` act on that
list, not on the whole row), tied values keeping the lower id, and fails on a value that is not finite anywhere in the row.
The card keeps exactly that list: the K largest of the row by value, the lower id first at a tie (signed zeros are equal,
and the value reported is the winning element's own, so a `-0` stays `-0`), where K is `top_k`. Each group's K best contain
every id of the row's K best that falls in the group, and the merge takes the K best of the lists in the same order. The flag
is raised by every id of the row, not by the candidates. Everything after is the same call with the same bits, the same
`position` and the same seed.

- Host test (`zig/src/native/xpu_sampled_topk_test.zig`, `zig build test-draft-xpu-host`): a model of the two kernels and the
  plain scan, for random rows, rows of a few levels (many ties, signed zeros), lengths on and around the 1,024-id group,
  `top_k` 1 to 64 and five `top_p`, `min_p`, temperature settings: `pickRow` over the whole row equals the draw from the
  candidates (1,080 draws), the candidates equal the full sort's first 64, and a NaN, `+Inf` or `-Inf` anywhere (first
  or last id of a group, outside the candidates) fails both the same way. With the merge's tie rule reversed the order
  test fails.
- Served: `compare.out` (`token_sha256` per request).

## Builds and hardware

- Branch: the two commits that carry this receipt (kernel and host code, then the kernel contract). Base: `main` at
  `6b739dd5e`. Both binaries carry the same local VRAM-guard patch to `zig/src/xpu/rt.zig`, which is not part of this change
  and not committed; the runs also set that patch's card variable, left out of the scripts.
- Zig 0.17.0. Binaries: `zig build xpu-native -j2 -Doptimize=ReleaseFast -Dxpu-dpas=true -Dngen-include=<nGEN>` in each tree.
- `tensorfold-native` SHA-256, main: `91828565f980e55ac7d56835ed49fb617e8aa211c603948c5f77e9f7c1558ef9`
  (MD5 `8d37d0f55c713f4b2f65baeb0f332fbb`); branch:
  `c748436aded95ce62c1877d3dc66bca4a04f9849eac38da175551d947200ebc4` (MD5 `26ae47dc157a2f4d507edd3b7fda7086`).
  A rebuild at the last commit of the branch gives the same branch SHA-256.
- Card: Intel Arc Pro B70 (BMG-G31, PCI id `0xe223`), the host's card 0 (`ZE_AFFINITY_MASK=0`). The host has a second
  B70, which these runs did not use. One card only.
- Host: Intel Core Ultra 7 270K Plus, 30 GiB RAM, Ubuntu 26.04.1, Linux 7.0.0-43-generic.
- Driver: compute-runtime (`libze-intel-gpu1`, `intel-opencl-icd`, `intel-ocloc`) 26.35.39758.21, IGC 2.41.5
  (`intel-igc-core-2`, `intel-igc-opencl-2`), Level Zero loader 1.32.0 (`libze1`).
- Checkpoints: `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` and its `Qwen3.8-27B-DFlash2` drafter, local copies. Hugging Face
  revisions, from each copy's download metadata: target `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`, drafter
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.

## Method

`runs/` holds every request of four passes in the order main, branch, branch, main (`p1-main`, `p2-branch`, `p3-branch`,
`p4-main`), one JSON line a request. A pass starts a fresh `tensorfold-native serve` (port 8091, bf16 KV, the DFlash2 drafter,
`--context 10240 --prompt-cache-gib 0`) at `--parallel 1`, then 4, then 8 (all eight streams fit in bf16: `/health` reports
`max_batch_size` 8), and stops it. A cell is one prompt at one width: a wave of that many concurrent requests, stream i
with seed 1000 + i, 256 new tokens, thinking off, the checkpoint's own sampling (temperature 1.0, top_k 20, top_p 0.95; the
request sends none), one warm-up wave and three measured ones. Prompts: the short pair of `bench_topk.py` (34 tokens of
code, 29 of text) and the two corpora of the earlier receipt cut near 8,000 tokens (7,588 code, 7,912 text;
`scripts/build_corp_code.sh` rebuilds `corp_code.txt`, `corp_text.txt` is that receipt's). At one stream, two greedy
requests (`temperature` 0, short code and text) go first. Every request records the server's `tensorfold.token_sha256`.

## Exactness: same tokens, main against branch

`compare.out`, 840 requests (four passes, 2 greedy and 208 sampled a pass):

- 54 (mode, width, prompt, stream) cells; each carries one distinct `token_sha256` over the two main passes, the two branch
  passes and the four waves of each: `MISMATCH main/branch/passes/waves: 0`. Every request produced 256 tokens.
- The same prompt and seed gives the same hash at 1, 4 and 8 streams (a stream in a shared round equals the lone one):
  `MISMATCH across 1, 4 and 8 streams: 0 of 34` (prompt, seed) pairs.
- The two greedy requests have the same hash on main and on the branch (greedy is unchanged).

After the timed passes, a review changed the kernel's tail and the host bookkeeping, not the candidates of the served
vocabulary: a group's cleared winner becomes the `(-inf, 0x7fffffff)` sentinel (a group with fewer live ids than K no
longer repeats its last id), and the forward notes the rows it holds candidates for after its sync. On that final tree
one more pass (1, 4 and 8 streams, the same cells, 210 requests) gives the receipt's `token_sha256` for every
(prompt, seed, mode): 0 of 210 differ (`final-tree/served.jsonl`). The contract, the host test, the device tests,
`zig build test` and `pr_check` were run again on it; `contract-run.out` is that run. The timed passes were not repeated.

## Speed

Decode tok/s, served, sampled, DFlash2, one card. A cell reads `main / branch = ratio (pair 1-2, pair 4-3; gap)`: main is the
mean of passes 1 and 4, branch the mean of passes 2 and 3, each pass's value the median of its three measured waves; the
ratio is branch over main on the unrounded means; the two paired ratios are branch pass 2 over main pass 1 and branch pass 3
over main pass 4, and the gap is their difference. "Per stream" is the mean over the wave's streams of each stream's
`(tokens - 1) / (last token - first token)`; with 4 or 8 streams at 8,000 tokens the prefills take turns, so a stream's rate
depends on which one is served first. "Wave" is the wave's completion tokens over its wall time, prefill included, so at 8,000
tokens it is mostly prefill.

| streams | prompt tokens | kind | per stream, main / branch | ratio (pairs; gap) | wave, main / branch | ratio (pairs; gap) |
|---|---|---|---|---|---|---|
| 1 | 34 | code | 153.00 / 158.61 | 1.037 (1.037, 1.037; 0.000) | 146.76 / 151.90 | 1.035 (1.035, 1.035; 0.000) |
| 1 | 29 | text | 80.20 / 80.56 | 1.005 (1.004, 1.005; 0.001) | 78.54 / 78.88 | 1.004 (1.004, 1.005; 0.001) |
| 1 | 7,588 | code | 65.00 / 65.40 | 1.006 (1.005, 1.007; 0.001) | 31.94 / 32.01 | 1.002 (1.001, 1.003; 0.002) |
| 1 | 7,912 | text | 60.66 / 60.93 | 1.004 (1.004, 1.005; 0.001) | 30.23 / 30.27 | 1.001 (1.000, 1.002; 0.002) |
| 4 | 34 | code | 67.64 / 74.88 | 1.107 (1.109, 1.105; 0.003) | 219.78 / 245.27 | 1.116 (1.114, 1.118; 0.005) |
| 4 | 29 | text | 46.37 / 51.27 | 1.106 (1.103, 1.109; 0.006) | 175.25 / 192.56 | 1.099 (1.102, 1.096; 0.006) |
| 4 | 7,588 | code | 22.32 / 23.19 | 1.039 (1.026, 1.053; 0.027) | 42.38 / 43.29 | 1.021 (1.020, 1.023; 0.003) |
| 4 | 7,912 | text | 21.73 / 22.77 | 1.048 (1.043, 1.053; 0.010) | 41.87 / 42.67 | 1.019 (1.017, 1.021; 0.003) |
| 8 | 34 | code | 37.51 / 41.39 | 1.103 (1.126, 1.081; 0.045) | 257.12 / 284.62 | 1.107 (1.102, 1.112; 0.010) |
| 8 | 29 | text | 31.19 / 34.49 | 1.106 (1.101, 1.110; 0.009) | 213.63 / 238.90 | 1.118 (1.134, 1.103; 0.031) |
| 8 | 7,588 | code | 12.11 / 12.62 | 1.042 (1.038, 1.047; 0.010) | 45.14 / 45.98 | 1.019 (1.018, 1.019; 0.001) |
| 8 | 7,912 | text | 11.97 / 12.50 | 1.044 (1.079, 1.010; 0.068) | 44.05 / 44.85 | 1.018 (1.018, 1.018; 0.001) |

## Tests

On the branch, on card 0, one binary at a time (`tests/`):

- `tf-xpu-qwen27-means-test`, `tf-xpu-qwen27-seam-test`, `tf-xpu-qwen27-close-test`: all three printed `ok`
  (`tests/branch-*.out`): group means worst error over the sum of absolute terms 4.19e-7 (prompt) and 2.80e-7 (decode) against a
  bound of 2e-6; a 1,100-token prompt cut at 64 and at 1,024 gives 0 of 248,320 logits different; three open and close rounds,
  29,999, 30,000 and 29,998 MiB free after each.
- `zig build test`: rc 0 (`tests/zig-build-test.rc`); its output (`tests/zig-build-test.out`) holds stderr lines from tests that expect a failure (`failed command`, `DeviceLost`), which are not failures
- `python3 tools/pr_check.py --base 6b739dd5e`: 0 errors.

The kernel contract (`zig build -Dxpu-contract=true tf-xpu-contract-run`, `contract-run.out`):

```text
SUMMARY bmg-g31 0xe223 10 kernels: 10 pass gates 1-4, 0 fail, 0 known findings; 7 controls: 7 red as planted, 0 missed; 0 limit problems; gates 5607 ms, limits 1 ms
```

`topk_part` and `topk_merge` register as one kernel, "xpu top-k (the candidates)", with a stated rule (a row with a NaN or an
infinity anywhere raises the flag, whatever the candidates hold; else the flag is clear and the K candidates are the K
largest, the lower id first at a tie). It passes gate 1 (NaN, `+Inf` and `-Inf` planted in rows 0 to 3, outside the top
values for the NaN and `-Inf` rows), gate 2, and gate 3 on 14 shapes: 1 to 16 rows, 1 to 64 candidates, 64 to 262,144 ids
including 1,023, 1,025 and 100,003 (not whole groups), and rows holding one value everywhere or zeros of both signs on top.
Two planted controls, each a copy of `topk_merge` in `zig/kernels/xpu/contract_gate_controls.cl` over the real `topk_part`,
go red for their reason: the merge that breaks ties to the higher id at gate 3 (`candidate 0 is 15.875 at id 248007, the rule
gives 15.875 at id 317`), and the merge whose flag reads the merged candidates instead of the groups' flags at gate 1
(`row 0 flag 0 where the row holds a value that is not finite`). `limits.out` has the gate 5 records
(`topk_part`, `topk_merge`: 256 work-items, 192 SLM bytes, sub-group 16, 256 GRF, no spill).

## Files

`runs/<pass>-<side>-par<N>.jsonl` (requests; `-greedy` for the two greedy ones), `final-tree/served.jsonl`,
`compare.out`, `contract-run.out`,
`limits.out`, `tests/`, `scripts/` (`bench_topk.py` the client, `run_pass.sh` and `driver.sh` the passes, `compare_topk.py`,
`run-tests.sh`, `build_corp_code.sh`).
