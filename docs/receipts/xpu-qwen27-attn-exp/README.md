# XPU Qwen3.8-27B attn_decode: a key's twelve exponentials as one vector exp

Receipts for a change to `attn_decode` in `zig/kernels/xpu/qwen27/decode.cl` (refs #2543). For each key the kernel
computed twelve scalar exponentials, `exp(m - mn)` and `exp(sc - mn)` for the six query heads of a K/V head. They are now
the 16 lanes of one vector `exp` (head a's two arguments in lanes a and a + 8), and each head reads its two results back
with `sub_group_broadcast`. The arguments, `fmax`, the update order and every operand are unchanged, and there is no new
branch, so the output bytes are equal. Measured on 2026-11-22 against `main` at `14951bc65`.

## Builds and hardware

- Branch `xpu-qwen27-attn-exp`: kernel change `2aa76b383`, planted control and its contract wiring `bc332f14e`, on
  `main` at `14951bc65`. Neither binary carries a local patch: `zig/src/xpu/rt.zig` is main's.
- Zig 0.17.0. Binaries: `zig build -j4 -Dxpu xpu-native -Doptimize=ReleaseFast -Dxpu-dpas=true -Dngen-include=<nGEN>`
  in each tree. Contract runner and limits audit: `zig build tf-xpu-contract-run xpu-kernel-contract-limits
  -Dxpu-contract=true -Dxpu-dpas=true -Dngen-include=<nGEN> -Doptimize=ReleaseFast`.
- `tensorfold-native` SHA-256, main: `bd78137b649c987d4d2848f212790e46457e0fb18cdf0e08795432f6bf074d12`
  (MD5 `9bf82d00509328c873abeeaa7c654ab6`); branch:
  `90349ad00c0015c2ba7dd4dcdb80dc813a06b71c4f4bf0811f237c40023889fd` (MD5 `d211292a5c26a572209f94aabb3e169b`)
  (`binaries.sha256`).
- Card: Intel Arc Pro B70 (BMG-G31, PCI id `0xe223`), the host's card 0 (PCIe 5.0 x16, `ZE_AFFINITY_MASK=0`) for every
  served and kernel timing. The kernel contract ran on card 1 (`ZE_AFFINITY_MASK=1`). One host, one card at a time.
- Host: Intel Core Ultra 7 270K Plus, 30 GiB RAM, Ubuntu 26.04.1, Linux 7.0.0-43-generic.
- Driver (`dpkg -l`): `intel-igc-core-2` 2.41.5, `intel-igc-opencl-2` 2.41.5, `intel-ocloc` 26.35.39758.21,
  `intel-opencl-icd` 26.35.39758.21, `libze-intel-gpu1` 26.35.39758.21, `libze1` 1.32.0, `libigdgmm12` 22.10.2.
- Checkpoints: `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` and its `Qwen3.8-27B-DFlash2` drafter, local copies. Hugging Face
  revisions from each copy's download metadata: target `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`, drafter
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.

## Served: same tokens, and the speed

Four server passes in this order, each a fresh `tensorfold-native serve` on card 0 (`--parallel 1 --context 132096
--prompt-cache-gib 0`, bf16 KV, the DFlash2 drafter, no `XPU_TREE`): main, branch, main, branch (`scripts/driver.sh`,
`scripts/run_pass.sh`, with the page cache dropped after each). One pass runs `scripts/bench_tree.py`: code prompts of
7,588, 28,946 and 121,022 tokens (cells 8192, 32768 and 131072 of the script, the first lines of `corp_code`, see
Reproduce), greedy temperature 0, 256 new tokens, thinking off, each drafted (the default) and plain (`"draft": false`),
one warm-up request and three measured ones per cell. Every request records the server's `tensorfold.token_sha256`;
`runs/` holds all of them (`p<N>-main.jsonl`, `p<N>-branch.jsonl`, the pass's server and bench logs).

### Tokens

Each cell's 16 requests (four passes, main and branch, one warm-up and three measured each) carry one `token_sha256`,
and the drafted and the plain cell of a depth share it: `CELLS WITH MORE THAN ONE HASH: 0`. The drafter's counters agree too:
at each depth the `speculative` object (drafted, accepted, rounds) is identical on every drafted request of both sides
(1,065 drafted and 186 accepted in 71 rounds at 7,588 tokens; 1,050 and 188 in 70 at 28,946; 1,005 and 188 in 67 at
121,022). All 96 requests have a `token_sha256` (`analyse.out`, from `scripts/analyse.py runs`).

| prompt tokens | token_sha256 (first 16 hex) | prompt_sha256 (first 16 hex) |
|---|---|---|
| 7,588 | `9764d2d6fb7cbdbf` | `829dddadac0f329f` |
| 28,946 | `1d59abb499c48841` | `0bd40816987651da` |
| 121,022 | `0ac2491477780561` | `60986a50b4990e56` |

### Speed

Decode tok/s, served, one stream: the median of a pass's three measured requests (the warm-up request is left out).
`main` is the mean of passes 1 and 3, `branch` the mean of passes 2 and 4; the ratio is branch over main on the
unrounded means; the last two columns are the paired ratios, pass 2 over 1 and pass 4 over 3.

: the median of a pass's three measured requests (warm-up request excluded)

| prompt tokens | mode | main p1 | branch p2 | main p3 | branch p4 | main mean | branch mean | branch / main | p2/p1 | p4/p3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 7588 | drafted | 62.31 | 63.09 | 62.21 | 63.09 | 62.26 | 63.09 | 1.013 | 1.013 | 1.014 |
| 7588 | plain | 32.67 | 32.69 | 32.67 | 32.69 | 32.67 | 32.69 | 1.001 | 1.000 | 1.001 |
| 28946 | drafted | 36.33 | 37.42 | 36.31 | 37.41 | 36.32 | 37.41 | 1.030 | 1.030 | 1.030 |
| 28946 | plain | 28.94 | 28.98 | 28.94 | 28.96 | 28.94 | 28.97 | 1.001 | 1.002 | 1.001 |
| 121022 | drafted | 13.30 | 13.95 | 13.30 | 13.96 | 13.30 | 13.96 | 1.049 | 1.049 | 1.049 |
| 121022 | plain | 19.50 | 19.52 | 19.50 | 19.53 | 19.50 | 19.52 | 1.001 | 1.001 | 1.002 |

Drafted decode gains 1.3 % at 7,588 prompt tokens, 3.0 % at 28,946 and 4.9 % at 121,022 (the two paired ratios of a
cell differ by 0.001 or less). Plain decode is level (1.001 to 1.002 at the three depths): it runs one row a token
through `attn_decode`, where a drafted round runs 15 tree nodes, and the kernel's saving there is 2.4 % at 20,000 tokens
and 0.8 % at 58,543 (`kernel/`), a small part of a token. Every request is in `analyse.out`.


## attn_decode GPU time

`attn_decode` GPU ms a drafted round (`XPU_PROFILE=1`, the dev runner, card 0, 20,000 and 58,543 prompt tokens; tree 3 is
4 rows a round, tree 15 is 16 rows; the serial column is ms a plain token). `kernel/` holds the eight logs and
`kernel/summarize.py` (`summary.out`) prints the table. The runner trees also carried an unlaunched second kernel
(`attn_decode_rows`, not part of this change); `attn_decode`'s source in them is the one on main and on the branch.

| prompt tokens | 4 rows, main -> branch | 16 rows, main -> branch | plain token, main -> branch |
|---|---|---|---|
| 20,000 | 8.06 -> 7.60 ms (-5.7 %) | 33.51 -> 30.99 ms (-7.5 %) | 2.932 -> 2.861 ms (-2.4 %) |
| 58,543 | 23.12 -> 21.72 ms (-6.1 %) | 95.69 -> 90.43 ms (-5.5 %) | 8.796 -> 8.722 ms (-0.8 %) |

`attn_merge` is unchanged (`summary.out`). Repeat runs of one cell agree to 0.1 %. With a 15-node tree at 58,543 tokens
the verify round goes from 169.9 to 160.7 ms on card 0 (`kernel/*-58543-tree15.log`: "verify ... ms a round").

Where the rest of the time goes (ablations of the new kernel, numerically wrong, timing only; 16 rows, 20,000 tokens,
31.5 ms a round): V loads replaced by a constant 23.8 ms, K loads replaced by a constant 24.9 ms, the dot cut to two
terms 26.1 ms, no `o * al` multiplications 30.5 ms. The kernel waits on K/V load latency and on the dot's dependent
chain, not on its instruction count; hand-scheduled variants of the loads (block reads, V requested before the dot, the
next key's K in registers) were slower, 31.4 to 65.5 ms, so this change is the one exact edit that paid.

## Why the bits cannot differ

For every (row, head, slice, key) the operations are main's: `mn = fmax(m, sc)`, `al = exp(m - mn)`, `p = exp(sc - mn)`,
`l = l*al + p`, `o = o*al + p*kv`, `m = mn`, in that order on those operands. Only the place where `exp` runs moves: head
a's `m - mn` is in lane a of a vector, its `sc - mn` in lane a + 8, and one `exp` serves all lanes. It needs two
facts, both measured on the card (`proofs/`, ctypes over the OpenCL runtime, the images' compiler options):

1. The vector `exp` equals the scalar `exp` on every input: `exp` of all 4,294,967,296 float bit patterns, NaNs and
   infinities included, as a SIMD16 vector against the same value through `sub_group_broadcast` (16 scalar
   `math.exp` sequences, as `attn_decode` had): 0 mismatches (`proofs/expprobe_all.out`, 18.6 s).
2. The vector `fmax` equals the scalar one: 536,870,912 pairs (all pairs of 25 special values: signed zeros, NaN
   payloads, infinities, denormals; nearby and random patterns), 0 mismatches (`proofs/fmaxprobe.out`).

So no non-finite or signed-zero input can differ either, which is why the kernel needs no finite-score guard. Lanes 6, 7,
14 and 15 hold -inf and give NaN that nothing reads. Rerun with `ZE_AFFINITY_MASK=0 python3 -I
scripts/expprobe_all.py` and `scripts/fmaxprobe.py` (numpy and `libOpenCL.so.1`; about 20 s each).

## Kernel contract

`contract/contract-run-branch.out`, card 1, `tf-xpu-contract-run`:

```text
SUMMARY bmg-g31 0xe223 21 kernels: 21 pass gates 1-4, 0 fail, 0 known findings; 12 controls: 12 red as planted, 0 missed; 0 limit problems
```

`contract/contract-run-main.out` is the same run on main: 21 kernels, 11 controls. The branch adds one control and no
kernel. `attn_decode` and `attn_merge` pass gates 1 to 4 against the bytes of the upstream `1a5644892` pair (`PASS shapes
xpu attn_decode + attn_merge (bf16 cache, tree): 20 shapes bit-equal`). The new planted control,
`tf_ctl_attn_exp_lanes` (the same kernel with the scores in lanes 0 to 7 only, so every `p` is `exp(-inf)`), goes red at
gate 3: `FAIL shapes control: xpu attn_decode whose exponential vector holds the scores in lanes 0 to 7 only`. Gate 5
(`contract/limits-branch.out`, `xpu-kernel-contract-limits`: `PASS limits: 25 registered XPU launches against bmg-g31, 0
problems`, `attn_decode`: 16 work-items, SLM 0, 256 GRF, 0 spill bytes, 0 private bytes, record current; the live gate
says the same of the loaded image). The first line of the run log, `Size of SLM (163840) larger than available (131072)`,
is the `160 KiB of SLM` limits control being refused as planted.

Host checks, no GPU: `zig build test-kernel-contract test-qwen27-config-xpu-host test-qwen27-prompt-tail-xpu-host`
(48 of 48 steps, 288 of 288 tests; the rerun on the final commit, which only changed two
comments, ran the 252 tests that were not cached and passed them all).

## Rerun after the review fix

Review: in `attn_decode`, `sub_group_broadcast(mv, a)` sat inside `if (lane == 0)`, so only lane 0 reached the
collective. Commit `8b99d80e3` computes `const float ma = sub_group_broadcast(mv, a);` before the branch on every lane
and lane 0 stores `ma`; the same hoist is in the planted control `tf_ctl_attn_exp_lanes`, whose score-lane mutation is
unchanged. Nothing else moved. The branch binary is new, `tensorfold-native` SHA-256
`b286eee7271352a1c8044692374a61a47cf801b863786e1445a3b0cd60378dc3` (MD5 `707d02a181dda93c756e692d01452ab3`), built
as above on 2026-11-24; the main binary is the one of the table above (same SHA-256). Same host, driver stack and
checkpoints, every run on card 0 (`ZE_AFFINITY_MASK=0`).

- Kernel contract, `contract/contract-run-branch-fix.out`: `SUMMARY bmg-g31 0xe223 21 kernels: 21 pass gates 1-4, 0 fail,
  0 known findings; 12 controls: 12 red as planted, 0 missed; 0 limit problems`. Apart from timings the file is the same as
  `contract/contract-run-branch.out`, which ran on card 1. `xpu-kernel-contract-limits`: every launch fits, `attn_decode`
  16 work-items, SLM 0, 256 GRF, 0 spill bytes.
- Served, `runs-fix/` (same script, `scripts/run_pass.sh`; one fresh server a pass, one pass a side and depth group:
  branch then main at 7,588 and 28,946 tokens, then branch then main at 121,022; `summarize.py` prints the table):
  every request of every cell carries the `token_sha256` of the first table, in full, and the drafter's counters are
  unchanged (71 rounds, 1,065 drafted, 186 accepted at 7,588; 70, 1,050, 188 at 28,946; 67, 1,005, 188 at 121,022).

| prompt tokens | mode | token_sha256 (first 16 hex) | main | branch | branch / main |
|---|---|---|---|---|---|
| 7588 | drafted | `9764d2d6fb7cbdbf` | 62.26 | 63.28 | 1.016 |
| 7588 | plain | `9764d2d6fb7cbdbf` | 32.67 | 32.69 | 1.001 |
| 28946 | drafted | `1d59abb499c48841` | 36.34 | 37.47 | 1.031 |
| 28946 | plain | `1d59abb499c48841` | 28.94 | 28.98 | 1.001 |
| 121022 | drafted | `0ac2491477780561` | 13.30 | 13.97 | 1.050 |
| 121022 | plain | `0ac2491477780561` | 19.50 | 19.54 | 1.002 |

Decode tok/s, the median of three measured requests (one pass a side, so no paired ratio); the speed matches the table above.

## Limits

- Code prompts only, at the three depths named. One host, one card for the served numbers, one driver stack.
- Tokens were also compared at the kernel level, serial and drafted (trees of 15 and 3), for code and text at 8,192 and
  30,000 prompt tokens, with `XPU_KV_INT8=1` and on the multi-stream path; those dev-runner runs are not in this receipt.
  The served cells above are the receipt.
- The kernel saves 5.5 to 7.5 % of `attn_decode`'s GPU time in a drafted round; what a served request gains is that
  kernel's share of its decode time, which grows with the prompt (the served table).

## Reproduce

Set `WORK` to a directory holding `bin-main/`, `bin-branch/` (each `xpu/bin/tensorfold-native`), `inputs/` and `scripts/`
(from here), and `MODELS` to the checkpoints' parent:

```sh
scripts/build_corp_code.sh inputs/corp_code.txt
BINMAIN=... BINPR=... MODEL=... DRAFTER=... OUT=runs scripts/driver.sh
python3 -I scripts/analyse.py runs
```

`corp_code` is rebuilt, not stored: the top-level `*.py` files of the `python3.14` standard library at
`/usr/lib/python3.14` (Ubuntu 26.04 `python3.14` 3.14.4-1ubuntu0.2; 157 files), sorted by file name with case and
punctuation ignored, concatenated and cut to 1,500,000 bytes; its SHA-256 must be
`31b2f903b4a2ea01728927e53436d924994c93cf3114c7b7f35e58d906961e30` (`build_corp_code.sh` prints it). `prompt_sha256` in
every line identifies the exact prompt.
