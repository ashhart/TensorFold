# XPU Qwen3.8-27B attn_merge: 16 subgroups a head

Receipts for the XPU decode attention merge (`attn_merge` in `zig/kernels/xpu/qwen27/decode.cl`). The kernel merges the
slices' partial maxima, sums and values into the attention output. It keeps each output lane's reduction order, so
its bytes equal the previous kernel's; only the work-group grid changes, from one subgroup a (head, row) to 16.
Measured on 2026-11-10, from 16:12 to 20:04 local time, against `main` at `8a7838a65`.

## Builds and hardware

- Branch: `5c45a2d65` (kernel and `forward.zig` at `673c90137`, kernel contract registration at `5c45a2d65`).
  Base: `main` at `8a7838a65`. Both binaries carry the same local VRAM-guard patch to `zig/src/xpu/rt.zig`, which is
  not part of this change and not committed; the runs also set that patch's card variable, left out of the scripts.
- Zig 0.17.0. Binaries: `zig build xpu-native -j2 -Doptimize=ReleaseFast -Dxpu-dpas=true -Dngen-include=<nGEN>` in each
  tree. Test binaries: `zig build xpu-qwen27-tests -j1 -Doptimize=ReleaseFast -Dxpu-dpas=true -Dngen-include=<nGEN>`.
- `tensorfold-native` SHA-256, base: `260496d22a190b43b69f4ed73e3ce6d01f13d7440d5001dbbae996538923aa34`
  (MD5 `a5a506f0f892d03446405b7e36654b73`); branch:
  `104f81c59df31af8076e8283361dbbcc3dbc8c1edabcb7bc06964b21faf55195` (MD5 `4e3ae59031c39e6d3560da108afd20a5`).
- Card: Intel Arc Pro B70 (BMG-G31, PCI id `0xe223`), the host's card 0 (`ZE_AFFINITY_MASK=0`). The host has a second
  B70, which these runs did not use. One card only.
- Host: Intel Core Ultra 7 270K Plus, 30 GiB RAM, Ubuntu 26.04.1, Linux 7.0.0-43-generic.
- Driver: compute-runtime (`libze-intel-gpu1`, `intel-opencl-icd`, `intel-ocloc`) 26.35.39758.21, IGC 2.41.5
  (`intel-igc-core-2`; `libigc2` 2.41.16), Level Zero loader 1.32.0, `libigdgmm12` 22.10.2.
- Checkpoints: `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` and its `Qwen3.8-27B-DFlash2` drafter, local copies.
  Hugging Face revisions, from each copy's download metadata: target `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`,
  drafter `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.

## Exactness: same tokens, main against branch

`runs/` holds every request of eight server passes, `<kv>-<pass>-<side>.jsonl`: bf16 KV then int8 KV
(`XPU_KV_INT8=1`), each as main, branch, branch, main. One pass starts a fresh `tensorfold-native serve` on port 8091
(`--parallel 1 --context 132096 --prompt-cache-gib 0`, with the DFlash2 drafter), runs `scripts/bench_merge.py` and
stops. Each cell is one prompt at greedy temperature 0, 256 new tokens, thinking off, drafted (the default) and plain
(`"draft": false`), with one warm-up request and three measured ones. Every request records the server's
`tensorfold.token_sha256`.

For each of the 14 cells (7 prompts, two KV formats) the 32 requests (four passes, drafted and plain, both sides)
carry one distinct `token_sha256`: `MISMATCH CELLS: 0` in `compare.out`. The output hash is therefore identical for
main and branch, and for drafted and plain, at every depth and in both KV formats.

## Speed

Decode tok/s, served, one stream, the median of a pass's three measured requests. Each cell below reads
`main / branch = ratio (range)`: main is the mean of the two main passes (1 and 4), branch the mean of the two branch
passes (2 and 3), the ratio is branch over main on the unrounded means (so its last digit can differ from the two
rounded means shown), and the range is the ratio of pass 2 to pass 1 and of pass 3 to pass 4. `compare.out` pools the
requests of both passes of a side; `abba.out` lists every pass.

| prompt tokens | kind | bf16 plain | bf16 drafted | int8 plain | int8 drafted |
|---|---|---|---|---|---|
| 34 | code | 33.49 / 33.50 = 1.000 (1.000-1.000) | 141.32 / 141.14 = 0.999 (0.998-0.999) | 33.43 / 33.44 = 1.000 (1.000-1.000) | 138.09 / 137.92 = 0.999 (0.999-0.999) |
| 29 | text | 33.49 / 33.46 = 0.999 (0.998-1.000) | 80.74 / 80.58 = 0.998 (0.998-0.999) | 33.44 / 33.44 = 1.000 (1.000-1.000) | 84.01 / 83.90 = 0.999 (0.999-0.999) |
| 7588 | code | 32.17 / 32.78 = 1.019 (1.019-1.019) | 59.00 / 59.08 = 1.001 (1.001-1.002) | 32.17 / 32.74 = 1.018 (1.017-1.019) | 64.02 / 63.97 = 0.999 (0.999-1.000) |
| 7912 | text | 32.04 / 32.68 = 1.020 (1.020-1.020) | 61.10 / 61.21 = 1.002 (1.002-1.002) | 32.07 / 32.69 = 1.019 (1.019-1.019) | 74.90 / 75.03 = 1.002 (1.002-1.002) |
| 28946 | code | 25.20 / 28.89 = 1.147 (1.146-1.147) | 29.45 / 29.53 = 1.003 (1.003-1.003) | 25.63 / 29.48 = 1.150 (1.150-1.150) | 33.76 / 33.71 = 0.999 (0.998-0.999) |
| 30966 | text | 24.72 / 28.63 = 1.158 (1.158-1.158) | 31.21 / 31.31 = 1.003 (1.003-1.003) | 25.28 / 29.34 = 1.160 (1.160-1.160) | 35.67 / 35.61 = 0.998 (0.998-0.998) |
| 121022 | code | 13.81 / 19.31 = 1.398 (1.398-1.398) | 10.63 / 10.67 = 1.004 (1.004-1.004) | 14.61 / 20.77 = 1.422 (1.422-1.422) | 11.53 / 11.53 = 1.000 (1.000-1.000) |

Plain (not drafted) decode gains 1.8 % to 2.0 % at about 8,000 prompt tokens, 14.7 % to 16.0 % at about 30,000 and
39.8 % to 42.2 % at 121,022 tokens (code only). At 34 and 29 prompt tokens it is level (0.998 to 1.000). Drafted
decode is level (0.998 to 1.004 in the table). The two paired ratios of a cell differ by 0.002 or less, so the order of
the passes does not show.

At 121,022 tokens drafted decode is slower than plain decode on both sides (bf16: drafted 10.63 on main and 10.67 on the
branch, plain 13.81 and 19.31); this receipt does not look at why.

## Several streams

`--parallel 4` is the multi-stream path, where `attn_merge` reads `POS` (one position a row). `--context 132096`
holds one stream here (`max_batch_size` 1 in `/health`), 65536 two, 40000 four: the context is per stream. These
passes use `--context 40000 --parallel 4`, so the 121,022-token prompt is not run. Four prompts of different depths
go out at the same moment from four threads, so the rows have different `POS`: 34, 7,588, 28,946 (code) and 30,966
(text) tokens, the same texts as the single-stream runs (same `prompt_sha256`). Same settings as above (greedy, 256
new tokens, thinking off, one card), drafted and plain, one warm-up wave and three measured ones, bf16 and int8 KV,
main then branch: four passes (`par4/`, `par4.out`), on 2026-11-10 from 21:01 to 21:31 local time. The server logs
`DFlash2 trees for a lone stream`: main drafts trees only for a stream that is alone (`LaneHost.lone`);
rounds shared by several streams verify their draft windows through `runner_multi.zig`, which passes `POS`.
Per-request draft counts were not recorded.

For each of the 16 cells (4 prompts, drafted and plain, two KV formats) the 8 requests (four waves, both sides) carry
one distinct `token_sha256`, and it is the one of the same cell in the single-stream passes (`runs/`):

- main = branch: 16 of 16 cells (`par4.out`, `MISMATCH CELLS: (a) 0`).
- the four waves equal: 16 of 16 (`(b) 0`).
- four streams = the same request served alone, on main and on the branch: 16 of 16 (`(c) 0`).

Speed is the four streams' 1,024 tokens over the wave's wall time, prefill included, median of the three measured
waves. A stream's own tok/s is not given: the prefills take turns, so it depends on which stream is served first.

| KV | mode | main s | branch s | main tok/s | branch tok/s | branch / main |
|---|---|---|---|---|---|---|
| bf16 | drafted | 50.2 | 50.1 | 20.41 | 20.43 | 1.001 |
| bf16 | plain | 52.9 | 51.8 | 19.36 | 19.75 | 1.021 |
| int8 | drafted | 49.0 | 48.9 | 20.90 | 20.93 | 1.001 |
| int8 | plain | 52.0 | 51.0 | 19.68 | 20.09 | 1.021 |

One pass a side, so no A-B-B-A order check here.

## Tests

Three device tests built by `xpu-qwen27-tests` ran on card 0, one binary at a time, for each of the two trees. All
three passed on both: 3 of 3 on the base, 3 of 3 on the branch, 0 failed, 0 skipped (`tests/`).

- `tf-xpu-qwen27-means-test`: GPTQ group means, worst error over the sum of absolute terms 4.19e-7 (prompt) and
  2.80e-7 (decode) against a bound of 2e-6, equal on both.
- `tf-xpu-qwen27-seam-test`: a 1,100-token prompt cut at 64 and at 1,024 gives the last logits of the whole prompt;
  0 of 248,320 logits differ in each cut, on both.
- `tf-xpu-qwen27-close-test`: three open and close rounds, with the same free memory after each, on both.

The kernel contract (`zig build -Dxpu-contract=true tf-xpu-contract-run`, `contract-run.out`):

```text
SUMMARY bmg-g31 0xe223 10 kernels: 10 pass gates 1-4, 0 fail, 0 known findings; 7 controls: 7 red as planted, 0 missed; 0 limit problems; gates 7802 ms, limits 0 ms
```

`attn_merge` is registered as a replacement for the frozen kernel of `8a7838a65`
(`zig/kernels/xpu/contract_merge_upstream.cl`): gate 3 requires its bytes on 21 shapes, 1 to 16 tree rows and 33 to
131,073 visible keys. Both new planted controls (the new kernel on the old grid, and a copy that weighs the lanes past
the last slice) go red at gate 3. Gate 1 (non-finite inputs) plants NaN, +inf and -inf in the first slice of rows 0 to
3, and +inf and -inf in the last slice of row 4 (two heads; 2 slices there, 33 visible keys): a merge that reads
values past a row's last slice and weighs them by zero turns that infinity into NaN, and the gate compares the classes
of the 30,720 outputs with the float64 reference's (516 non-finite, 30,204 finite). Before the kernel read zero there
it failed: `contract-old-kernel-red.out` is the contract run of the previous kernel with these two plants (`FAIL
nonfinite xpu attn_merge`: 2 of 30,720 outputs are NaN where float64 gives an infinity; summary `9 pass gates 1-4, 1
fail`). Gate 5 (`limits.out`): 16 work-items, sub-group 16, 0 SLM, 0 spill and 0 private bytes. The first line of
`contract-run.out`, `Size of SLM (163840) larger than available (131072)`, is the `160 KiB of SLM` limits control
being refused as planted.

## Final tree check

After the timed passes the load of `attn_merge` changed: the values past a row's last slice are now a literal zero
instead of a reload of the last slice (`min(s0 + k, used - 1)`), so an infinity in that slice is not turned into NaN by
the zero weight. The bytes are the same for finite values and now for infinities as well. The timed passes above ran the
previous load and were not repeated. On the final tree, on the host's card 1 (`ZE_AFFINITY_MASK=1`), one server
(`--parallel 1 --context 132096`, bf16 KV, DFlash2 drafter), `bench_merge.py` with `CELLS="0:code;8192:code"`, one
warm-up and one measured request, drafted and plain (`final-tree/served.jsonl`): 8 requests, 34 and 7,588 prompt tokens.
All 8 have the `prompt_sha256` and the `token_sha256` of the same prompt and mode in `runs/` (`5d721308b699d401` and
`9764d2d6fb7cbdbf`, the first 16 hex digits). The three device tests pass 3 of 3 on card 1
(`final-tree/tests-*.out`). The prompts of that run were built from `corp_code` rebuilt with
`scripts/build_corp_code.sh` (see Reproduce).

## Why the bytes are equal

For one output lane the old kernel adds the slices' weighted values in slice order, starting from zero. The new one
reads 16 slices' values at once into registers and adds them in the same order, with one subgroup lane per output
dimension and 16 subgroups per head. The maximum is an order-independent `fmax`, the weights are the same `exp`, and
the lanes past the last slice load zero and carry weight zero, so they add an exact zero.

## Limits

- Code only at 121,022 tokens (the 131K cell): there is no text prompt at that depth.
- Four simultaneous streams were measured (`--parallel 4`, `--context 40000`), not more, and not at 121,022 tokens.
- The contract's multi-stream row positions (the `POS` table) are not in the replacement test; the tree table is the
  only row source in gate 3. The kernel reads `POS` only to set `used`, a row's slice count, the same way before
  and after; gate 3 sweeps that count from 2 to 4,097 slices. The single-stream runs use one stream; the four-stream
  runs (Several streams) read `POS` with four different values.
- One card, one host, one driver stack. The second B70 and other Intel GPUs were not run.
- The contract was re-run on the final tree (`contract-run.out`), with the same pass, fail and control counts as the
  first run.
- The lean checker reports the same inherited problems on both trees (`pr_check` output in the pull request).

## Reproduce

Set `WORK` to a directory holding `bin-main/tensorfold-native`, `bin-branch/tensorfold-native` and `scripts/`, and
`MODELS` to the checkpoints' parent. Unpack the inputs next to the bench script:

```sh
mkdir scripts/inputs
scripts/build_corp_code.sh scripts/inputs/corp_code.txt
xz -dc inputs/corp_text.txt.xz > scripts/inputs/corp_text.txt
WORK=... MODELS=... scripts/driver.sh
python3 -I scripts/compare.py runs compare.md
python3 -I scripts/abba.py runs
```

`WORK=... MODELS=... scripts/par4.sh` runs the four-stream passes into `par4/`, and
`python3 -I scripts/par4_compare.py par4 runs` prints `par4.out`. `scripts/run-tests.sh` runs the three device
tests. The prompts are the two corpora cut to a token depth by the bench script's own calibration (a 20,000-character
probe); `prompt_sha256` in every line identifies the exact prompt. `corp_text` is Markdown documentation text.
`corp_code` is rebuilt, not stored: it is the top-level `*.py` files of the `python3.14` standard library at
`/usr/lib/python3.14` (Ubuntu 26.04 packages `python3.14`, `libpython3.14-minimal` and `libpython3.14-stdlib`, version
`3.14.4-1ubuntu0.2`; 157 files), concatenated without a separator, sorted by file name with case and punctuation
ignored (lower-cased, anything but `a-z0-9` removed, then the name; so `abc.py`, `_aix_support.py`,
`_android_support.py`, ... , `code.py` after `codeop.py`), and cut to the first 1,500,000 bytes.
`scripts/build_corp_code.sh OUT` does this and prints the SHA-256, which must be
`31b2f903b4a2ea01728927e53436d924994c93cf3114c7b7f35e58d906961e30`. The file this receipt used before was byte-identical
to that output (same SHA-256), and the final tree check above built identical prompts from the rebuilt file. Other
Python versions or packages give other bytes, hence other prompts.
