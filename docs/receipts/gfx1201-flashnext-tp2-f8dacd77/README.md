# gfx1201 Flash Next tp 2 measurement at TensorFold v1.2.7-399-gf8dacd77

Two-rank decode and cold-prefill measurement for [#2401](https://github.com/ashhart/TensorFold/issues/2401):
the first Flash Next receipt on gfx12 recorded in this repository.
Measured on 2026-11-13, on a pair of Radeon AI PRO R9700 (gfx1201, 32 GiB each), one host, PCIe, one model process.
The tested source is **`f8dacd777f1d846c48befd37159e74d52f759edb`** (`v1.2.7-399-gf8dacd777`: 399 commits past the `v1.2.7` tag, clean tree, binary version string 1.2.7), built with the pinned Zig 0.17.0.
This receipt does not qualify later `main` commits, other model families, other checkpoints or multi-host configurations.
No inference implementation changes are included.

## Environment

- [system.txt](system.txt): kernel, Ubuntu 24.04.5, EPYC 7543 host, 251 GiB RAM, 3 × R9700 + RX 9070 XT
- [provenance.json](provenance.json): the exact build command, Zig 0.17.0, ROCm 7.2.4 (HIP 7.2.53211-97f5574fe, AMD clang 22), both binaries' SHA-256
- [checkpoints.json](checkpoints.json): `TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP` at revision `0a1bf6faa0a2f16a314da889e8e6c80226455dbf` — 29 files checked against public Hugging Face metadata, all 16 LFS digests matched, 0 mismatches
- Serving: `serve <checkpoint> --tp 2 --rank R --master 127.0.0.1 --prompt-cache-gib 0`, rank 0 on HIP device 3 and rank 1 on HIP device 0 (two of the three R9700s; the third and the 16 GiB 9070 XT stayed idle). MTP drafts 3 and thinking on, the defaults.
  Decode: `--parallel 4 --context 4096`. Cold prefill: `--parallel 1 --context 40960`, fresh process.

## Decode

`tools/bench_openai.py … --tokens 256 --reps 3 --temperatures 0` and
`tools/bench_concurrent.py … --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial`, as in CONTRIBUTING.md.

| | this run, tp 2 | #2401's receipt, same card count, v1.2.3 |
|---|---|---|
| `bench_openai`, fibonacci-raw | 121.2 tok/s | 110.1 tok/s |
| `bench_concurrent` code, 1 / 4 streams (aggregate) | 121.5 / 213.2 tok/s | 109.8 / 193.0 tok/s |
| `bench_concurrent` chat, 1 / 4 streams (aggregate) | 103.5 / 171.0 tok/s | 93.0 / 154.6 tok/s |

[summary.json](summary.json) retains every repeat and hash; the raw receipts are [bench-openai.json](bench-openai.json) and [bench-concurrent.json](bench-concurrent.json).

## Cold prefill

Exact token arrays from this tree's own `README.md` + `RUNBOOK.md` + `CHANGELOG.md` (200,929 tokens), tokenized once with the server's `/v1/tokenize` and cut to each size, `/v1/completions` with `max_tokens=1`, 3 runs each, prompt cache off, fresh process per session. Raw runs in [run.log](run.log); the exact prompt token arrays are saved as [prefill-token-arrays.json](prefill-token-arrays.json), captured from the same checkpoint's tokenizer.

| prompt tokens | median of 3 | cached_tokens | #2401's v1.2.3 receipt |
|---|---|---|---|
| 2,048 | 1.72 s, 1,191 tok/s | 0 | 2.42 s, 848 tok/s |
| 8,192 | 7.31 s, 1,121 tok/s | 0 | 11.11 s, 737 tok/s |
| 32,768 | 30.96 s, 1,058 tok/s | 0 | 47.59 s, 689 tok/s |

A 65,536-token prompt is not admitted at the 40,960-token window. The weights take 27.3 GiB per rank, leaving
room for 4 concurrent streams in the decode session.

## Exactness

What these runs exercised, per [summary.json](summary.json):

- `bench_openai` fibonacci-raw, 3 repeats at temperature 0: identical `output_sha256` across all repeats.
- `bench_concurrent`, every cell: `alone` comparisons **30 equal / 0 unequal** (3 per 1-stream cell, 12 per 4-stream cell), `serial` comparisons **4 equal / 0 unequal** (1 per cell).
- Both concurrent aggregate numbers came from runs whose replies matched their solo replies token for token.

## Skips and remaining coverage

- The qwen3.5/3.6/3.8 family's `serve_check.py` HTTP suite and `tf-qwen35-test` invariants do not apply to this
  family's checkpoint and were not run; a separate qwen-family gfx1201 receipt is the right place for those gates.
- No multi-host rank pair (both ranks on one host, `--master 127.0.0.1`).
- No prompt-cache, recall or learned state: `--prompt-cache-gib 0` throughout, per the CUDA recipe's
  qualification terms at `--tp 2` with drafts.
- Only `--parallel 4 --context 4096` (decode) and `--parallel 1 --context 40960` (cold prefill) were measured;
  other windows are not covered.
- fp8 KV, drafter checkpoints and structured output were not exercised.

## Notes

- HIP enumeration on this host places the spare 32 GiB card at index 3; ranks were pinned by
  `HIP_VISIBLE_DEVICES` after verifying occupancy, not by trusting the index.
- The two rank logs ([rank0.log](rank0.log), [rank1.log](rank1.log)) retain each rank's own startup report and
  request log; `run.log` retains the full orchestration trace.
