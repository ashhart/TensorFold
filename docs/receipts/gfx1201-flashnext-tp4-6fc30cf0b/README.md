# gfx1201 Flash Next world 4 measurement at TensorFold v1.2.7-1722-g6fc30cf0b

Four-rank (world 4) Flash Next measurement, with a same-binary world 2 control, for [#2401](https://github.com/ashhart/TensorFold/issues/2401):
the first world-4 Flash Next receipt on HIP recorded in this repository, on four Radeon AI PRO R9700 (gfx1201, 32 GiB each), one host, one model process per rank.
The tested source is **`6fc30cf0bf3bf0f6488d62f93550e56d7ded1944`** (`v1.2.7-1722-g6fc30cf0b`, clean tree), built with the pinned Zig 0.17.0 after the `--tp` world generalization of #2917.
This receipt does not qualify other model families, other checkpoints, other worlds, later `main` commits or multi-host configurations. No inference implementation changes are included.

## Environment

- [system.txt](system.txt): kernel, Ubuntu 24.04.5, EPYC 7543 host, 251 GiB RAM, 4 × R9700 (gfx1201)
- [provenance.json](provenance.json): the exact build command, Zig 0.17.0, ROCm 7.2.4 (hipcc reporting HIP 7.14.60850, AMD clang 23), the binary's SHA-256
- [checkpoints.json](checkpoints.json): `TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP` at revision `0a1bf6faa0a2f16a314da889e8e6c80226455dbf` — 29 files checked against public Hugging Face metadata, all 16 LFS digests matched, 0 mismatches
- Serving: `serve <checkpoint> --tp 4 --rank R --master 127.0.0.1 --prompt-cache-gib 0`, one rank per card, ranks pinned by `HIP_VISIBLE_DEVICES`.
  Decode: `--parallel 4 --context 4096`. Cold prefill: `--parallel 1 --context 40960`, fresh process per session. MTP drafts 3 and thinking on, the defaults.
- World 2 control: the same binary, the same commit, two ranks on two of the same cards (`--tp 2`), identical flags and battery — run to isolate the world effect from the engine changes between receipts.

## World 4, and its world 2 control on the same binary

| | world 4 (4 ranks) | world 2 control (2 ranks) | world effect |
|---|---|---|---|
| `bench_openai`, fibonacci-raw | 122.9 tok/s | 121.2 tok/s | +1.4% |
| `bench_concurrent` code, 1 / 4 streams (aggregate) | 123.8 / 242.0 tok/s | 121.5 / 249.4 tok/s | +1.9% / −3.0% |
| `bench_concurrent` chat, 1 / 4 streams (aggregate) | 103.5 / 205.2 tok/s | 103.5 / 198.6 tok/s | 0% / +3.3% |
| cold prefill 2,048 | 1,412 tok/s | 1,219 tok/s | +15.8% |
| cold prefill 8,192 | 1,358 tok/s | 1,165 tok/s | +16.6% |
| cold prefill 32,768 | 1,263 tok/s | 1,079 tok/s | +17.1% |

Reading, in the measured scope only: the world pays on cold prefill (the prompt work splits across ranks), is neutral to slightly negative on 4-stream decode aggregate, and leaves single-stream decode unchanged. The earlier tp 2 receipt (`v1.2.7-399`, in #3608 until it lands) measured 121.2 tok/s decode and 1,191/1,121/1,058 tok/s prefill on two of the same cards; it crosses commits and is context, not a comparison — the control column above is the clean one.

## Cold prefill

Exact token arrays from this tree's own `README.md` + `RUNBOOK.md` + `CHANGELOG.md` at the tested commit (217,595 tokens), tokenized by the server's own `/v1/tokenize` and cut to each size, `/v1/completions` with `max_tokens=1`, 3 runs each, prompt cache off, fresh process per session.
The exact prompt token arrays are saved as [prefill-token-arrays.json](prefill-token-arrays.json), captured from the same checkpoint's tokenizer on the same world-4 mesh. Raw runs in [prefill.log](prefill.log) and [tp2-prefill.log](tp2-prefill.log).

## Exactness

Per [summary.json](summary.json), in both the world 4 and the world 2 control runs:

- `bench_openai` fibonacci-raw, 3 repeats at temperature 0: identical `output_sha256` across all repeats.
- `bench_concurrent`, every cell: `alone` comparisons **30 equal / 0 unequal** (3 per 1-stream cell, 12 per 4-stream cell), `serial` comparisons **4 equal / 0 unequal**, zero failed requests.
- Both concurrent aggregates came from runs whose replies matched their solo replies token for token.

## Skips and remaining coverage

- World 4 admission is geometry-based on HIP (heads, experts and vocabulary must split evenly); worlds other than 2 and 4 were not measured.
- No multi-host ranks (all ranks on one host, `--master 127.0.0.1`); the world 2 control used two of the four cards.
- No prompt-cache, recall or learned state: `--prompt-cache-gib 0` throughout.
- Only `--parallel 4 --context 4096` (decode) and `--parallel 1 --context 40960` (cold prefill) were measured; other windows are not covered.
- fp8 KV, drafter checkpoints, structured output and other checkpoints were not exercised.
- The qwen3.5/3.6/3.8 family's `serve_check.py` HTTP suite and `tf-qwen35-test` invariants do not apply to this family's checkpoint and were not run.

## Notes

- All four cards carried one rank each during the world 4 runs; the host's display shares no measured card.
- The rank logs ([decode-rank0.log](decode-rank0.log) … [decode-rank3.log](decode-rank3.log), [prefill-rank0.log](prefill-rank0.log) … [prefill-rank3.log](prefill-rank3.log)) retain each rank's own startup report. [provenance.json](provenance.json) records the build: the exact command, the exit code and the binary's SHA-256; [build.log](build.log) is the build's captured output, empty because the build was quiet on success.
