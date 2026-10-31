# gfx1151 Qwen3.8-Flash-Next HIP series: exactness, performance and memory

Receipt for [#1667](https://github.com/ashhart/TensorFold/pull/1667), the HIP Qwen3.8-Flash-Next series on gfx1151.
Measured on 2026-10-31, on an AMD Strix Halo (gfx1151) machine with 192 GiB of unified memory (184.6 GiB reported by
the kernel), ROCm 7.15 (HIP 7.15.26333), zig 0.17.0, Python 3.13.
The tested source is the tip of the series' code commits, **`37aed1a0384625d4ff35c23e7df2b502d2a8b7f6`** (`new`). The comparison is
`main` at **`94b90c9db65dcb71bbc39715901d6fb1f00407d0`** (`base`), the commit the series is based on.
The receipt commit adds only this folder. This receipt does not qualify other checkpoints, other GPUs, multi-GPU runs,
other platforms or machines with less memory.

## Checkpoint and binaries

- Checkpoint `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP`, revision `2b170fa6309d5d1ee380b35636075fac7945f286`.
  `perf/checkpoint.json` lists each file with its size and SHA-256.
- Binary SHA-256 values are in `perf/provenance.json` and `exactness/provenance.json`. Build:
  `zig build native-hip tf-flashnext-hip-test -Dhip=true -Dhip-arch=gfx1151 -Doptimize=ReleaseFast` with the ROCm
  `hipcc`, include and device-library paths. Every row of this receipt ran on these two binaries.
- `perf/logs/session-01.log` to `session-06.log` are the harness logs in time order: cold prefill at 2K to 32K, at 64K
  and 128K, at the native window, the decode clients, the 16K decode, and the memory A/B. `new` and `base` alternate
  in each round of each session.
- One server process ran at a time, under the machine's GPU lock. Each run started a fresh server and stopped it.
  No limit of the machine was raised.

## Exactness ([exactness/](exactness/))

Server: `--context 32768 --parallel 4 --prompt-cache-gib 1 --no-thinking`, `--metrics-open`.

| Check | Result |
| --- | --- |
| `zig/tests/qwen35/serve_check.py` | 15 ok, 0 fail; 1,285 of 1,688 drafted tokens accepted |
| Token-hash audit (identical request bodies, `draft` ignored) | 11 groups, 0 differ |
| `zig/tests/qwen35/serve_cancel.py` | 3 ok, 0 fail |
| `zig/tests/flashnext_hip/rewind.py`, prompt cache on against off | 0 replies differ; up to 10,240 cached tokens |
| `tf-flashnext-hip-test generate --chat --max 64 --hash --chunk C`, 10,368-token prompt | `c8f6888043c972e0` for C = 512, 2,048, 8,192 and 20,480 |

`serve_check.py` ran under `record_http.py`, which keeps each request and reply in `exactness/*-serve-check.jsonl`
and passes both on unchanged. `exactness/seam-corpus.json` names the public CPython library files of the seam prompt.

## Performance ([perf/](perf/))

`new` and `base` alternate in each round, 2 rounds. Greedy, seed 1234, thinking off, prompt cache off.
Prompts: token arrays of exactly N tokens from the top-level `.py` files of the CPython standard library
(`perf/prompts/corpus.json` names each file and its SHA-256). `perf/summary.md` has every row and every round.

- Cold prefill: `max_tokens` 1, a fresh server a run, `--context` N + 8,192 (262,144 for the 261,888-token row).
  Prefill tok/s = prompt tokens / TTFT.
- 16,384-token prompt: 256 and 1,024 output tokens, 3 reps each in 2 rounds, after one unrecorded 16-token request.
  Serial runs the server with `--no-drafts`; Default MTP runs the server defaults.
  Decode tok/s = (output tokens - 1) / (last token time - first token time).
- `tools/bench_openai.py` and `tools/bench_concurrent.py --levels 1,4 --alone --serial`, 256 tokens, 3 reps.

| Prompt tokens | 2,048 | 8,192 | 32,768 | 65,536 | 131,072 | 261,888 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `new` prefill tok/s | 673.7 | 558.2 | 513.5 | 496.7 | 471.6 | 437.3 |
| `base` prefill tok/s | 293.2 | 261.3 | 260.4 | 259.3 | 258.0 | 244.8 |

| 16,384-token prompt, 1,024 tokens | Prefill tok/s | Decode tok/s | TTFT (s) | Wall time (s) | Spec-decode hit rate (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `new` serial | 533.5 | 35.71 | 30.75 | 59.39 | - |
| `base` serial | 353.3 | 29.99 | 46.55 | 80.70 | - |
| `new` Default MTP | 533.5 | 49.05 | 30.94 | 51.86 | 61.8 |
| `base` Default MTP | 353.3 | 38.26 | 46.54 | 73.32 | 68.1 |

The Default MTP reply has the same token hash as the serial reply on both builds, and both builds give the same tokens.
Four concurrent streams, aggregate decode: 117.4 against 89.65 tok/s (code), 99.75 against 72.95 tok/s (chat);
concurrent equals alone and drafted equals plain on both builds.

## Memory ([perf/memory-ab/](perf/memory-ab/))

`memory_ab.sh`: a fresh `new` server at `--context 20480 --parallel 1 --prompt-cache-gib 0` for each setting,
2 rounds, one cold 16,384-token prefill each. `perf/memory-ab/summary.txt` lists the load line, the GPU peak over idle
and the RSS peak (bytes) and the TTFT of each run. Round 1:

| Setting | Load line, device | GPU peak over idle (GiB) | RSS peak (GiB) | 16K TTFT (s) |
| --- | ---: | ---: | ---: | ---: |
| `new` defaults (`prefill_step=4096`, resident n-gram rows) | 81.82 GiB | 82.73 | 104.4 | 30.74 |
| `new --policy prefill_step=20480` | 105.63 GiB | 106.5 | 104.5 | 29.69 |
| `new --policy prefill_step=1024` | 77.36 GiB | 78.27 | 102.6 | 33.90 |
| `new --policy rows=mapped`, rows not in the page cache | 81.82 GiB | 82.73 | 78.57 | 46.30 |
| `new --policy rows=mapped`, rows in the page cache | 81.82 GiB | 82.73 | 78.62 | 30.92 |
| `new --policy prefill_step=1024,rows=mapped` | 77.36 GiB | 78.27 | 78.62 | 34.13 |
| `base` | 76.58 GiB | 78.50 | 78.54 | 46.47 |

The prompt chunk sets the GPU memory; the resident copy of the 29.80 GiB of n-gram tables sets the RSS.
The first `rows=mapped` run follows a run whose copy at load dropped the rows from the page cache; the second follows it.

- Process: peak resident set of the server. Device: `mem_info_gtt_used` + `mem_info_vram_used` of the card, sampled at
  1 Hz, as the peak over the idle value read before the server started. The two overlap on unified memory: do not add them.
- Each server run has a summary in `perf/runs/*.json` with its load line, and its samples in `perf/memory/`.

## Build, test and kernel contract ([checks/](checks/))

On `new`:

- `contract.log`: `SUMMARY gfx1151 83 kernels: 83 pass gates 1-4, 0 fail, 4 known findings; 5 controls: 5 red as
  planted, 0 missed; 0 limit problems`.
- `limits.log`: `PASS kernel-contract-limits: every registered launch fits gfx1030, gfx1100, gfx1151 and gfx1201,
  every record current, every control red`.
- `test-hip.log` (`zig build test -Dhip=true`), `hip-kernel-test.log`, `test-cuda-host.log`: exit 0.
  Zig prints a passing test's stderr under a `failed command:` line.
- `perf/hash-new.log` and `perf/hash-base.log`: `tf-flashnext-hip-test generate --chat --max 64 --hash`, logits hash
  `243ecea88fe50d9f` on `new` and on `base`.

`per-commit.tsv`: each of the 33 code commits, built and tested alone on an earlier `main` (`abf822585`):
`zig build native-hip tf-flashnext-hip-test`, `zig build test -Dhip=true`, `zig build hip-kernel-test`, then
`tf-flashnext-hip-test generate --chat --max 64 --hash`. Every exit code is 0 and every hash is `243ecea88fe50d9f`.
The table names the tested commit and the commit in this pull request. Their added and removed lines are the same,
with two exceptions: commits 19-33 ran before their gate-5 records were refreshed (`hip_records_*.zig` only), and
commit 5 ran with a 20,480-row default chunk, which this pull request sets to 4,096 rows.

## Not run

- Metal and CUDA device tests (no such hardware). HIP devices other than gfx1151: only the limits check.
- Sampled (temperature above 0) decode speed. Multi-GPU. Other checkpoints.
- No standard-server comparison: `mlx_lm` needs Metal.

## Publication

Each subfolder is a sanitized copy. Machine-local paths, the host name and the user name are replaced with
placeholders (`publication.json` counts the replacements per file). Request and reply values, token ids, token hashes
and telemetry are unchanged. No model weights or credentials are included. `SHA256SUMS` lists every file but itself.
