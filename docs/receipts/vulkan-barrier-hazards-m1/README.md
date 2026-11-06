# Hazard-tracked barriers receipts, Apple M1 (13-inch)

Qwen3.5-2B 4-bit, 64 greedy steps, two runs per process. Old build is the two-columns build (`9eef5a52`, #2017); new build records a barrier between dispatches only when they conflict (`c03ebda7`).
Each case ran three old and three new processes, interleaved in both orders. Run 2 is the prefill plus 63 decode steps plus the per-step logits readback.

| Case | Old run 2 | New run 2 | Gain, median (pair range) | Logits digest, all six processes |
|---|---:|---:|---|---|
| unicode-24 | 2.11 s | 1.96 s | 7.1% (5.6 to 7.4) | `e5447d9a3ab5` |
| document-256 | 3.70 s | 3.52 s | 4.8% (4.7 to 5.3) | `147a2d2e0232` |
| document-1024 | 9.81 s | 9.49 s | 3.3% (2.7 to 3.4) | `7f9d8e65e90e` |

Why barriers. A microbenchmark of `qmm_q4_tile` on the real decode shapes (the 175 projections of one decode step, 400 dispatches a shape, batched, outputs byte-equal in every mode) times the engine's barrier after each dispatch against two variants:

| n x k | Dispatches a step | Barrier after each | Compute-only barrier | No barrier |
|---|---:|---:|---:|---:|
| 16 x 2048 | 36 | 85.1 us | 48.3 us | 3.0 us |
| 512 x 2048 | 12 | 62.7 us | 63.3 us | 12.8 us |
| 2048 x 2048 | 42 | 67.8 us | 65.8 us | 42.8 us |
| 2048 x 6144 | 24 | 204.7 us | 184.7 us | 122.8 us |
| 4096 x 2048 | 6 | 87.8 us | 87.8 us | 82.8 us |
| 6144 x 2048 | 54 | 122.7 us | 123.1 us | 122.0 us |
| 248320 x 2048 | 1 | 4890.8 us | 4848.4 us | 4826.8 us |
| decode step, projections only | 175 | 23.62 ms | 21.72 ms | 16.92 ms |

A compute-only barrier barely changes the cost, and in the engine it changed nothing: unicode-24 run 2 moved by -0.14% with equal digests. Removing barriers between dispatches that do not conflict is what pays.

Contract runners and probes on the GPU at both heads, exit 0 each, every line equal:

| Runner | Exit | Result at both heads (equal apart from elapsed milliseconds) |
|---|---|---|
| `tf-vulkan-kernel-contract` | 0 | SUMMARY Apple M1 (G13G B1) 4 kernels: 4 pass gates 1-4, 0 fail, 0 known findings; 8 controls: 8 caught, 0 missed; 0 limit failures |
| `tf-stage1-datamove` | 0 | RESULT datamove on "Apple M1 (G13G B1)": kernels=7 passed=7 failed=0 controls=0 known=0 |
| `tf-attention-contract` | 0 | RESULT attention-contract on "Apple M1 (G13G B1)": passed=15 total=15 |
| `tf-vulkan-greedy-contract` | 0 | SUMMARY vulkan greedy contract on "Apple M1 (G13G B1)": 1 kernel, 1 pass gates 1-4, 0 fail; 1 control: 1 red as planted, 0 missed |
| `tf-contract-pointwise` | 0 | RESULT contract-pointwise on "Apple M1 (G13G B1)": passed=16 known=1 total=17 |
| `tf-stage1-gpu` | 0 | RESULT stage1 on "Apple M1 (G13G B1)": passed=25 total=25 |
| `tf-stage2-gpu` | 0 | RESULT stage2 on "Apple M1 (G13G B1)": passed=9 total=9 |
| `tf-stage3-gpu` | 0 | RESULT stage3 on "Apple M1 (G13G B1)": passed=4 total=4 |
| `tf-stage4-gpu` | 0 | RESULT stage4 on "Apple M1 (G13G B1)": passed=6 total=6 |
| `tf-stage5-gpu` | 0 | RESULT stage5 on "Apple M1 (G13G B1)": passed=7 total=7 |

10 of 10 runners print the same lines at both heads.

The first build of this change failed its digest gate on unicode-24: logits digest `e8535a02dd1e` where the pinned one is `e5447d9a3ab5`. The greedy draw records three dependent dispatches with `Pipeline.record` and relied on the barrier that call used to add, and its first fill needs the earlier dispatches fenced. `record` keeps its barrier now, only `Pipeline.run` elides, and `Commands.sync` fences open dispatches before a direct recorder starts.

Files:

- `summary.json` holds every process row, every pair, the benchmark rows, the setup and the decision rule, generated from the raw run files.
- `identity-probe.txt` holds the stage 1 probe on the GPU: 25 of 25.
- `suites.txt` holds both heads' runner lines in full.
- `PREREG.md` was written before the GPU runs it covers, including a record of the first build's failure.
- `raw-files-sha256.txt` lists the SHA-256 of every raw run file the summary was generated from.

Byte identity: each process writes the SHA-256 of its full logits stream. A case passes only if all six digests are equal and the ids match the pinned ids.
