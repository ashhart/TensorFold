# qmm_q4_tile_rows receipts, Apple M1 (13-inch)

Qwen3.5-2B 4-bit, 64 greedy steps, two runs per process. Old build is the batched-submission build (`b7770d64`), new build has `qmm_q4_tile_rows` (`9eef5a52`).
The kernel computes four rows of x and two columns (16 apart in a 32-column tile) a thread. Each case ran three old and three new processes, interleaved in both orders.
Run 2 is the prefill plus 63 decode steps plus the per-step logits readback.

| Case | Old run 2 (median) | New run 2 (median) | Gain: median (pair range) | Logits digest, all six processes |
|---|---:|---:|---|---|
| unicode-24 | 2.2 s | 2.1 s | 5.9% (5.5 to 6.2) | `e5447d9a3ab5` |
| document-256 | 5.2 s | 3.7 s | 28.7% (28.7 to 29.0) | `147a2d2e0232` |
| document-1024 | 15.7 s | 9.8 s | 37.8% (37.8 to 37.8) | `7f9d8e65e90e` |

How the kernel got its shape. Each row below is a two-arm pair on the same inputs, three pairs a case, with the same gates:

| New arm | Old arm | Case | Old run 2 | New run 2 | Median change | Range over pairs |
|---|---|---|---|---|---|---|
| four rows, two columns (this kernel) | four rows, one column (this kernel before) | document-256 | 3.91 s | 3.69 s | +5.6% | +5.6% to +6.4% |
| four rows, two columns (this kernel) | four rows, one column (this kernel before) | document-1024 | 10.59 s | 9.79 s | +7.6% | +7.4% to +7.6% |
| eight rows, one column, arrays (rejected) | four rows, one column | document-256 | 3.89 s | 4.08 s | -4.8% | -4.9% to -4.6% |
| eight rows, one column, arrays (rejected) | four rows, one column | document-1024 | 10.59 s | 11.35 s | -7.1% | -7.3% to -7.1% |
| eight rows, one column, no arrays (rejected) | four rows, one column | document-256 | 3.89 s | 4.09 s | -5.2% | -5.6% to -5.0% |
| eight rows, one column, no arrays (rejected) | four rows, one column | document-1024 | 10.58 s | 11.38 s | -7.6% | -7.8% to -7.1% |

Eight rows a thread is slower than four in both forms, so the limit is register pressure, not the compiler's handling of arrays. Two columns share each unpacked x between two weights and keep fewer values live than eight rows.

Files:

- `summary.json` holds every process row (with peak resident memory), every pair, the ELF hashes, the setup and the decision rule. It is generated from the raw run files.
- `identity-probe.txt` holds the stage 1 probe on the GPU: 25 of 25, and the tiled path equal to the default image on every projection case, including row counts of 4, 5, 7 and 33.
- `kernel-contract-m1-13-inch.txt` holds the contract runner's output on the GPU.
- `PREREG.md` was written before the GPU runs it covers. It holds the forecasts, the gates and the decision rules, in the order the experiments ran.
- `raw-files-sha256.txt` lists the SHA-256 of every raw run file the summary was generated from.

Byte identity: each process writes the SHA-256 of its full logits stream. A case passes only if all six digests are equal and the ids match the pinned ids.
The digests equal the ones the scalar, vec4, tile and batched builds gave on the same inputs.
