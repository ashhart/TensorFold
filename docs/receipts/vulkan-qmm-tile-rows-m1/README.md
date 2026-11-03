# qmm_q4_tile_rows receipts, Apple M1 (13-inch)

Qwen3.5-2B 4-bit, 64 greedy steps, two runs per process. Old build is the batched-submission build (`b7770d64`), new build adds `qmm_q4_tile_rows` (`a3e7d070`).
Each case ran three old and three new processes, interleaved in both orders. Run 2 is the prefill plus 63 decode steps plus the per-step logits readback.

| Case | Old run 2 (median) | New run 2 (median) | Gain: median (pair range) | Logits digest, all six processes |
|---|---:|---:|---|---|
| unicode-24 | 2.2 s | 2.2 s | 4.1% (3.4 to 4.5) | `e5447d9a3ab5` |
| document-256 | 5.2 s | 3.9 s | 24.8% (24.1 to 25.0) | `147a2d2e0232` |
| document-1024 | 15.7 s | 10.6 s | 32.4% (32.3 to 32.4) | `7f9d8e65e90e` |

Files:

- `summary.json` holds every process row (with peak resident memory), every pair, the ELF hashes, the setup and the decision rule. It is generated from the raw run files.
- `identity-probe.txt` holds the stage 1 probe on the GPU: 25 of 25, and the tiled path equal to the default image on every projection case, including row counts of 4, 5, 7 and 33.
- `kernel-contract-m1-13-inch.txt` holds the contract runner's output on the GPU.
- `PREREG.md` was written before any GPU run. It holds the forecast, the gates and the decision rule.
- `raw-files-sha256.txt` lists the SHA-256 of every raw run file the summary was generated from.

Byte identity: each process writes the SHA-256 of its full logits stream. A case passes only if all six digests are equal and the ids match the pinned ids.
The digests equal the ones the scalar, vec4, tile and batched builds gave on the same inputs.
