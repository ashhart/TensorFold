# qmm_q4_vec4 receipts, Apple M1 (13-inch)

Qwen3.5-2B 4-bit, 64 greedy steps, two runs per process. Old build is #1544 (`b3ab9ba2`), new build adds `e742e023`.
Each case ran three old and three new processes, interleaved in both orders. Run 2 is the prefill plus 63 decode steps plus the per-step logits readback.

| Case | Old run 2 (median) | New run 2 (median) | Gain: median (pair range) | Logits digest, all six processes |
|---|---:|---:|---|---|
| unicode-24, first pair | 15.6 s | 9.4 s | 40.1% (40.0 to 41.0) | `e5447d9a3ab5` |
| unicode-24, replicate | 15.3 s | 9.0 s | 41.0% (39.8 to 41.3) | `e5447d9a3ab5` |
| document-256 | 40.0 s | 20.7 s | 48.5% (48.2 to 48.5) | `147a2d2e0232` |
| document-1024 | 122.8 s | 59.5 s | 51.5% (51.5 to 51.5) | `7f9d8e65e90e` |

Files:

- `summary.json` holds every process row, every pair, the ELF hashes, the setup and the decision rule. It is generated from the raw run files.
- `identity-probe.txt` holds the probe that runs both images on 17 projection cases and compares the output bytes.
- `PREREG.md` was written before any GPU run. It holds the forecast, the gates and the decision rule.
- `raw-files-sha256.txt` lists the SHA-256 of every raw run file the summary was generated from.
- `kernel-contract-*.txt` holds the contract runner's output, one file per GPU.

Byte identity: each process writes the SHA-256 of its full logits stream. A case passes only if all six digests are equal and the ids match the pinned ids.
