# Batched submission receipts, Apple M1 (13-inch)

Qwen3.5-2B 4-bit, 64 greedy steps, two runs per process. Old build is the tile build (`618ed699`), new build adds batched submission (`5f8d4c3b`).
Each case ran three old and three new processes, interleaved in both orders. Run 2 is the prefill plus 63 decode steps plus the per-step logits readback.

| Case | Old run 2 (median) | New run 2 (median) | Gain: median (pair range) | Logits digest, all six processes |
|---|---:|---:|---|---|
| unicode-24 | 7.0 s | 2.3 s | 68.2% (67.4 to 69.6) | `e5447d9a3ab5` |
| document-256 | 10.2 s | 5.2 s | 48.8% (47.5 to 52.5) | `147a2d2e0232` |
| document-1024 | 22.7 s | 15.7 s | 30.5% (29.5 to 31.1) | `7f9d8e65e90e` |

Files:

- `summary.json` holds every process row (with peak resident memory), every pair, the ELF hashes, the setup and the decision rule. It is generated from the raw run files.
- `identity-probe.txt` holds the stage 1 probe run on the GPU with this change: 21 of 21, and the tiled image equal to the default image on every projection case.
- `kernel-contract-m1-13-inch.txt` holds the contract runner's output on the GPU.
- `PREREG.md` was written before any GPU run. It holds the forecast, the gates and the decision rule.
- `raw-files-sha256.txt` lists the SHA-256 of every raw run file the summary was generated from.

Byte identity: each process writes the SHA-256 of its full logits stream. A case passes only if all six digests are equal and the ids match the pinned ids.
The digests equal the ones the scalar, vec4 and tile builds gave on the same inputs.
