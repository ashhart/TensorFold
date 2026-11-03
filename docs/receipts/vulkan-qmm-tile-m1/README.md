# qmm_q4_tile receipts, Apple M1 (13-inch)

Qwen3.5-2B 4-bit, 64 greedy steps, two runs per process. Old build is the vec4 build (`e742e023`), new build is `618ed699`.
Each case ran three old and three new processes, interleaved in both orders. Run 2 is the prefill plus 63 decode steps plus the per-step logits readback.

| Case | Old run 2 (median) | New run 2 (median) | Gain: median (pair range) | Logits digest, all six processes |
|---|---:|---:|---|---|
| unicode-24 | 9.2 s | 7.3 s | 19.3% (18.3 to 24.3) | `e5447d9a3ab5` |
| document-256 | 20.5 s | 10.2 s | 49.9% (48.5 to 50.6) | `147a2d2e0232` |
| document-1024 | 59.7 s | 22.3 s | 62.6% (61.4 to 64.0) | `7f9d8e65e90e` |

An earlier measurement, on the build before the last tile was padded (`a2dcaa1d`; the n = 16 projections kept the vec4 image):

| Case | Old run 2 (median) | New run 2 (median) | Gain: median (pair range) | Logits digest, all six processes |
|---|---:|---:|---|---|
| unicode-24 | 9.0 s | 6.9 s | 24.7% (19.5 to 27.6) | `e5447d9a3ab5` |
| document-256 | 20.6 s | 10.5 s | 48.8% (48.0 to 50.6) | `147a2d2e0232` |
| document-1024 | 59.6 s | 22.2 s | 62.8% (62.7 to 63.0) | `7f9d8e65e90e` |

Files:

- `summary.json` holds every process row (with peak resident memory), every pair, the ELF hashes, the setup and the decision rule, for both builds. It is generated from the raw run files.
- `identity-probe.txt` holds the probe that runs the tiled image and the default image on every projection case and compares the output bytes.
- `kernel-contract-m1-13-inch.txt` holds the contract runner's output on the GPU.
- `PREREG.md` was written before the first GPU run. Its addendum, written after the final pairs, says what changed and why.
- `raw-files-sha256.txt` lists the SHA-256 of every raw run file the summary was generated from (`final/` and `interim/`).

Byte identity: each process writes the SHA-256 of its full logits stream. A case passes only if all six digests are equal and the ids match the pinned ids.
The digests equal the ones the scalar and vec4 builds gave on the same inputs.
