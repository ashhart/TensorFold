# qmm_q4_tile receipts, Apple M1 (13-inch)

Qwen3.5-2B 4-bit, 64 greedy steps, two runs per process. Old build is #1990's head (the vec4 build, `1e1b7d8c`), new build is #2015's head (`b6d9a35b`). Both include the float32 fix.
Each case ran three old and three new processes, interleaved in both orders (old,new,new,old,old,new). Run 2 is the prefill plus 63 decode steps plus the per-step logits readback. Linux 7.3.0-2-20261110.1-lab-ARCH, Honeykrisp from the Omarchy Mesa build (library sha256 `d75f4cbc`).

| Case | Old run 2 (median) | New run 2 (median) | Gain: median (pair range) | Logits digest, all six processes |
|---|---:|---:|---|---|
| unicode-24 | 10.2 s | 9.1 s | 10.9% (6.7 to 22.2) | `e946185d64ae` |
| document-256 | 23.8 s | 15.3 s | 35.6% (32.4 to 37.1) | `3188aaae7d11` |
| document-1024 | 70.2 s | 38.7 s | 44.9% (34.8 to 45.1) | `82c353787e89` |

The gain depends on the environment. The first version of this receipt measured 19.3%, 49.9% and 62.6% on earlier builds with Linux 7.1.12-2-20261013.18-sep-ARCH and Mesa git-9323171c12 (digests `e5447d9a3ab5`, `147a2d2e0232`, `7f9d8e65e90e`). These runs do not separate the cause.

First version's table (earlier builds, earlier environment; old build the vec4 build `e742e023`, new build `618ed699`):

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

- `fixed-heads-summary.json` holds every process row (with peak resident memory), the pairs, the ELF hashes and the setup of the fixed-heads run. It is generated from the raw run files.
- `summary.json` is the first version's summary (earlier builds, earlier environment, both tile builds). It stays as the record of those runs.
- `identity-probe.txt` holds the probe that runs the tiled image and the default image on every projection case and compares the output bytes. It ran on the earlier builds and was not re-run on the fixed heads.
- `kernel-contract-m1-13-inch.txt` holds the contract runner's output on the GPU with the earlier builds. It was not re-run on the fixed heads.
- `PREREG.md` was written before the first GPU run. Its addendum, written after the final pairs, says what changed and why.
- `raw-files-sha256.txt` lists the SHA-256 of every raw run file the summary was generated from (`final/` and `interim/`) for `summary.json`.

Byte identity: each process writes the SHA-256 of its full logits stream. A case passes only if all six digests are equal and the ids match the pinned ids.
The digests equal the ones the scalar and vec4 builds gave on the same inputs.
