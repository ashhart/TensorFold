# qmm_q4_tile receipts, Apple M1 (13-inch)

The "old" arm is #1990 built from source commit `46571f111b2f0e92963561702b7eaab41868f823`, ELF SHA-256
`2a1d481c692900c1cefb08fa308d5852d2ec748ee7bc24c89ff013c26ac36d72` (6,089,352 bytes). The "new" arm is #2015 built
from source commit `6e6f47f228a662cb26dedf5b24bf7c50c780fe43`, ELF SHA-256
`b3b7f5e17752b31370490a7d4fe3f4ff92cf3593ed3bed447759f52c5561de41` (6,128,512 bytes). Both include the float32
fix. These two ELFs produced every time and digest in this file. The ELFs predate comment- and name-only commits.
The retained receipt does not record the shell build commands. The branch now sits on #1990's later head
`70ff845318d32e99e0598c82dc5c388f8b7a869d` (266d2f62 merged with 8267ec1). That head was measured separately
against its scalar twin in #1990's comments; no table here comes from it.
Each case ran three old and three new processes, interleaved in both orders (old,new,new,old,old,new). Every process ran under `taskset` on the four P cores (CPU 4-7). Run 2 is the prefill plus 63 decode steps plus the per-step logits readback. Linux 7.3.0-2-20261110.1-lab-ARCH, started 2026-11-12 16:16Z, GPU timeouts 0. All 18 processes returned 0 with the pinned ids and the pinned logits digest.

| Case | Old run 2 (median) | New run 2 (median) | Gain: median (pair range) | Logits digest, all six processes |
|---|---:|---:|---|---|
| unicode-24 | 11.8 s | 8.4 s | 28.6% (27.9 to 31.2) | `6bfbb1a55a3e` |
| document-256 | 26.9 s | 12.2 s | 54.4% (53.9 to 61.0) | `8901e17ccef7` |
| document-1024 | 76.9 s | 25.8 s | 66.8% (66.1 to 66.9) | `a472a62c7e22` |

`pinned-heads-summary.json` holds every process row of this run. It is generated from the raw run output.

An earlier run on the same chip did not pin the cores (old build `1e1b7d8c`, new build `b6d9a35b`, started 2026-11-11 19:15Z). Its gains were lower and its logits digests differ from the ones above; the two runs do not say which of the changes caused that:

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

- `pinned-heads-summary.json` holds every process row, the pairs, the ELF hashes and the setup of the pinned run in the first table. It is generated from the raw run output.
- `fixed-heads-summary.json` is the same for the unpinned run of 2026-11-11 in the second table.
- `summary.json` is the first version's summary (earlier builds, earlier environment, both tile builds). It stays as the record of those runs.
- `runners-2a5a06c2/` holds the output of the ten contract and stage runners at #2015's head (`2a5a06c2`, 2026-11-12, GPU timeouts 0), one file per runner, with `log.txt` listing the exit codes. `tf-vulkan-kernel-contract.txt` includes `qmm_q4_tile_f32`: four kernels pass gates 1 to 4, eight controls caught. `tf-stage1-gpu.txt` holds the projection cases with `tile=identical` (the tiled image against the default image, byte for byte, on the bf16 projections). No byte-level comparison of the f32 tile image against `qmm_f32` exists; the equal logits digests of the two heads are the identity evidence for it.
- `PREREG.md` was written before the first GPU run. Its addendum, written after the final pairs, says what changed and why.
- `raw-files-sha256.txt` lists the SHA-256 of every raw run file the summary was generated from (`final/` and `interim/`) for `summary.json`.

Byte identity: each process writes the SHA-256 of its full logits stream. A case passes only if all six digests are equal and the ids match the pinned ids.
The digests of #1990's head and #2015's head are equal on every input.
