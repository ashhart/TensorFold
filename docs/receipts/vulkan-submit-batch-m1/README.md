# Batched submission receipts, Apple M1 (13-inch)

Qwen3.5-2B 4-bit, 64 greedy steps, two runs per process. Old build is #2015's head (the tile build, `b6d9a35b`), new build is #2016's head (adds batched submission, `d9b28417`). Both include the float32 fix.
Each case ran three old and three new processes, interleaved in both orders. Run 2 is the prefill plus 63 decode steps plus the per-step logits readback. Linux 7.3.0-2-20261110.1-lab-ARCH, Honeykrisp from the Omarchy Mesa build (library sha256 `d75f4cbc`).

| Case | Old run 2 (median) | New run 2 (median) | Gain: median (pair range) | Logits digest, all six processes |
|---|---:|---:|---|---|
| unicode-24 | 8.0 s | 3.4 s | 57.8% (57.0 to 59.5) | `e946185d64ae` |
| document-256 | 15.1 s | 9.9 s | 34.1% (34.0 to 36.8) | `3188aaae7d11` |
| document-1024 | 38.7 s | 32.3 s | 16.6% (16.2 to 17.3) | `82c353787e89` |

The gain depends on the environment. The first version of this receipt measured 68.2%, 48.8% and 30.5% on earlier builds with Linux 7.1.12-2-20261013.18-sep-ARCH and Mesa git-9323171c12 (digests `e5447d9a3ab5`, `147a2d2e0232`, `7f9d8e65e90e`). Running the earlier document-256 builds (old `151319df`, new `e84e79dd`) in the current environment gave 20.8 s and 16.6 s, a 20.5% gain, with the earlier digest `147a2d2e0232`. Every run time here is slower than in the first version, and these runs do not separate the cause. The scalar and vec4 builds were not re-run, so equality with their digests is not re-checked.

Files:

- `fixed-heads-summary.json` holds every process row (with peak resident memory), the pairs, the ELF hashes, the setup, and the earlier-builds run described above. It is generated from the raw run files.
- `summary.json` is the first version's summary (earlier builds, earlier environment). It stays as the record of those runs.
- `identity-probe.txt` holds the stage 1 probe run on the GPU with the earlier builds: 21 of 21, and the tiled image equal to the default image on every projection case. It was not re-run on the fixed heads.
- `kernel-contract-m1-13-inch.txt` holds the contract runner's output on the GPU with the earlier builds. It was not re-run on the fixed heads.
- `PREREG.md` was written before any GPU run. It holds the forecast, the gates and the decision rule.
- `raw-files-sha256.txt` lists the SHA-256 of every raw run file `summary.json` was generated from.

Byte identity: each process writes the SHA-256 of its full logits stream. A case passes only if all six digests are equal and the ids match the pinned ids.
