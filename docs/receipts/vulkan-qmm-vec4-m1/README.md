# qmm_q4_vec4 receipts, Apple M1 (13-inch)

Qwen3.5-2B 4-bit, 64 greedy steps, two runs per process. Old build is #1990's head built with the vec4 selection forced to the scalar kernel (`qmmVariant = .scalar` in `ops_q4.zig`, not committed; ELF `4d1c6ba6`), new build is #1990's head (`1e1b7d8c`). The float32 fix sits above the vec4 commits, so an older #1544 build would not carry it.
Unicode-24 and document-256 ran three old and three new processes (old,new,new,old,old,new). Document-1024 ran two pairs, one old,new and one new,old, because the scalar arm takes about 250 s a process. Run 2 is the prefill plus 63 decode steps plus the per-step logits readback. Linux 7.3.0-2-20261110.1-lab-ARCH, Honeykrisp from the Omarchy Mesa build (library sha256 `d75f4cbc`).

| Case | Old run 2 (median) | New run 2 (median) | Gain: median (pair range) | Logits digest, all processes |
|---|---:|---:|---|---|
| unicode-24 | 15.6 s | 10.2 s | 34.8% (34.6 to 35.2) | `e946185d64ae` |
| document-256 | 40.4 s | 23.8 s | 41.0% (41.0 to 41.0) | `3188aaae7d11` |
| document-1024 (two pairs) | 123.0 s | 70.1 s | 43.0% (43.0 to 43.0) | `82c353787e89` |

The gain depends on the environment. The first version of this receipt measured 40.1%, 48.5% and 51.5% against #1544's head (`b3ab9ba2`; new build `e742e023`) on earlier builds with Linux 7.1.12-2-20261013.18-sep-ARCH and Mesa git-9323171c12 (digests `e5447d9a3ab5`, `147a2d2e0232`, `7f9d8e65e90e`). These runs do not separate the cause.

First version's table (earlier builds, earlier environment):

| Case | Old run 2 (median) | New run 2 (median) | Gain: median (pair range) | Logits digest, all six processes |
|---|---:|---:|---|---|
| unicode-24, first pair | 15.6 s | 9.4 s | 40.1% (40.0 to 41.0) | `e5447d9a3ab5` |
| unicode-24, replicate | 15.3 s | 9.0 s | 41.0% (39.8 to 41.3) | `e5447d9a3ab5` |
| document-256 | 40.0 s | 20.7 s | 48.5% (48.2 to 48.5) | `147a2d2e0232` |
| document-1024 | 122.8 s | 59.5 s | 51.5% (51.5 to 51.5) | `7f9d8e65e90e` |

Files:

- `fixed-heads-summary.json` holds every process row (with peak resident memory), the pairs, the ELF hashes and the setup of the fixed-head run. It is generated from the raw run files.
- `summary.json` is the first version's summary (earlier builds, earlier environment). It stays as the record of those runs.
- `identity-probe.txt` holds the probe that runs both images on 17 projection cases and compares the output bytes. It ran on the earlier builds and was not re-run on the fixed head.
- `PREREG.md` was written before any GPU run. It holds the forecast, the gates and the decision rule.
- `raw-files-sha256.txt` lists the SHA-256 of every raw run file the summary was generated from.
- `kernel-contract-*.txt` holds the contract runner's output, one file per GPU, from the earlier builds. It was not re-run on the fixed head.

Byte identity: each process writes the SHA-256 of its full logits stream. A case passes only if all six digests are equal and the ids match the pinned ids.
