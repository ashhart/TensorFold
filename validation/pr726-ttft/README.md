## What this changes

This receipt covers the BF16 text overlap of [PR726](https://github.com/ashhart/TensorFold/pull/726) with the earlier [PR608 TTFT contribution](https://github.com/machinegenieorg/TensorFold/pull/2). The source base is PR726 `d49a906f`; candidate runtime commit is `5e08e130`. The 31 runtime/test files are pinned in `provenance.json`.

The candidate adds committed first-token delivery before initial drafts, whole-identical-short-prompt snapshots, target and draft allocation reuse, raw first-logit reuse, and guarded omission of initialization fills. All five `TF_PR608_EXACT_*` switches remain default off. The latter four require `TF_PR608_EXACT_PROMPT_CACHE=1`. Exact snapshots admit at most 512 prompt tokens and use a 256 MiB budget; target/draft pools have 3 GiB/1.5 GiB retained budgets, at most 16 owners, and normal target capacity at most 512. Public target factories remain initialized; unsupported FP8/image requests decline the exact paths.

PR726's host prefix cache remains separate. It can restore a 4096-token long-chat prefix; the short cache restores the whole identical prompt, not a general prefix. Appending conversation turns does not automatically produce a short-cache hit.

Credit @philip-pentatonic for the Qwen CUDA engine and DFlash work in [PR608](https://github.com/ashhart/TensorFold/pull/608), and @BobClawblaw for PR726's host-prefix, vision and FP8 work. Jonathan Regan-Beasley contributed and tested this integration in a personal capacity.

## Receipt

One fresh process per arm, in order **B, D, A, C**: A stock/cache-off; B stock/1 GiB host cache; C candidate/cache-off; D candidate/1 GiB host cache. C/D enable all five switches. This is one sequential correctness panel, not balanced timing. Hardware was DGX Spark GB10, Linux aarch64, with Zig 0.17.0. Inference ran as a native host process. CPU-only builds used the immutable container image pinned in `provenance.json`; both binaries used common AOT/fatbins. Driver/library versions were not independently retained.

Each arm ran code14 and chat31: first use, one repeat and one n16 burst, all with 256 output tokens. Two genuine long-chat requests used 32 output tokens. Context was 32768, parallel 16, BF16 KV, greedy seed 1234, no thinking, and ignore-EOS. “First use” means a prompt in an already-loaded process; chat follows code. Four live `/tokenize` checks matched each arm's frozen fixture. All 152 literal request bodies, output SHA256 values and token fingerprints matched across arms. The accepted actual-CUDA gate and 439/439 Linux host checks are attested separately.

Selected **B → D** diagnostics (TTFT milliseconds; client decode tokens/s; full-response or burst wall seconds):

| Request | TTFT | Decode | Full response / wall |
|---|---:|---:|---:|
| Code first n1 | 140.04 → 140.93 | 69.48 → 65.44 | 3.810 → 4.037 |
| Code repeat n1 | 125.53 → 12.39 | 69.50 → 65.35 | 3.795 → 3.914 |
| Code n16 | 1066.41 → 144.74 | 23.47 → 25.31 | 11.937 → 10.223 |
| Chat first n1 | 131.17 → 132.35 | 49.80 → 49.15 | 5.252 → 5.320 |
| Chat repeat n1 | 127.84 → 13.79 | 49.78 → 49.08 | 5.250 → 5.209 |
| Chat n16 | 1088.51 → 153.76 | 17.54 → 18.76 | 15.628 → 13.749 |
| Long follow-up | 351.15 → 338.28 | See request evidence | 0.944 → 1.025 |

N16 TTFT/decode are per-stream medians. Slowest TTFT and all four arms are retained in `summary.json`; burst wall runs from earliest send to latest completion. TTFT starts at send and ends at the first nonempty SSE text. Decode is `(tokens−1)/(last−first)`; it is not aggregate throughput. Code repeat loses about 6% decode speed and 119 ms overall; cold code costs 227 ms more and long follow-up costs 81 ms more. Identical output does not imply equal work: code n1 uses 45 rounds in B and 48 in D.

Offline verification needs only Python 3.11+:

```sh
python3 -B validation/pr726-ttft/verify.py --source-root .
python3 -B -m unittest discover -s validation/pr726-ttft -p test_verify.py
```

To repeat a panel against an already-running pinned server, use the included standard-library client:

```sh
python3 -B validation/pr726-ttft/run_overlap.py --base "$BASE_URL" \
  --model qwen27-overlap --arm D --suite both \
  --long-manifest validation/pr726-ttft/long-manifest.json --output D-client.json
```

Start a fresh server per arm with the recorded context/parallel/precision/settings, `TENSORFOLD_CUDA_KERNELS="$AOT"`, reserve 8 GiB, and `--prompt-cache-gib 0` for A/C or `1` for B/D. Use the stock binary for A/B and candidate for C/D. The client does not start servers or load models. Binary/model/kernel hashes, original receipt hashes and exact request bodies are retained; no weights or operational scripts are included.

## Limits

D's host-prefix hook disables pool reuse and reset omission: both pools record zero reuse and zero avoided fills. Exact snapshots and first-logit reuse do show 34 hits. This is not evidence that all mechanisms executed with the host cache on. The device qualifier used private build-only registration; its source is excluded and its accepted gates/hash receipt are retained.

FP8, images, higher limits, error paths and other models were not qualified by this HTTP panel. The preserved failing gates preceded this final fixture; R3 changes only the genuine chat-history test, not runtime. Timings need repeated balanced ordering before stable performance magnitudes can be claimed. Investigating the remaining speculative-work differences and conservative host-hook exclusions is further work, not an implemented optimization.
