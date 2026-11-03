# Dense BF16 Qwen3.5 prompt-score qualification

Partial receipt for #1136 and draft PR #2020. The complete strict CUDA run
finished with exit 1 because of launch-limit problems. Served performance,
lifecycle, cold-prefill, existing-format and memory receipts are still pending.
These files establish only the checks stated below.

## Source and environment

[manifest.json](manifest.json) identifies the exact source, stacked parent,
inference/kernel trees and native binary. The serving source is unchanged from
its build at `327ef639`; subsequent changes affect documentation, the independent
reporter and contract fixtures. No later serving-source change is implied.

One NVIDIA RTX PRO 6000 Blackwell Max-Q workstation GPU, sm_120, rank 1,
96 GB class and 300 W limit; Linux x86_64, driver 615.78.08, Zig 0.17.0,
CUDA nvcc 13.3.73, ReleaseFast. The physical host is 128 GB class. Serving uses
four CPU cores and a 32 GiB container cap; this does not qualify a physical
32 GiB machine. Other GPUs/backends and multiple ranks are unrun.

Checkpoint and tokenizer: `Qwen/Qwen3.5-27B`, revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`. Stored BF16 projection weights,
BF16 activations, FP32 sums and the checkpoint's FP32 GDN normalization scales.
No private checkpoint, request fixture, new weights or drafter is distributed.

## API exactness

[api.json](api.json) records 32 passed checks, including drafted/plain,
scored/unscored, repeated and concurrent/solo output/score comparisons, top-k
admission, invalid requests and recovery. It records full reply token IDs and
SHA-256 hashes of tokens and FP32 score bits/ranks. One warmup is excluded.
It is an exactness matrix, not a fixed-prompt service-latency benchmark.

[fixtures.json](fixtures.json) contains the exact public prompt token arrays at
lengths 1, 32, 129, 257 and 4097. Their file hash is in the manifest and API
receipt. The 4097-token case crosses the native prompt-chunk seam.

Reproduce against a loopback native server with context 8192, parallel 4,
`--prompt-cache-gib 0 --no-thinking`:

```sh
python3 tools/zig/check_qwen35_prompt_scores.py http://127.0.0.1:8091 local-model docs/receipts/qwen35-bf16-scoring/fixtures.json --output api.json
```

## Independent reference

[fp64.json](fp64.json) passes six prompt rows, 12 decode rows and 24 named
projection captures. Worst absolute head error is `2.588785719126463e-6`;
worst absolute logprob error is `6.144032695942769e-6`. There are zero argmax
differences. Absolute-error coordinates and bound-ratio coordinates are recorded
separately, with selected/top/rank checks and per-capture bounds.

This computes FP64 head dots and full-vocabulary logsumexp on identical native
BF16 normalized rows. It is not a full-model FP64 forward. The projection
captures apply each named checkpoint matrix to deterministic synthetic BF16
rows; they do not establish layer-by-layer whole-model fidelity.

```sh
zig-out/bin/tf-qwen35-score-reference MODEL docs/receipts/qwen35-bf16-scoring/reference-fixtures.json captures
python3 tools/zig/qwen35_scores_fp64.py MODEL captures --revision fc05daec18b0a78c049392ed2e771dde82bdf654 --output fp64.json
```

See the [recipe](../../recipes/qwen35-bf16-scoring.md) for build commands,
precision contracts, intentional refusals and the remaining readiness gates.
No throughput improvement or external-provider comparison is established here.

## Complete CUDA contract

[cuda-contract.log](cuda-contract.log) retains the complete run, and
[cuda-contract.json](cuda-contract.json) identifies its binary, verdict and known findings.
The required command was run without a diagnostic subset or disabled gate:

```text
SUMMARY 550 kernels: 550 pass gates 1-4, 0 fail, 6 known findings; 365 controls: 365 red as planted, 0 missed; 342 limit problems; gates 4013421 ms, limits 726300 ms
exit: 1
```

All four new function records pass on the live sm_120 device. Six pre-existing
NaN findings reproduce: Qwen BF16/FP8 paged and contiguous attention, plus GLM
scores/index scores. No known entry or numerical tolerance was added to hide a
failure. The 342 limit problems comprise 188 registry problems and 154 live
record/device mismatches. This is a failed full contract, not a clean pass.

[limits-comparison.json](limits-comparison.json) records a separate parent and
candidate five-SM asset audit: both exit 1 with 637 problems and identical
639-line diagnostic multisets (including the two final failure lines). All 20
new function/SM records are current. This isolates no new asset-audit diagnostic;
it does not establish a parent full/live-only run, which was not run. Existing
record declarations are unchanged apart from appending the four new records.
Launch-limit catalog cleanup or a maintainer resolution remains a readiness gap.
