Raw evidence is packaged in [evidence.tar.gz](evidence.tar.gz) to keep the PR file list reviewable. It contains the complete original phase snapshot, including raw JSON/logs, reproduction sources/fixtures and their original SHA manifests. `archive-members.sha256.json` hashes every payload member. Extract into an empty directory with `tar -xzf evidence.tar.gz`, then verify those SHA256 values with Python/hashlib. Summaries below remain inline. No measurement or source input was changed; no executable, library or model weights are included.

# Independent tile override bracket on RTX PRO 4500

One compiled native binary, flag `TF_CUDA_TEST_MLP128` set to `0`, then `1`, then `0`. All three boots completed with fresh calibration and matched request history. All community/classic token hashes and feature/fallback dictionaries match. Long-context draft/plain hashes also match where measured. This is physical 4500 qualification of the override, not an RTX 5090 or PRO 6000 result.

The full serving settings are retained: 131072 context, vision, tools, reasoning, FP8 KV, int8 prefill, BF16 DFlash2, 8 GiB host prompt cache, parallel 1, headless compute mode, 200 W, upstream minimum 0.5 GiB GPU reserve. The explicit version label 1.1.6 is recipe compatibility metadata; source and binary hashes identify the implementation.

Medians within each cell are compared against the geometric mean of bracketing stock medians. One bracket does not establish a confidence interval. No samples from the historical zero-reserve protocol are pooled here.

| Metric / cell | Stock before | Candidate | Stock after | Gain |
|---|---:|---:|---:|---:|
| community: code/T1 | 233.278 | 238.444 | 230.904 | +2.739% |
| community: chat/T1 | 145.319 | 145.557 | 144.145 | +0.571% |
| community: code/T0 | 217.114 | 217.693 | 215.602 | +0.618% |
| community: chat/T0 | 165.713 | 166.102 | 164.688 | +0.546% |
| **community geometric mean** | | | | **+1.114%** |
| classic: fibonacci-raw | 215.766 | 217.062 | 215.316 | +0.706% |
| classic: gpu-chat-no-think | 164.824 | 165.820 | 164.517 | +0.698% |
| **classic geometric mean** | | | | **+0.702%** |

Rates are decode tokens/s, separate from prefill/TTFT. The MLP community code/T1 cell is higher than its other cells and is retained as observed variability. The attention short-context cells are essentially flat. These new long-context measurements are reported independently of the earlier approximately 4% drafted result.

comparison.json contains exact medians, hashes, feature results, memory samples and binary identity. state.json retains calibration cost tables and draft counters. Per-boot raw responses/server logs and source/fixtures are included. Runtime weights, binaries, libraries and toolchain are not repackaged. The retained community client SHA is checked by the worker. Runtime recipe/environment identities are in the parent receipt.

The full CUDA contract SUMMARY gate remains outstanding after the separately documented resource failures. Passing served checks do not replace it. All new receipt updates remain local and unpublished.
