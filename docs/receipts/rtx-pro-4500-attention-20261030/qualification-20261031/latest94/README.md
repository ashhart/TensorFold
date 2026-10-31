Raw evidence is packaged in [evidence.tar.gz](evidence.tar.gz) to keep the PR file list reviewable. It contains the complete original phase snapshot, including raw JSON/logs, reproduction sources/fixtures and their original SHA manifests. `archive-members.sha256.json` hashes every payload member. Extract into an empty directory with `tar -xzf evidence.tar.gz`, then verify those SHA256 values with Python/hashlib. Summaries below remain inline. No measurement or source input was changed; no executable, library or model weights are included.

# Local rebase qualification on 94b90c9d

The code-only contribution was applied independently to 94b90c9db65dcb71bbc39715901d6fb1f00407d0. These are new binaries, not the old public PR binaries. Same-binary override 0/1/0, fresh calibration per boot; medians compared against geometric mean of bracketing stock medians. All matched tokens, features and fallback checks pass. Full context 131072, vision, reasoning, tools, FP8 KV, int8 prefill, DFlash2, 8 GiB host cache, parallel1, 200 W and upstream 0.5 GiB reserve remain enabled. Version1.1.6 is recipe compatibility metadata; source/binary hashes are authoritative.

| Decode metric | Gain |
|---|---:|
| community geometric mean | -0.379% |
| classic geometric mean | +0.018% |
| long geometric mean | +2.240% |
| 8192/draft1 | +2.390% |
| 8192/draft0 | +1.066% |
| 32768/draft1 | +3.474% |
| 32768/draft0 | +2.045% |

One bracket gives no confidence interval. Short community results vary by cell; raw negative cells are retained. No physical5090/PRO6000 claim. The full CUDA contract SUMMARY remains outstanding after resource failures; served tests do not replace that gate. This local rebase and evidence are unpublished. Historical public PR source01e25d3a results remain separate.

Broader [host checks](host-checks/README.md) retain the inherited fixture crashes and explicitly separated passing common test-only-overlay run;108 fresh tests and cached remainder,285/285 steps. No unmodified-source or fullGPU pass is inferred.
