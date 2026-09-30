# Decision scoring: GB10 and MLX validation

## Environment and scope

Full `neko-legends/GLM-5.3-Flash-Uncensored-EXL3`, checkpoint revision
`07135ec082f8f11f7a71e4244a4e5167a0f96277`, on two GB10 hosts, TP=2.
DFlash2 was loaded, context 200000, thinking disabled for decisions.
The running deployment received the PR's cache/drafter isolation and the
updated decision validation/probability code on both ranks before restarting.

**The deployment also contains local GLM performance patches outside this PR.**
These are end-to-end measurements of that deployment, not a benchmark of an
unmodified upstream checkout, and not a throughput or concurrency benchmark.
Context was reduced from 205168 after startup admission refused that window.

## Full-model HTTP checks

Seven deliberately simple questions with known answers, all correct:

| Input / task | Expected | Result | Winning label probability | label_mass | HTTP seconds | Prompt tokens |
|---|---|---|---:|---:|---:|---:|
| Duplicate charge / category | billing | billing | 0.999395 | 0.993278 | 0.2560 | 45 |
| API HTTP 500 / category | technical | technical | 0.998296 | 0.991734 | 0.2691 | 50 |
| Quote and demo / category | sales | sales | 0.999177 | 0.997143 | 0.2584 | 49 |
| Monday delivery / delivered Monday? | yes | yes | 0.998190 | 0.032741 | 0.2465 | 32 |
| Monday delivery / delivered Friday? | no | no | 0.998755 | 0.014025 | 0.2114 | 32 |
| Explicit HIGH / levels LOW, MEDIUM, HIGH | 2 | mean 1.994257 | 0.996727 | 0.987006 | 0.2424 | 42 |
| Explicit NISKI / Polish priority levels | 0 | mean 0.011690 | 0.992365 | 0.983101 | 0.2740 | 59 |

Ten successful sequential single-question decision requests (seven cases plus
three probability checks): min **0.2114 s**, median **0.2455 s**, max **0.2740 s**.
Times are client-observed HTTP wall time, including LAN round trip, on an
already-loaded model. This small sample has no defensible p95 or throughput claim.
No text tokens were generated (`completion_tokens == 0`).

Additional checks passed:

- Reversing choice option order still selects billing.
- Multiple questions in one request return both answers correctly.
- Label probabilities are finite and normalized; label_mass is finite and within [0, 1].
- Returned label token IDs are distinct, one per option.
- Increasing temperature from 1 to 2 flattens label probabilities without changing label_mass.
- Temperature 1e-320 still gives finite probabilities.
- Invalid template kwargs (`[]`, false, unknown keys, thinking true) return HTTP 400.
- Greedy chat output is identical before and after decisions.
- A conversation continuation after a decision matches the same continuation without it.

The low yes/no label_mass is important: the relative probability among lowercase
`yes` and `no` is not calibrated confidence over every possible model response.
Seven correct examples demonstrate operation, not general decision accuracy.
This run did not compare logits against SGLang on the identical full checkpoint.

## Full-model MLX HTTP checks

Apple M4 Max, 48 GB; `Vontra/Qwen3.8-27B-MLX-4bit`, TensorFold PR code,
MLX backend, drafts off, context 1024, thinking off,
`TENSORFOLD_MEMORY_LIMIT_GB=26`. Loaded weights occupied 14.6 GiB;
startup reported 90.3 seconds. The test server used a separate localhost port
and was stopped afterwards.

The same seven known-answer questions as above passed **7/7**:

| Task | Result | Winning label probability | label_mass | HTTP seconds | Prompt tokens |
|---|---|---:|---:|---:|---:|
| Duplicate charge / category | billing | 0.999195 | 0.991777 | 0.4236 | 53 |
| API HTTP 500 / category | technical | 0.999767 | 0.998274 | 0.5381 | 60 |
| Quote and demo / category | sales | 0.999748 | 0.996657 | 0.7416 | 59 |
| Delivered Monday? | yes | 0.998901 | 0.896961 | 0.7300 | 40 |
| Delivered Friday? | no | 0.999569 | 0.672203 | 0.6706 | 40 |
| Explicit HIGH / score | mean 1.999942 / 2 | 0.999959 | 0.999327 | 0.6258 | 50 |
| Explicit NISKI / Polish score | mean 0.000214 / 2 | 0.999809 | 0.991759 | 0.8715 | 67 |

Ten sequential single-question requests: min **0.4236 s**, median **0.5982 s**,
max **0.8715 s**, 40–67 prompt tokens and zero completion tokens. These are
localhost client-observed wall times on an already-loaded model, not throughput
measurements. The first successful decision request is included; no warmup
request was discarded. Different models, tokenizers and hardware mean this is
not a controlled speed comparison with the GB10 run.

Passed the same extra checks: option reordering, multiple questions, finite and
normalized probabilities, distinct label IDs, temperature versus label_mass,
tiny temperature, invalid kwargs, and identical greedy chat and conversation
continuation before/after decisions. MLX yes/no label_mass was 0.672–0.897,
unlike the much lower mass on GLM; neither is a general confidence calibration.

## Regression coverage

- Host suite: **3196 passed, 739 skipped** (CUDA excluded).
- GB10 real CUDA kernels, tiny synthetic checkpoint: **4 passed** for MTP/DFlash2
  with zero/64 MiB snapshot budgets.
- Two actual GB10 ranks with NCCL: **4 PASS** for the same policy/budget matrix,
  comparing post-decision continuations and conversation switches with serial
  fresh-prefill results. The MTP save case verifies a prefix actually resumes.
- See `tests/cuda/glm_decision_tp.py` and `tests/cuda/GLM_DECISION_TP.md` for the
  repeatable two-host regression and its test-only memory admission override.

Synthetic-checkpoint regressions test cache lifecycle, not language quality;
the HTTP checks above use the full trained GLM checkpoint.
