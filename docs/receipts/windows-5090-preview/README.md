# Windows 5090 preview evidence

This draft ports the previously tested Windows runtime onto current main and publishes the serving program as `tensorfold.exe`.
The new source build needs fresh GPU qualification before release or deployment.
Its CPU build checks do not transfer the historical executable's performance or fidelity results to the new head.

## Remaining performance issue

The historical E36 executable's counterbalanced single-stream run used cache-on, cache-off, cache-off, cache-on cohorts with paired workloads.
The highest-thinking coding repeat requests had a geometric mean cache-on/off decode ratio of **0.997091**, about **0.29% slower**, with five of six pairs below one.
Both order-pair medians were below one, at 0.995450 and 0.998206.
No regression allowance was accepted and this result did not authorize cache promotion.

The preview now excludes initialization-tagged first-round samples from the lone-stream timing estimator.
A synthetic production-loop regression reproduces a 50 ms switch sample replacing a retained 10 ms calibration; the fix retains that calibration while later 5 ms steady samples still update it.
The same tests retain the first 16-row trial, original timing events and complete greedy/sampled token streams.
This fixes an estimator defect; its effect on the historical 0.29% GPU deficit remains unverified until matched 5090 measurements can run.
The six pairs include observations from the same four cohorts, not six independent machines or runs.

`cache-decode-pairs.csv` retains all 72 numeric pair observations across medium/xhigh, prose/chat/code and first/repeat requests.
It excludes prompts, replies, device identifiers and process logs.
Recompute the twelve cells with:

```sh
python3 tools/release/compare_cache_decode.py docs/receipts/windows-5090-preview/cache-decode-pairs.csv
```

`historical-evidence.json` identifies the old executable and the scope of these observations.
The old binary is not redistributed here because its original compiler-to-source binding is incomplete.
The preview must use fresh source-built CUDA images and runtime libraries with their own input/output manifests.

## Reproducing the recorded configuration

The preview is compiled with `-Dqwen-prefill-max-rows=8192`.
That is a prompt chunk capacity, not a context-window setting or a new fidelity qualification.
Set `TF_QWEN_PREFILL_ROWS=4096` for the upstream arithmetic path, with the experimental settings below unset.
The alternate hybrid profile requires the complete tuple below; startup rejects partial combinations before device/model access.
The directory and library values are operator-provided inputs, not bundled weights or machine-specific paths.

| Setting | Recorded value |
| --- | --- |
| `TF_QWEN_NVFP4_MLP_DIR` | Alternate calibrated NVFP4/BF16 layer checkpoint directory |
| `TF_CUBLAS_BRIDGE` | Absolute path to the package's fresh `bin/tf_cublaslt_bridge.dll` |
| `TF_CUBLAS_PREFILL` | `1` |
| `TF_QWEN_NVFP4_PREFILL` | `1` |
| `TF_QWEN_NVFP4_TMA` | `1` |
| `TF_QWEN_NVFP4_SCOPE` | `layers` |
| `TF_CUDA_WARPS8` | `2` |
| `TF_QWEN_EXPAND_KV` | `1` |
| `TF_QWEN_PREFILL_ROWS` | `7680` |
| `TF_DRAFT_BLOCK` | `8` |
| `TF_DRAFT_GRAPH` | `1` |
| `TF_DRAFT_QWEN3_NORM` | `1` |
| `TF_DRAFT_FEEDBACK_WEIGHT` | `0.2` |

Use the dense affine4/group64 primary Qwen3.8-27B checkpoint and its DFlash2 drafter, retaining the package's `lib/tfgrammar.dll` for structured output and required tools.
Serve one stream with `bin/tensorfold.exe serve <primary> --name qwen27 --parallel 1 --drafter <drafter>` and choose `reasoning_effort` per request.
For a paired cache comparison, run the same executable/checkpoints/settings with `--prompt-cache-gib 0`, then an explicit nonzero budget, using identical prompts and alternating run order.
Do not enable the inactive FP8 attention/activation, alternate GDN, target-prefill graph or multi-lane proposal experiments through this profile.

## Validation still required

The historical candidate passed cache correctness, actual eight/sixteen-stream overlap and progress checks, cancellation, and medium/xhigh coding tool workflows.
Those results apply to that executable only.
The rebased preview still needs same-checkpoint token equality, independent fidelity for the hybrid weight path, cold prompt throughput and counterbalanced decode measurements on each advertised GPU/shape.
The active hybrid path keeps the primary affine embedding/head and selected vectors, while alternate NVFP4/BF16 projections supply the layers.
That checkpoint combination is not an arithmetic-equivalence claim against the primary affine checkpoint alone.
CUDA runtime contract execution, active 7,680-row prefill fidelity and the newly produced bridge's GPU behavior remain unrun.
The main inference server is deliberately left untouched while this draft is prepared.
