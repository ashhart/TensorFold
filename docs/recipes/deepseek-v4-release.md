# Single-Spark DeepSeek release (2026-09-30)

The reduced serial release is built, installed and running on the existing
0731 mixed IQ2_XXS/Q2_K/Q8_0 GGUF. TensorFold owns HTTP, chat, sampling and
request lifecycle; the pinned ds4 native library supplies tokenization and
CUDA evaluation. No model conversion, replacement download or donor rebuild.

## Launch

```bash
cd ~/Documents/Projects/ml-infra
make tensorfold-deepseek-3d
```

The local default binds `0.0.0.0:8000` for Tailscale, model
`deepseek-v4-flash`, context 262144, tp=1, parallel=1, no drafts.
The native donor pin is d183482b413ecd2e3b540b290e6497437e9fbb73.
Runtime source commit d164576; ml-infra integration commit 9bd307e.
The isolated installed wheel and receipts are under
`~/.local/share/tensorfold-deepseek-3d/`; source/header/native hashes are in
[build evidence](../evidence/deepseek-v4-build-ready.json).

## Hardware evidence

[Real E2E results](../evidence/deepseek-v4-e2e.json) passed real chat, code,
SSE completion, required tools plus followup, thinking, chat during actual
30-step/octree-256 Hunyuan generation (15,375,500-byte GLB), and CUDA moderation.
The API confirms context_length=262144. The native graph was allocated at that
context; startup logged 6,194.3 MiB actual graph allocation against a conservative
11,105.2 MiB allocator estimate. Minimum available memory during the run was
100,297,027,584 bytes (93.41 GiB). Swap did not grow during the run; preexisting
swap occupancy was 9,825,873,920 bytes. Mapped weights remain reclaimable;
this available-memory measurement does not mean all 81 GiB of model pages are
simultaneously resident or that larger workloads cannot cause page faults.

Companion generation was measured independently first. Its 2.91 GiB peak is
rounded up to the local 3 GiB reserve. Runtime admission budgets the full
80.76 GiB GGUF, native graph geometry, 1 GiB runtime allowance, companion growth
and an 8 GiB floor before native loading. Current companion occupancy is already
in MemAvailable and is not counted twice. New failure-sentinel capacity checks
passed; prior native/engine/HTTP focused run: 11 passed, 1 CPU-library skip;
that skipped IQ2 primitive was then run with the built library and passed.
ml-infra ownership/failure tests: 17 passed.

The E2E harness is `scripts/qualify_tensorfold_deepseek_3d.py` in ml-infra.
It records results and GLB under `~/.cache/tensorfold/qualification/` and stops
only the candidate if available memory falls below 8 GiB.

## Limits and operator state

A full 262144-token prompt, exhaustive numerical qualification, long-duration
soak and optional concurrency/drafting/DSpark features are not claimed.
The reduced release requires allocated context plus representative real behavior,
not the old expanded G10 matrix. Unmodified CUDA arithmetic is reused from the
pinned donor; this run is inference/coexistence evidence, not an independent
GPU numerical oracle comparison. Unsupported structured outputs remain refused.

The user authorized stopping the original ds4 service for final testing and
requested no restart. `deepseek-3d-coder-262144.service` is inactive and rollback
logic was removed. Hunyuan and CUDA moderation remain healthy. Hermes workers
were paused and remaining cards completed manually from this evidence; no worker
was restarted. The original keep-ds4-live development instructions are superseded
for this operator-owned qualification only.

Changes are committed locally. No GitHub authentication, push or PR publication
was performed; [PR draft](deepseek-v4-pr.md) is ready for the user's fork.
