# Level 2 prototype validation

Implementation targets TensorFold 0.6.3 (`9356df5`) with the scope in the
[implementation plan](../plans/model-sleep-level2.md). This is local prototype
evidence, not full-model or release qualification.
The tested production code is committed through `92f4f87` on `feat/model-sleep-level2`.

## Actual CUDA checks

Environment: Linux, Python 3.12, NVIDIA GeForce RTX 4090 Laptop GPU (16 GiB, SM 8.9),
PyTorch 2.13.0+cu130, CUDA runtime 13.0, Triton 3.7.1. Extensions compiled locally
using CUDA toolkit 13.4. All runs used the working source tree.

`tools/sleep_fixture.py` generates a deterministic random affine 4-bit/group-64
checkpoint: two layers (one GDN and one attention), hidden width 128, vocabulary
256. Qualification used a 1024-token context, 32-token prompts, 16-token replies
and three sleep/wake cycles per run.

| Run | Loaded allocated bytes | Loaded reserved bytes | Asleep allocated/reserved bytes, every cycle | Token equality |
| --- | ---: | ---: | ---: | --- |
| One stream, greedy | 239,616 | 2,097,152 | 0 / 0 | Before = after every wake |
| Two streams, greedy | 429,056 | 2,097,152 | 0 / 0 | Concurrent = solo = after every wake |
| Two streams, sampled | 429,056 | 2,097,152 | 0 / 0 | Concurrent = solo = after every wake |
| Two streams + DFlash2, sampled | 128,013,824 | 150,994,944 | 0 / 0 | Drafted concurrent = serial solo = after every wake |

All old engine weak references expired; app, tokenizer and template identities
remained unchanged. The sampled run uses seed 1234 plus the stream offset,
temperature 1, top-k 20, top-p 0.95. It produces varying token IDs, recorded alongside
checkpoint file hashes, timings and memory readings in the
[raw sampled receipt](model-sleep-local.json).

The drafted run uses 64 random target layers of width 2048 and a one-layer DFlash2
fixture with the real five-tap interface. It generates and verifies draft candidates,
including an accepted candidate, and leaves real prefix/recurrent/drafter caches
before sleep. All three cycles release those allocations to zero. Its
[raw receipt](model-sleep-drafted-local.json) records the distinct checkpoint hashes.

Commands (in a CUDA-enabled environment):

```bash
PYTHONPATH=src python tools/qualify_sleep.py --synthetic --output sleep-serial.json
PYTHONPATH=src python tools/qualify_sleep.py --synthetic --parallel 2 --output sleep-concurrent.json
PYTHONPATH=src python tools/qualify_sleep.py --synthetic --parallel 2 --seed 1234 --output sleep-sampled.json
PYTHONPATH=src python tools/qualify_sleep.py --synthetic --synthetic-draft \
  --parallel 2 --seed 1234 --output sleep-drafted.json
```

Process RSS remains around 1.1–1.6 GB because the Python process, CUDA context,
compiler/runtime modules and frontend stay alive. Zero PyTorch allocator bytes
does not mean the process has no GPU driver allocations. These small checkpoint
timings are not forecasts for a 27B model.

## Host checks and review

Tests cover full-request admission before preparation, stream draining, nested
Responses/Anthropic translation, authorization and HTTP framing, conflicting calls,
timeout recovery, failed reload cleanup/retry, fatal shutdown, checkpoint identity,
context/math/stream preservation, weak-reference release, loader staging cleanup,
and stored-response continuation after replacing the runtime.

Red failures observed before implementation: missing lifecycle/adapter modules;
HTTP controls returned 404 instead of 401 and sleeping POSTs reached body handling;
three loader readers failed to close after injected read failures. Focused tests
passed after implementing those behaviors. The integration review checked CLI
runtime owners, app vision ownership, scheduler joining, callback exception frames,
global device caches and GET telemetry references.

The combined integration run passed **290 tests, four skipped**:

```bash
python -m pytest -q -p no:cacheprovider tests/test_model_lifecycle.py tests/test_cuda_sleep.py \
  tests/test_sleep_http.py tests/test_cuda_cli.py tests/control/test_telemetry.py \
  tests/test_metrics.py tests/test_responses_api.py tests/test_auxiliary_http_framing.py
```

An additional run passed **119 tests, ten skipped**:

```bash
python -m pytest -q -p no:cacheprovider tests/test_cuda_server_health.py \
  tests/test_http_request_bodies.py tests/test_cuda_precision.py tests/test_cuda_affine.py
```

The broad host-suite attempt was stopped in `test_prompt_attention.py` after 11m42s:
**2,357 passed, 763 skipped, 69 failed, six errors, six subtests passed**. This Linux
environment has MLX importable but no Metal backend, no PyTorch in the host-test
environment, and no `prompt_toolkit`. The TUI collection module was excluded after
its missing dependency stopped collection. The ten failures not directly reporting
the absent Metal backend were rerun from an isolated, unmodified `9356df5` checkout:
all ten reproduce there (seven other tests passed in that baseline run). The broad
suite is not claimed green or complete; the focused affected suites above are green.

The broader HTTP regression run passed 265 tests with one existing Linux
failure: the health-memory test expects a macOS process-footprint field. Loading the
baseline HTTP implementation reproduces that failure. The broader adapter regression
run passed 167 tests with 16 dependency skips; its final CLI compatibility correction
passed 68 tests with four skips. These runs overlap and should not be summed.

## Full-checkpoint follow-up

The subsequent [full-checkpoint validation](model-sleep-cuda-validation.md) covers
NVFP4 plus DFlash2, single-stream and concurrent sleep/wake, and unified-memory
reclamation, plus HTTP stream draining and stored conversation continuation. It also
records decode and 2048/8192-token cold-prefill comparisons against unmodified 0.6.3
before and after wake. The results above remain the original synthetic and host-test
evidence.

## Additional qualification

- Longer prompts, aggregate concurrent throughput, first-request latency immediately
  after wake and cold-storage reload latency.
- EXL3, full-size affine checkpoints and image-input hardware qualification. Their
  reload settings and host cleanup paths are covered; their full runtime reclamation
  is not established here.
- Metal, other families, Level 1, tensor parallelism and cache snapshots belong to
  later milestones; this adapter refuses unsupported opt-ins.
