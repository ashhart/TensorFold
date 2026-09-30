# Native DeepSeek bridge

TensorFold's `deepseek_v4` CUDA family now uses a candidate-only shared library
over ds4 revision `d183482b413ecd2e3b540b290e6497437e9fbb73`.
The MIT sources are vendored unchanged, with per-file SHA256 checks in
`cuda/vendor/ds4/tensorfold-source.json`. The shim is TensorFold-owned.
The live donor checkout, binary, model and services are untouched.

## Build on the Spark

From this checkout, in the candidate Python environment:

```bash
PYTHONPATH=src python -m tensorfold.families.deepseek_v4.cuda.build \
  --backend cuda --jobs 2 --build-dir "$HOME/.cache/tensorfold/deepseek-native-cuda"
```

This compiles a shared library for GB10 `sm_121a`; it does not load a model,
initialize a session or launch a server. CUDA 13's compiler, development headers,
cudart, cuBLAS and libcuda are required. Keep the worker's CPU/memory limits.
Use `--backend cpu` for the standalone ABI/IQ2 smoke library; production refuses
that CPU library. Build output includes `native-build.json` with donor and
library hashes. Native compilation happens outside the source and live donor.

## Runtime boundary

`NativeSession` launches a local child via multiprocessing spawn, loads ABI 1,
checks the donor revision and CUDA backend, then opens one mapped-GGUF session.
Native fatal exits are detected in the parent and reported as `NativeError`.
Cleanup is bounded, including terminate/kill if graceful close fails. Logits
cross the local pipe as float32 bytes; token arrays and token text are freed
according to donor ownership. There is no donor HTTP server or HTTP proxy.

`DeepSeekEngine` owns serial sync/decode, shared TensorFold position-keyed
sampling, evaluated-token callbacks, EOS handling and cancellation. Each
request invalidates its previous checkpoint before sync. It rejects unsupported
ranks, concurrency and drafting before admission or loading. Prompt plus reply
must fit the admitted context; explicit context is never silently reduced.

## Remaining integration

Capacity milestone S02 must implement
`cuda.capacity.admit(model_dir, context, context_explicit) -> dict`, returning
`context_window` (positive int) and the measured plan. The engine calls it before
starting a child or loading weights; a missing implementation fails closed.
Validate actual GGUF geometry, mapped/staging/cache/companion accounting and
memory floor there. Keep admission CPU-only; don't load another full model.

Candidate `config.json` must contain `model_type: deepseek_v4`,
`quantization_config.quant_method: gguf`, `gguf_file`, and `native_library`.
Existing preparation supplies the first three. S05 supplies the built absolute
library path. Family checking also requires the prepared provenance descriptor.
The engine's native session exposes `encode(text, rendered=...)` and
`token_text(id)` for S04's embedded tokenizer and shared HTTP integration.
These milestones remain required before serving the real model.

## Verification performed

Both CPU and CUDA libraries compiled on this Spark. CUDA ABI/revision/backend
loaded successfully without opening a model. The focused affected suite passed
122 checks, including native C fatal-exit isolation, binary logits/token
ownership, representative donor IQ2 arithmetic, serial callbacks/sampling,
option refusal and existing GGUF/quant preparation. No full model or Hunyuan
generation was attempted; actual inference/coexistence qualification is pending.

```bash
TENSORFOLD_TEST_CPU_LIBRARY="$HOME/.cache/tensorfold/deepseek-native-cpu/libtensorfold_ds4.so" \
  python -m pytest -q tests/test_deepseek_v4_native.py tests/test_deepseek_v4_native_engine.py
```

The CPU ABI check explicitly skips without that built library. A skip is not
native qualification. The native isolation fixture needs a C compiler.
