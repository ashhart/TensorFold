# Build the candidate without disturbing live models

Run the helper with the candidate venv's Python:

```bash
python tools/build_deepseek_v4_cuda.py \
  --gguf /path/to/existing-0731.gguf --model-dir /path/to/candidate/model \
  --companion-reserve-gib MEASURED_ADDITIONAL_GROWTH --jobs 2
```

The helper validates the actual header/schema and embedded vocabulary/pre/EOS,
reuses a matching verified native library or compiles one, builds a Linux
aarch64 wheel in temporary staging, installs with the invoking interpreter,
and prepares candidate sidecars naming the installed native library. Compilation
is for GB10 `sm_121a`; no model is opened. No services or client routes change.
`--build-dir` selects an isolated native cache. `--native-library` requires a
matching donor/source/shim/library receipt and refuses a stale binary.

Builds are offline (`pip --no-index --no-deps --no-build-isolation`), so the venv
must already provide the Python dependencies and setuptools/wheel. Use the
ml-infra launcher to create a candidate environment inheriting qualified base
packages. Wheels include native ELF, shim, pinned sources/manifest and MIT
licenses, with a platform-specific tag. The build never puts ELF files into
the source checkout or rebuilds the live donor.

`--tokenizer-dir` is optional. When supplied, its local tokenizer JSON files
must parse and match embedded vocabulary size/EOS ID; files are copied to the
candidate with recorded hashes. Default `tokenizer_backend=ds4` uses the GGUF's
embedded vocabulary through the native tokenizer. S04 supplies shared chat/tool
serving integration; this helper does not claim that step is done.

## Read-only preflight

Add `--preflight-only` to validate inputs and print JSON. It does not install,
compile, create sidecars, load a shared library, allocate GPU memory, or start
services. It reads only GGUF metadata/table pages through a read-only map.
The checkpoint fingerprint is explicitly SHA256 of the header plus file
size/mtime/device/inode, not a claimed hash of all tensor payloads. Preparation
records this as `descriptor.source_identity.sha256_scope=gguf-header`.

Preflight separates `artifact_valid` from `current_memory_fit=not_evaluated`.
Runtime capacity admission remains mandatory before native model loading.
Reserve is finite nonnegative additional companion growth; existing occupancy
must not be counted again. The runtime's 8 GiB floor is separate. Explicit
262144 context is preserved. No reserve value is guessed for final deployment.

## Verified on this Spark

Real0731 header: 1,328 tensors, 129,280 tokens, clean schema validation. Header
preflight created no model directory. The CUDA library was packaged into a real
`py3-none-linux_aarch64` wheel and installed into an independent candidate venv;
the installed native ABI and TensorFold console script loaded. Helper/preparation
checks: 31 passed. Independent ml-infra orchestration checks: 17 passed.
Real-model inference and Hunyuan generation/coexistence are still pending.

Final deployment remains the separate `make tensorfold-deepseek-3d` command in
ml-infra. Development does not execute that replacement while Hermes uses ds4.
