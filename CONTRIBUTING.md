# Contributing

TensorFold is an alpha runtime for low-memory local MoE inference. Contributions
are welcome, but the project has one hard rule: performance claims need fresh
evidence.

## Development Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Run the no-model checks before opening a change:

```bash
PYTHONPATH=src python3 -m pytest \
  tests/test_tensorfold_cli.py \
  tests/test_public_scrub.py \
  tests/test_tensorfold_release_manifest.py \
  tests/test_tensorfold_release_smoke.py \
  tests/test_safetensors_runtime.py \
  tests/test_weight_pager.py \
  -q
PYTHONPATH=src python3 tools/check_public_scrub.py
python3 tools/smoke_tensorfold_release.py
```

## Pull Request Expectations

- Keep public docs honest about what is measured and what is experimental.
- Do not add local paths, private names, model-cache paths, or machine-specific
  defaults to public files.
- Do not include model weights, generated `.safetensors`, pack files, logs, or
  benchmark artifacts in pull requests.
- Add or update tests for CLI behavior, packaging, and memory-accounting logic.
- For speed or memory changes, include the exact command, model, token count,
  RSS peak, resident weight peak, bytes read per token, and exactness status.

## Model Inference Safety

Large model inference can exhaust a small machine quickly. Start new model
runs with tiny token counts and explicit memory guards. Do not run benchmark
claims on a busy host unless you can account for the other resident processes.

## Compatibility

The public command is `tensorfold`. The `smarttensor` package and CLI remain as
compatibility/internal runtime surfaces.
