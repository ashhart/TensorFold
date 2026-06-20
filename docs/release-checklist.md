# Release Checklist

Run this checklist before publishing a TensorFold branch, wheel, or source
archive.

## 1. Product Tests

```bash
PYTHONPATH=src python3 -m pytest \
  tests/test_tensorfold_cli.py \
  tests/test_public_scrub.py \
  tests/test_tensorfold_release_manifest.py \
  tests/test_tensorfold_release_smoke.py \
  tests/test_safetensors_runtime.py \
  tests/test_weight_pager.py \
  -q
```

## 2. Legal And Community Files

Confirm these files exist before publishing:

- `LICENSE`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `CODE_OF_CONDUCT.md`

The package metadata says MIT, so the repository must include the license text.

## 3. Public Scrub

```bash
PYTHONPATH=src python3 tools/check_public_scrub.py
```

This checks the public package surface for absolute local paths, private names,
local model-cache paths, agent handoff names, and email-like strings.

## 4. Installed Wheel Smoke

```bash
python3 tools/smoke_tensorfold_release.py
```

The smoke builds a wheel, installs it into a temporary virtual environment,
then runs:

- `tensorfold --version`
- `tensorfold doctor`
- `tensorfold demo create`
- `tensorfold inspect`
- `tensorfold pack`
- `tensorfold selftest`

No large model is loaded.

## 5. Benchmark Claims

Do not publish a speed or memory claim unless the artifact states:

- model name and quantization
- tokens per second
- process RSS peak
- resident weight peak
- MLX/Metal peak when available
- bytes read per token
- exactness mode
- whether timing includes cold load or warm decode only

TensorFold is the runtime surface for low-memory MoE serving. A profile is only
native-speed or quarter-memory when a fresh benchmark artifact proves that for
that specific model and hardware.
