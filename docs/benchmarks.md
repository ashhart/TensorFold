# Benchmarks

TensorFold benchmarks are useful only when they state what was timed and how
memory was measured.

## No-Model Checks

```bash
tensorfold doctor
python3 tools/check_public_scrub.py
```

## Manifest And Pack Checks

```bash
tensorfold inspect /path/to/model/*.safetensors
tensorfold pack /path/to/model --out /path/to/packs
```

## Serving Smoke

Start with one or two generated tokens before scaling a profile:

```bash
tensorfold serve /path/to/model \
  --pack-dir /path/to/packs \
  --resident-budget 2GiB \
  --max-tokens-default 2
```

## Reporting Template

Record:

- command
- model
- profile
- prompt
- generated tokens
- tok/s
- process RSS
- resident peak
- MLX/Metal peak
- bytes read per token
- exactness mode
- cold/warm state

Claims without these fields should be treated as exploratory notes, not product
benchmarks.

## Release Canaries

`tensorfold canary qwen-frontier` is source-checkout release tooling. It wraps
the heldout route-atlas pipeline used to evaluate hard-memory Qwen profiles.
Installable releases return a clear source-checkout-required error for this
command until the full canary pipeline is packaged.
