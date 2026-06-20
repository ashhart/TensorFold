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
  --resident-budget 2GiB \
  --loader-backend native \
  --max-tokens-default 2
```

## Verified Internal Qwen 35B Artifact

The current internal high-water artifact for `Qwen3.6-35B-A3B` on a 24 GB Mac
mini records:

| Metric | Value |
| --- | --- |
| Speed | 22.122 tok/s |
| Generated tokens | 56 |
| Process RSS after run | 2.568 GB |
| Resident weight peak | 1.060 GB |
| Timing scope | Generated-token timing window, not end-to-end cold launch |
| Warmup tokens | 0 |
| Exactness | Exact guarded commits, no replay tape, no fallback blocks |
| Pack read state | drop-cache-after-read reads |

This internal artifact is not bundled in the public alpha, so cite it as a
verified internal result rather than a public reproducibility claim. A public
claim needs the sanitized artifact, exact command, prompt, environment, and a
fresh quiet-machine reproduction.

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

`tensorfold canary qwen-frontier` is reserved for future frontier benchmark
tooling and fails closed in this public runtime release. Do not use it for
published claims until the canary implementation is packaged and covered by the
release smoke.
