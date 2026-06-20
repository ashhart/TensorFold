# TensorFold

TensorFold is an Apple Silicon / MLX-first local runtime for running sparse MoE
language models under tight memory budgets.

It keeps model files intact, folds the active expert path into bounded resident
memory, and pages the rest with explicit telemetry.

> The point is not to make small models faster.
> The point is to run models the machine should not normally be able to host.

TensorFold is alpha software. It is already useful for safetensors inspection,
expert packing, release-safe no-model smoke tests, and low-memory runtime
experiments. It should not be described as a solved native-speed quarter-memory
system until a benchmark artifact proves that for a specific model, hardware
profile, and exactness mode.

## Start A Local OpenAI Endpoint

If you already have a supported local model directory, this is the shortest
path from checkout to an OpenAI-compatible endpoint.

```bash
git clone https://github.com/ashhart/TensorFold.git
cd TensorFold
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Later, check for updates from the TensorFold repo with:

```bash
./update.sh
```

The update helper runs a fast-forward-only Git pull from `origin`. It stops if
you have local edits so it does not tangle your changes with upstream updates.

Pick the folder that contains the model's safetensors or MLX shard files:

```bash
# Generic local checkout or downloaded model folder.
MODEL=/path/to/local/model

# Example: Qwen 35B MLX checkpoint folder.
MODEL=/path/to/models/Qwen3.6-35B-A3B-MLX-4bit

# Example: LM Studio model directory shape.
MODEL=$HOME/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit

# Example: oMLX or another model manager. Use the resolved local folder it
# prints, usually a directory ending in the model name.
MODEL=/path/to/models/Qwen3.6-35B-A3B-MLX-4bit

# Ollama note: the default Ollama blob store is not a TensorFold model directory.
# Use an exported or converted safetensors/MLX checkpoint folder.
MODEL=/path/to/exported-or-converted/Qwen3.6-35B-A3B-MLX-4bit
```

`MODEL` is just a temporary shell variable for your current terminal.
It is not a TensorFold config file, it is not saved anywhere, and it is not
required. The examples use it so you can change the path once and reuse the
same command.

If you prefer, paste the path directly:

```bash
tensorfold serve /path/to/models/Qwen3.6-35B-A3B-MLX-4bit \
  --served-name qwen35b-local \
  --resident-budget 2GiB \
  --loader-backend native \
  --host 127.0.0.1 \
  --port 8421
```

Start the server:

```bash
tensorfold serve "$MODEL" \
  --served-name qwen35b-local \
  --resident-budget 2GiB \
  --loader-backend native \
  --host 127.0.0.1 \
  --port 8421
```

Point OpenAI-compatible clients at:

```text
http://127.0.0.1:8421/v1
```

Try it with curl:

```bash
curl http://127.0.0.1:8421/v1/models
```

Send a chat request to `POST /v1/chat/completions`:

```bash
curl http://127.0.0.1:8421/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen35b-local",
    "messages": [
      {"role": "user", "content": "Write one sentence about local inference."}
    ],
    "max_tokens": 32,
    "temperature": 0
  }'
```

If you use a different `--served-name`, put that value in the request's
`model` field.

If you only want to check paths and MLX availability before serving:

```bash
tensorfold doctor --model-dir "$MODEL"
```

When trying a new model, start tiny: use `--max-tokens-default 1` on the server
or send `max_tokens: 1` in the request.

For generic client configuration:

```text
base_url = http://127.0.0.1:8421/v1
api_key  = anything-local
```

Most OpenAI-compatible SDKs require an API key string even for local servers;
TensorFold does not use it in this alpha path.

## Verified Internal Qwen 35B Artifact

A saved internal benchmark artifact verifies the current Qwen hard-quarter
high-water:

| Model/profile | Hardware | Speed | Memory | Scope |
| --- | --- | ---: | --- | --- |
| Qwen3.6-35B-A3B MLX 4-bit, guarded hard-quarter profile | 24 GB Mac mini | 22.122 tok/s | 2.568 GB RSS, 1.060 GB resident weight peak | Exact guarded generation, 56 generated tokens |

The artifact records a generated-token timing window: 56 generated tokens in
2.531 seconds, zero warmup tokens, no replay tape, no fallback blocks, exact
guarded commits, and drop-cache-after-read pack reads. It is not an end-to-end
cold-launch benchmark.

Summary: verified internal artifact, not a cold-launch benchmark, and not yet a public reproducibility claim.

That makes the number real, but still narrow: this release branch does not
bundle the sanitized artifact, the full command transcript, or a fresh
quiet-window rerun.

Publishable performance claims belong in [Benchmarks](docs/benchmarks.md) with
the command, model, prompt, generated token count, RSS, resident peak,
MLX/Metal peak when available, bytes read per token, exactness mode, and
cold/warm timing state.

## Platform Support

TensorFold is Apple Silicon / MLX-first today. The low-memory serving path and
the measured Qwen profile were developed around local MLX checkpoints, unified
memory, and Apple Metal-backed execution.

Linux support is split by surface:

| Surface | Linux status |
| --- | --- |
| `tensorfold doctor`, `selftest`, `inspect`, `manifest`, `pack` | Expected to work as ordinary Python tooling when dependencies are installed. |
| OpenAI-compatible `tensorfold serve` with low-memory MLX execution | Alpha only; not yet a proven TensorFold serving target on Linux. |
| CUDA/ROCm low-memory runtime | Not implemented in this public alpha. |

Linux summary: the release tooling is ordinary Python, but the low-memory
serving path is not yet a Linux product claim. A Linux backend should get its
own benchmark artifacts with the same RSS, resident peak, bytes-read, and
exactness fields.

## Project Snapshot

TensorFold is not only a safetensors inspector, a model quantizer, a prompt
wrapper, or a generic OpenAI-compatible server. The central primitive is
bounded-memory execution for sparse MoE models: keep the currently useful
expert path resident, stream or fault the rest, and report memory honestly.

Runtime path:

```text
model files -> manifest -> expert pack -> bounded runtime -> telemetry
```

The public release currently proves the packaging, no-model runtime path,
expert-pack tooling, public scrub, and installed-wheel smoke path. It does not
claim a universal speed result for every model or platform.

## What TensorFold Is

TensorFold is:

- a public `tensorfold` CLI over the existing SmartTensor runtime internals
- a safetensors and MLX-shard manifest reader that does not load tensor data
- an expert-pack builder for MoE checkpoints
- an OpenAI-compatible local serving surface for supported model families
- a release-hygiene layer with public scrub and installed-wheel smoke tests
- a place to land exact, measured low-memory inference profiles

The runtime is designed for Apple Silicon / MLX development first, but the
package-level tooling is ordinary Python and can be smoke-tested without MLX or
model weights.

## What TensorFold Is Not

TensorFold is not:

- a new model format
- a quantization scheme
- a cloud inference service
- a blanket claim that every frontier model runs at native speed in 2 GB
- a replacement for benchmark artifacts, exactness checks, or memory accounting

The model weights stay as model weights. TensorFold changes how they are
indexed, packed, paged, and accounted for at runtime.

## Proof Surface

The current public branch proves the product shell and release boundary:

| Surface | Evidence |
| --- | --- |
| CLI install path | `tensorfold --version`, `doctor`, `selftest`, `demo create`, `inspect`, `pack` |
| No-model demo | Built-in toy safetensors and toy MoE checkpoint generated by `tensorfold demo create` |
| Expert packing | `tensorfold pack` writes a contiguous `.pack` for the toy MoE fixture |
| Release hygiene | `tools/check_public_scrub.py` scans public files for local paths, private names, model-cache paths, agent handoff names, and email-like strings |
| Installed package smoke | `tools/smoke_tensorfold_release.py` builds a wheel, installs it in a temporary venv, and runs the no-model command path |
| CI gate | `.github/workflows/tensorfold.yml` runs product tests, scrub, and release smoke on macOS |

The current public branch does not include the private research logs, raw model
benchmarks, frontier canary pipeline, or detached experiment worktrees. Those
belong in research branches until they are clean enough to ship.

## Quick Start

Install from a checkout:

```bash
git clone https://github.com/ashhart/TensorFold.git
cd TensorFold
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Run the no-model health checks:

```bash
tensorfold --version
tensorfold doctor
tensorfold selftest
```

Check for repo updates:

```bash
./update.sh
```

Create and inspect the built-in demo:

```bash
mkdir -p .tensorfold-demo
tensorfold demo create .tensorfold-demo --force
tensorfold inspect .tensorfold-demo/toy.safetensors
tensorfold inspect .tensorfold-demo/toy-moe/model-00001-of-00001.safetensors
tensorfold pack .tensorfold-demo/toy-moe --out .tensorfold-demo/toy-packs
```

Pack experts for a supported real MoE checkpoint:

```bash
tensorfold pack /path/to/model --out /path/to/packs
```

Serve a supported model as an OpenAI-compatible endpoint:

```bash
MODEL=/path/to/models/Qwen3.6-35B-A3B-MLX-4bit

tensorfold serve "$MODEL" \
  --served-name qwen35b-local \
  --resident-budget 2GiB \
  --loader-backend native \
  --host 127.0.0.1 \
  --port 8421
```

Point OpenAI-compatible clients at:

```text
http://127.0.0.1:8421/v1
```

## Architecture

```text
TensorFold CLI
  -> safetensors / MLX shard manifest
  -> memory-aware tensor records
  -> optional expert pack files
  -> bounded runtime planner
  -> OpenAI-compatible local server
  -> telemetry for RSS, resident weights, Metal peak, reads, and exactness
```

Public package layout:

```text
src/tensorfold/      public CLI, demo fixtures, product entrypoint
src/smarttensor/     compatibility runtime internals
  adapters/mlx/      MLX loader, session core, and per-model MoE runners
docs/                quickstart, model support, memory accounting, benchmarks
tools/               release scrub and installed-wheel smoke
tests/               product, packaging, safetensors, and pager checks
```

The original `smarttensor` CLI remains available for engine-level compatibility:

```bash
smarttensor inspect /path/to/model/*.safetensors
```

Use `tensorfold` for product workflows and `smarttensor` when working directly
on runtime internals.

## Current Status

| Area | Status |
| --- | --- |
| Public CLI | Ready for alpha use |
| No-model demo | Ready |
| Safetensors inspection | Ready |
| Expert pack creation | Ready for supported MoE tensor layouts |
| OpenAI-compatible serving | Alpha, model-family dependent |
| Qwen / GPT-OSS low-memory profiles | Experimental, benchmark-gated |
| Frontier canaries | Reserved command, not bundled in this public runtime release |
| Native-speed quarter-memory claim | Not claimed by this public branch |

See [Supported Models](docs/supported-models.md) for the model-family matrix.

## Runtime Knobs

TensorFold exposes low-level controls because low-memory inference is not
one-size-fits-all:

- `--resident-budget`: target resident weight budget
- `--loader-backend`: select the available runtime loader backend
- `--pin-policy`: choose the resident pinning strategy
- `--exact-mode`: exactness contract for served generations

See [Memory Accounting](docs/memory-accounting.md) before comparing runs. RSS,
resident weight bytes, MLX/Metal allocations, pending read buffers, and KV cache
are different numbers.

## Benchmark Rules

Do not publish a speed or memory claim unless the artifact states:

- model name and quantization
- hardware
- prompt and generated token count
- tokens per second
- process RSS peak
- resident weight peak
- MLX/Metal peak when available
- bytes read per token
- exactness mode
- whether timing includes cold load or warm decode only

Claims without those fields are exploratory notes, not product benchmarks.

## Release Hygiene

Before publishing or sharing a release branch, run:

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

See [Release Checklist](docs/release-checklist.md) for the full publish gate.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Supported Models](docs/supported-models.md)
- [Memory Accounting](docs/memory-accounting.md)
- [Benchmarks](docs/benchmarks.md)
- [Privacy and Telemetry](docs/privacy-and-telemetry.md)
- [Contributing](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## Community And Security

TensorFold is MIT licensed. Please read [Contributing](CONTRIBUTING.md),
[Security Policy](SECURITY.md), and [Code of Conduct](CODE_OF_CONDUCT.md)
before opening public issues or pull requests.
