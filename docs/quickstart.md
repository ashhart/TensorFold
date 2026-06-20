# Quickstart

This guide starts with the fastest real-model path: point TensorFold at a local
model directory and expose an OpenAI-compatible endpoint. The no-model checks
below let you verify the package without loading a large checkpoint.

## 1. Install

```bash
git clone https://github.com/ashhart/TensorFold.git
cd TensorFold
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

To check for TensorFold updates later:

```bash
./update.sh
```

The update helper pulls from `origin` with `git pull --ff-only` and stops if
you have local edits.

## 2. Start A Local OpenAI Endpoint

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

In another terminal:

```bash
curl http://127.0.0.1:8421/v1/models
```

Send a chat completion request:

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

If you are using a different supported checkpoint, replace `MODEL` with the
folder that contains that model's safetensors or MLX shard files.

Use this base URL for OpenAI-compatible clients:

```text
http://127.0.0.1:8421/v1
```

Generic client config:

```text
base_url = http://127.0.0.1:8421/v1
api_key  = anything-local
```

When trying an unprofiled model, start with `max_tokens: 1` or add
`--max-tokens-default 1` to the server command.

## 3. Check The Runtime Without A Model

```bash
tensorfold doctor
tensorfold selftest
```

The doctor command prints Python, TensorFold, SmartTensor, and MLX availability.
The selftest command creates the built-in demo, inspects it, and packs the toy
MoE fixture. Neither command loads a real model.

## 4. Create A No-Model Demo

```bash
mkdir -p .tensorfold-demo
tensorfold demo create .tensorfold-demo --force
tensorfold inspect .tensorfold-demo/toy.safetensors
```

## 5. Pack A Toy MoE Model

```bash
tensorfold inspect .tensorfold-demo/toy-moe/model-00001-of-00001.safetensors
tensorfold pack .tensorfold-demo/toy-moe --out .tensorfold-demo/toy-packs
```

This proves the public TensorFold command path can find MoE-shaped expert
tensors and write a contiguous expert pack without loading a large model.

## 6. Pack A Supported MoE Model

```bash
tensorfold pack /path/to/model --out /path/to/packs
```

Expert packs arrange sparse MoE expert rows into contiguous files so the runtime
can issue fewer, larger reads.

## 7. Verified Qwen 35B Reference Point

On a 24 GB Mac mini, a saved internal artifact for the alpha hard-quarter
guarded profile for `Qwen3.6-35B-A3B` reached `22.122 tok/s` at `2.568 GB RSS`
with a `1.060 GB` resident weight peak.

The artifact covers a short generated-token timing window: 56 generated tokens,
zero warmup tokens, exact guarded commits, no replay tape, and no fallback
blocks. It is not an end-to-end cold-launch benchmark and it is not yet a public
reproducibility claim.

## 8. Platform Support

TensorFold is Apple Silicon / MLX-first for low-memory serving today. The
ordinary Python tooling (`doctor`, `selftest`, `inspect`, `manifest`, `pack`) is
intended to run on Linux too, but the OpenAI-compatible low-memory serving path
is not yet a proven Linux runtime target.

## 9. Benchmark Carefully

Use short runs first and record:

- tokens per second
- process RSS
- resident weight peak
- MLX/Metal peak
- bytes read per token
- exactness mode

Never compare a cold-load timed run against a warm decode-only baseline.
