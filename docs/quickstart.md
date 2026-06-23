# Quickstart

This guide uses no-model commands first so you can verify the installation
without loading a large checkpoint.

## 1. Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

## 2. Check The Runtime

```bash
tensorfold doctor
```

The doctor command prints Python, TensorFold, and MLX availability.
It does not load model weights.

## 3. Create A No-Model Demo

```bash
mkdir -p .tensorfold-demo
tensorfold demo create .tensorfold-demo --force
tensorfold inspect .tensorfold-demo/toy.safetensors
```

## 4. Pack A Toy MoE Model

```bash
tensorfold inspect .tensorfold-demo/toy-moe/model-00001-of-00001.safetensors
tensorfold pack .tensorfold-demo/toy-moe --out .tensorfold-demo/toy-packs
```

This proves the public TensorFold command path can find MoE-shaped expert
tensors and write a contiguous expert pack without loading a large model.

## 5. Pack A Supported MoE Model

```bash
tensorfold pack /path/to/model --out /path/to/packs
```

Expert packs arrange sparse MoE expert rows into contiguous files so the runtime
can issue fewer, larger reads.

## 6. Serve

```bash
tensorfold serve /path/to/model \
  --pack-dir /path/to/packs \
  --resident-budget 2GiB \
  --pack-read-workers 8 \
  --close-shard-handles \
  --mlx-cache-limit 0 \
  --host 127.0.0.1 \
  --port 8421
```

Then use any OpenAI-compatible client with:

```text
http://127.0.0.1:8421/v1
```

## 7. Benchmark Carefully

Use short runs first and record:

- tokens per second
- process RSS
- resident weight peak
- MLX/Metal peak
- bytes read per token
- exactness mode

Never compare a cold-load timed run against a warm decode-only baseline.
