# DeepSeek-V4.1-Flash (MLX)

TensorFold's `deepseek_v41` family runs the text model on Apple Silicon with MLX 0.32.2 or 0.32.3. It implements Single-Pass hyper-connections, CSA2 window and compressed-pool attention, the lightning indexer, Engram memory, routed and shared experts, and the native three-stage DSpark draft head. The family uses TensorFold's lane engine; exact multi-row verification, draft rollback and concurrent-stream equality are covered by its tiny-checkpoint suite.

The family loader accepts affine 3-bit or 4-bit group-64 weights with BF16 exceptions. The included converter turns DeepSeek's mixed FP4/FP8 source shards into MLX affine 3-bit group-64 weights, keeping the source untouched and processing one shard at a time. This path is text-only; it does not load the vision tower or aligner. Install the optional conversion dependencies:

```bash
python -m pip install "tensorfold[convert]"
python -m tensorfold.families.deepseek_v41.convert \
  --src /path/to/DeepSeek-V4.1-Flash-source \
  --out /path/to/DeepSeek-V4.1-Flash-MLX-Q3
```

The output receives a runnable safetensors index and tokenizer/config files only after every source shard converts successfully. A partial `--shards` probe deliberately does not create a runnable checkpoint.

```bash
tensorfold serve /path/to/converted-checkpoint --name deepseek-v41
```

Use `--mtp-drafts 0` to disable drafting. By default, the runtime uses the checkpoint's DSpark stages when present, up to the stage count supported by the lane engine. A checkpoint without those stages runs serially.

## Exactness and testing

The family preserves the same-runtime serial contract: each decode row's routing, expert accumulation, attention state, and cache update match its single-row execution. The DSpark tests cover greedy and sampled emissions, accepted and rejected drafts, stream sharing, and cache rollback.

Run the focused suite on a Mac with MLX installed:

```bash
python -m pytest -q tests/test_deepseek_v41_family.py
```

With `tensorfold[convert]` installed, test the CPU conversion path too:

```bash
python -m pytest -q tests/test_deepseek_v41_converter.py
```

## Memory and measurements

The full checkpoint is a several-hundred-gigabyte workload; use a Mac whose available unified memory can keep its converted weights resident. The family wires the Metal working set when MLX reports a recommended limit. No full-checkpoint throughput or quality figures are published on this page; use the standard TensorFold benchmark receipt before comparing a release build.
