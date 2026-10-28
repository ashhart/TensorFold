# Qwen3.8-27B Living Weights on Metal

The Metal engine uses the existing Qwen dense family description, pocket fitter, lesson acceptance checks, save
transaction and affine4 shard writer. It accepts the same 64-neuron SwiGLU pocket as CUDA. The original MLP codes,
scales and biases remain unchanged. Empty layers launch no pocket kernel, and untaught streams skip every pocket.

The trainer is implemented and checked on synthetic weights. Real-checkpoint lesson retention, restart, concurrent,
resumed and drafted/plain checks remain required before calling this backend qualified for learning.

## Forward and backward

The trainer runs the serving runner with fresh per-sequence states, one row at a time, as the dense CUDA trainer does.
It captures the native residuals, projections, prepared attention and DeltaNet rows, logits and recurrent checkpoints.
A pack holds at most 1,024 rows across eight sequences, with at most 512 input rows per sequence. Long conversation
keys use a separate, freshly admitted native cache. Target passes without a nudge still retain the residuals that the
shared pocket checks read.

The Metal backward differentiates its own forward. Its DeltaNet rounds the convolution before SiLU, rounds the
sigmoid and activation, uses RMS-normalized Q/K with BF16 scales, rounds softplus and applies an output SiLU gate.
The CUDA trainer's sigmoid output gate and normalization cannot be copied unchanged. BF16 nodes use straight-through
input gradients. Reductions and transposed affine products accumulate in FP32; no FP32 operand is fed to an M5 MMA.
Attention differentiates its gate from the probability product, rather than reconstructing context from a rounded
gated BF16 row. The independent reference uses binary64, including central differences of the stated smooth nodes.
This does not claim a derivative of the discontinuous quantizer.

The server reserves the trainer's future buffers before fitting streams, prompt frames, cache and the optional
Neural Engine share. An idle learner replaces that reservation with its actual allocations. A weight change clears
the prompt store and the native standing/rewind metadata, pending round and draft trees. Learning uses the lane host's
exclusive phase; it does not introduce a second serving scheduler.

## Checks

Build-only and host checks, without a GPU:

```sh
zig build tf-qwen27-learning check-qwen27-speed-host test-qwen27-learning-host \
  test-qwen27-slide-flags-host test-qwen27-pocket-host test-kernel-contract -Dcpu=apple_m1
```

Run GPU checks only when the project's GPU queue permits them. The standalone test creates a small synthetic folder:

```sh
zig-out/bin/tf-qwen27-learning zig-out/qwen27-learning-fixture
swiftc tools/zig/qwen27_learning_numeric.swift -o zig-out/bin/qwen27-learning-numeric
zig-out/bin/qwen27-learning-numeric .
zig-out/bin/qwen27-learning-numeric . --sigmoid-control
```

The final command must fail the gated-RMS gradient check. It substitutes the CUDA sigmoid derivative in the test's
source copy. The normal checks retain five identical runs, input/guard checks, FP64 references and their measured
errors. The native fixture checks serving/trainer logits, solo/packed loss, repeated gradients, no-nudge residuals,
open-pocket scaling and untaught streams.

The native fixture is also reached through the Metal contract runner's `qwen27 learning` selection. Every learner
launch has a source-derived thread ceiling and a registered limits source. Bounded fixtures are not the complete
contract receipt; run the full Metal contract, its controls and the supported-chip limits before qualification.

The lead's real-model cells must use fresh writable copies, the unchanged CUDA lesson gate, original-folder hashes,
held-out wordings and unrelated prompts. Check taught-folder restart with and without `--slide`, drafted/plain,
concurrent/solo, resumed/fresh and untaught output equality. Record trainer time and peak memory separately from
ordinary serving speed. The initial transposed-gradient implementation favors full FP32 arithmetic; whole-checkpoint
training latency is not established by the synthetic fixture.
