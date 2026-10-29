# Qwen3.8-27B Living Weights on Metal

The Metal engine uses the existing Qwen dense family description, pocket fitter, lesson acceptance checks, save
transaction and affine4 shard writer. It accepts the same 64-neuron SwiGLU pocket as CUDA. The original MLP codes,
scales and biases remain unchanged. Empty layers launch no pocket kernel, and untaught streams skip every pocket.

The trainer is implemented and checked on synthetic weights. Real-checkpoint lesson retention, restart, concurrent,
resumed and drafted/plain checks remain required before calling this backend qualified for learning.

## Forward and backward

The trainer runs the serving runner with fresh per-sequence states, one row at a time, as the dense CUDA trainer does.
It captures the native residuals, projections, prepared attention and DeltaNet rows, answer logits and recurrent checkpoints.
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
ordinary serving speed. The transposed-gradient implementation keeps full FP32 arithmetic; whole-checkpoint
training latency is not established by the synthetic fixture.

## Lesson work and timing

Each learner row uses the same native stages and barriers, in one command buffer instead of one per four layers.
Its single-row round metadata lives on the stack. Serving retains its existing submission interval. Rows still run
in order and wait before the next row uploads shared metadata. This preserves the decode arithmetic; it does not
replace it with prompt matrix products. Prefix reuse is still disabled for this trainer.

Only rows predicting an answer run the vocabulary projection, loss and head gradient. The other rows have zero loss
weight and their head input gradient is explicitly zero. Earlier layers still differentiate through every prompt row,
including attention and recurrent dependencies. Four independent affine-gradient rows share each weight decode, with
each row's original FP32 FMA sequence in increasing output-column order. The optimizer and lesson gate are unchanged.

Set `TF_Q27_LEARN_DIAG=1` before starting the server to log every trainer pass: mode, sequences, full and answer rows,
wall time, forward planning/upload remainder, encoding, commit/wait and GPU spans, backward/loss encoding, commit/wait
and GPU spans, submission/wait counts, resets, and affine weight-decode/FMA counts. During output fitting a target pass is a packed Adam batch;
the server's outer "steps" counter does not count those batches. GPU spans are contained within completion waits;
they must not be added to wall time. Planning/upload is the forward wall remainder after encoding and completion.
The existing `slide pocket: round ...` line still reports read, fit, values, judge and check totals with rows/passes.

Before measuring the full lesson, build and run the scalar replacement check on the GPU under its queue lock:

```sh
swiftc tools/zig/qwen27_learning_affine.swift -o zig-out/bin/qwen27-learning-affine
zig-out/bin/qwen27-learning-affine .
zig-out/bin/qwen27-learning-affine . --reorder-control
```

The last command must fail. It reverses each output-column accumulation in the production source copy. The normal
check compares every output byte to the frozen scalar kernel, including partial row tiles, zero/nonfinite gradients, add/replace,
five repeats, unchanged inputs and guard bytes. These device checks and full-model timing remain required.
