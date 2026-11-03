# Qwen3.8-27B Living Weights on Metal

The Metal engine uses the existing Qwen dense family description, pocket fitter, lesson acceptance checks, save
transaction and affine4 shard writer. It accepts the same 64-neuron SwiGLU pocket as CUDA. The original MLP codes,
scales and biases remain unchanged. Empty layers launch no pocket kernel, and untaught streams skip every pocket.

The trainer is implemented with host checks and synthetic device fixtures. Fresh device execution, real-checkpoint
retention, restart, concurrent, resumed and drafted/plain checks are required before calling it qualified for learning.

## Forward and backward

The trainer runs the serving runner with fresh per-sequence states, one row at a time, as the dense CUDA trainer does.
It captures the native residuals, projections, prepared attention and DeltaNet rows, answer logits and recurrent checkpoints.
A pack holds at most 1,024 rows across eight sequences, with at most 512 input rows per sequence. Long conversation
keys use a separate, freshly admitted native cache. Target passes without a nudge still retain the residuals that the
shared pocket checks read.

The Metal backward differentiates its own forward. Its DeltaNet rounds the convolution before SiLU, rounds the
sigmoid and activation, uses RMS-normalized Q/K with BF16 scales, rounds softplus and applies an output SiLU gate.
These gate and normalization derivatives follow this backend's forward. BF16 nodes use straight-through
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
  test-qwen27-slide-flags-host test-qwen27-pocket-host test-qwen27-living-host \
  tf-qwen27-pocket tf-qwen27-untaught test-kernel-contract -Dcpu=apple_m1
python3 tools/zig/metal_learning_sources.py --check
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

Real-model cells must use fresh writable copies, the unchanged CUDA lesson gate, original-folder hashes,
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

## Full-checkpoint receipt runner

`tools/zig/qwen27_living_ab.py` runs one owned server at a time on disposable, non-linked checkpoint copies.
Reserve the GPU exclusively for the entire invocation. It binds only loopback, uses a private temporary API key,
retains every request, response, token array, timing, refusal and process exit, and verifies the original files again.
A protocol fake checks the runner on the host; that fake executes no model arithmetic.

Build the base binary from the pinned main and the candidate from this branch with identical settings.
Build the three device checkers above from the candidate. Supply a public prompt panel with at least four prompts,
including prose, code and a long prompt spanning multiple prompt chunks. The JSON panel is an array of strings
and must include a 2049-to-8192-token prompt after tokenization.
The default panel includes a long calibration prompt; all rendered cases must fit 8192 tokens.
Use the same context, parallelism, cache setting and qualified drafter in both arms. The runner requires nonzero
proposed and accepted drafts for each panel and each temperature; a draft flag alone cannot pass.

```sh
python3 tools/zig/qwen27_living_ab.py \
  --base-binary BASE_BINARY --candidate-binary CANDIDATE_BINARY \
  --backend metal --checkpoint ORIGINAL_CHECKPOINT --checkpoint-revision CHECKPOINT_REVISION \
  --hardware 'GPU and RAM class' --runtime 'OS and runtime versions' \
  --pocket-binary zig-out/bin/tf-qwen27-pocket \
  --learning-binary zig-out/bin/tf-qwen27-learning \
  --untaught-binary zig-out/bin/tf-qwen27-untaught \
  --panel PUBLIC_PANEL.json --tokens 256 --reps 3 --output NEW_RECEIPT_DIRECTORY
```

Pass additional shared server arguments as a JSON argv array through `--server-args`; include the context,
parallelism, disabled prompt cache and qualified drafter settings. The default context is 8192 with four slots and
no prompt retention. Choose an adequate admitted context for the supplied panel. No host limits are raised.
The runner alternates main and candidate with `--slide` off, then checks an empty `--slide` folder, teaches a fixed
access-word fact, checks two held-out questions greedy and sampled, and requires unchanged unrelated token arrays.
It checks drafted/plain and concurrent/solo equality before and after teaching, after restart and after deletion.
It deletes through the existing endpoint, rolls the copy back through the existing CLI, and requires original subject
and unrelated token arrays again. All-vocabulary BF16 logit hashes compare up to 512 teacher-forced rows per case.
Full-prompt state and final logit hashes compare 128-row and native-capacity chunks. Both use untaught streams and
load models sequentially. The small device fixtures run only after the servers have closed.

The default speed bound is zero slowdown in every cell's median HTTP wall time over three alternating repetitions.
Raw runtime counters are retained separately. Predeclare any measurement-noise allowance with `--max-slowdown-pct`
before running; a relaxed allowance must not be presented as the exact-speed gate. Run the full panel separately on
M3 Ultra and M5 Ultra. Record peak physical/device memory and full lesson time from the retained logs separately.
These receipts do not establish unmeasured prompt lengths, resumed state, cancellation or peer configurations.

For portability, first run the same command on CUDA, omitting the three Metal device binaries and using CUDA binaries
and hardware/runtime labels. A producer invocation without `--from-other` deliberately exits incomplete after keeping
`portable-model/`; it cannot claim the two-way gate. Transfer that serving copy to the Metal test machine and pass
`--from-other CUDA_PORTABLE_MODEL`. Then run CUDA with `--from-other METAL_PORTABLE_MODEL`. Preserve both receipts.
Imported folders are served without learning or rollback; their historical backup metadata is not a portable undo log.
A complete receipt requires the other backend's folder, held-out recall, actual drafting and all equality checks.
A refusal, missing counters, failed checker or incomplete portability cell cannot become a passing summary.

No real-checkpoint retention, speed or whole-model numerical qualification is implied by these commands.
The shared lesson acceptance checks and directional error limit remain unchanged.

## Local backward checks

`zig build test-qwen27-linear-host test-qwen27-learning-reference-host` runs without a device.
The loader check passes 2-, 3-, 4-, 5-, 6- and 8-bit projections through the actual linear builder,
including a rank slice, and compares every copied byte with the original views.

`zig build tf-qwen27-learning-local` builds a bounded, checkpoint-free Metal panel.
Run `zig-out/bin/tf-qwen27-learning-local` under the machine's GPU lock.
The same panel is called by `tf-kernel-contract --only "qwen27 learning local" --no-limits`.
Its four source mutations contribute to the contract's control tally and a missed control fails the run.
Compile, launch, ownership or repeat errors fail the run; they never count as a caught numerical fault.

Independent binary64 references cover pocket input adjoints with saved/open neuron scaling, raw affine4 input
adjoints with add mode and a ragged four-row batch, weighted partial-RoPE Q/K RMS adjoints, and all split outputs
of a five-row GDN recurrence across its four-row checkpoint. Every checked output repeats byte for byte five times,
all guards survive and inputs stay unchanged. The affine fixture's dyadic expression is exact; the other three
retain the existing local numerical panel's 2e-6 absolute allowance. These local fixture limits do not replace or
relax the whole-model directional rule or any declared compiler/resource limit.

The host also checks independent smooth central slopes for the pocket, weighted RMS/RoPE and recurrence references.
Stored BF16 inputs are base nodes, not a differentiable quantizer. The recurrence's saved checkpoint offsets are
frozen at the base point for its local straight-through check. These are local mathematical checks, not proof of
real-checkpoint gradients through every served rounding point. The panel supplements the contract; it does not
claim all shapes, exceptional classes, accumulator witnesses or all 22 backward kernels are qualified.

Before advertising learning, retain the complete whole-model directional panel, full contract and resource receipts,
pre-M5/M5 slide-off token and speed comparisons, the mixed-width release checkpoints, keep-rate and exact removal,
and the CUDA/Metal saved-folder round trip. Every refusal and every failed device cell remains in the receipts.
