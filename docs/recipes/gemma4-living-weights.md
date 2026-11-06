# Gemma 4 Living Weights

The experimental native CUDA and Metal implementations for #1803 share a GeGLU pocket, learner and pending-save
transaction. Public `--slide` remains refused by name until the raw whole-graph gradient panel and device gates
pass. Compiling these paths does not qualify learning. The tiny raw BF16 panel currently fails four of six cells
under the unchanged rule: median of the four coarsest central slopes, relative error at most 0.15.

```sh
zig build test-gemma4-attention-backward-host test-gemma4-learning-host -j1 --summary all
zig build diagnose-gemma4-raw-gradient-host -j1 --summary all
```

The second command intentionally retains the current numerical failures. Local mathematical checks and the
recording trainer are separate from that raw-quantized diagnostic and from real-checkpoint learning.

## Native implementation

`learning_profile.zig` admits the dense 12B and 31B geometry with MLX affine4/group64 projections and embeddings.
Other serving widths keep their original serving path. Per-layer-input and shared-KV-owner configurations retain
their existing named refusals. Training is bounded to one fresh sequence of at most 256 prediction rows. It uses
a separate native cache and scratch, retaining every intermediate before the next 16-row chunk overwrites it.
The training sequence fits entirely within the 1,024-token sliding window.

A pocket adds up to 64 GeGLU neurons at each normalized pre-feedforward input. It adds their down projection before
post-feedforward RMS and the layer scalar. CUDA prompt, verification and decode, and both Metal forward paths use
that seam. Empty and wholly untaught pockets allocate and launch nothing. Untaught streams skip saved prefix reuse
and use the exact original packed base rows and columns. Without a learned folder or `--slide`, no owner is created.

The trainer differentiates the capped tied head, final RMS, layer scalar, both sandwich-normalized residual branches,
the base GeGLU and existing pocket input, Q/K rotary and weighted RMS, and V's unweighted RMS. Global layers have
one raw K projection used by distinct prepared K and V nodes. Their two input adjoints are added before applying
the K projection's adjoint. Attention uses scale one, grouped KV heads and fixed causal logical positions.

CUDA's backward records its high/low BF16 exponential operand convention and interleaved 32-key online-max owners.
Metal uses the float probability profile. M5 operand behavior still needs a device receipt. The original host
`attention_backward.zig` remains an independent prepared-attention reference, including sliding/ring/image masks;
it is never a production fallback. Local scalar and composite chain checks use binary64 objectives and that
reference, separately from native resource and whole-model qualification.

`learning_buffers.zig` preflights the complete device tape, persistent adjoints, largest original-row projection
staging area, host rows, dedicated forward state and peak pocket publication. Staging reads original checkpoint
rows rather than either backend's packed inference tiles. Training operators stay on the native device.
CUDA uses the serving context and stream. Metal uses the serving device and queue with an ordered compute/blit
session. A failed completion retains the complete trainer and native parent, including the driver and sibling
allocations, until a later open boundary confirms completion.

## Shared fitting and persistence

Both backends use `living/families.zig`, the common learner, writer rounding and `pocket_save.zig`. The family
activation is tanh GeGLU. Learned rows use the same gate/up/down order, affine4/group64 format, tensor namespace
and configuration marker on both backends. Base intermediate width remains separate from the appended 64 rows.
The loader extracts those rows and compacts original down-projection columns before native repacking.

The shared gate evaluates the values the writer will store and requires its existing recall, retention and safety
checks. Acceptance limits are unchanged. A failed save remains pending; ordinary open refuses that transaction.
Retry uses the same live values. Host tests exercise the actual writer, exact base-byte preservation, pending
failure/retry, reopening and an invalid-token lesson through the actual shared learner.

## Qualification still required

The raw BF16 diagnostic has zero slopes at some early subject rows and unstable slopes at answer rows. Its failures
are not waived by the passing local derivatives. Native numerical tests, compiler records and all served GPU
geometries are separate gates. Unmeasured CUDA launch records remain pending.

Before enabling `--slide`, qualify both 12B and 31B on CUDA, pre-M5 Metal and M5 Metal. Retain original-versus-candidate
token arrays and timings with slide off, empty slide on, taught and untaught streams, drafted/plain execution,
cache cuts, rejection, restart, lesson removal and CUDA-to-Metal saves in both directions. Measure kept lessons,
supplied held-out recall, unrelated replies, lesson time and peak memory separately. An unchanged original folder
and exact take-out replies are necessary safety receipts; neither proves that a lesson was learned.
