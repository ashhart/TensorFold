# Qwen4 Living Weights: host backward and implementation plan

This is the host mathematical step for #1804. It does not enable learning in the native Qwen4 engine.
`families/qwen4/qsa_backward.zig` computes the continuous selected-attention adjoint for FP32 or FP64 arrays.
Q and upstream gradients are `[rows, heads, width]`; K/V and their adjoints are token-major
`[capacity, kv_heads, width]`. Query head `h` reads KV head `h / (heads / kv_heads)`.
The served QSA scale is 0.0625 at width 256. The host routine accepts an explicit scale for synthetic checks.

For each causal row, dense attention reads keys `0..end`. Sparse attention reads exactly the supplied unique IDs,
including any incomplete causal tail. IDs beyond `end`, duplicate IDs and empty selections are refused.
The selection is a fixed tape. No surrogate derivative is supplied for pooling, score ranking or top-k decisions.
Unselected and future K/V rows are never read and their adjoints stay zero. Shared K/V adjoints accumulate every
query-head and row use. Shape, selection, alias and input-finiteness admission precedes gradient writes.
Numeric overflow clears the output adjoints and returns an error. Inputs and the selection tape remain unchanged.

The proved attention formula is `p = softmax(scale * Q K^T)`, `O = p V` over the fixed selection:

```text
dP[j] = dot(dO, V[j])
dS[j] = p[j] * (dP[j] - sum_i p[i] dP[i])
dQ += scale * dS[j] * K[j]
dK[j] += scale * dS[j] * Q
dV[j] += p[j] * dO
```

The separate output-gate pullback takes saved forward context and sigmoid values. It uses the saved sigmoid for
the context adjoint and the ordinary sigmoid derivative for the logit adjoint. Forward cuts are straight-through
in this local mathematical model. Independent finite differences freeze each cast's offset at the base point;
they do not differentiate a discontinuous rounding function. Both cut and uncut output-gate cases are checked.

```sh
zig build test-qwen4-qsa-backward-host -j1 --summary all
```

The independent reference evaluates a dense binary64 masked loss rather than calling the backward's helpers.
Tiny cases check every Q/K/V and gated-output coordinate at three central step sizes. Directional cases use the
served 24 query heads, 4 KV heads and width 256, with 129 dense keys and 2,051 sparse keys including a three-token
tail. Tests also cover poisoned unused cache rows, shared-head accumulation, malformed tapes, aliases and repeats.
FP64 checks use `1e-8 + 1e-6 * abs(slope)`; FP32 checks use `2e-7 + 3e-5 * abs(slope)`.
These are host derivative-check limits, not new device fidelity or whole-model acceptance thresholds.

The CUDA tensor-core attention path stores BF16 probability operands; Metal's split softmax retains FP32
probabilities. This step does not model those backend-specific probability cuts, split reduction orders, Q/K
RMS/RoPE or their stored intermediates. It is not a gradient receipt for either device profile. The native forward,
its probability stores and all shader sources remain unchanged. Future backward kernels need their own contracts.

## CUDA pocket and trainer

1. Adapt the dense pocket storage and shard writer used by Qwen3.5, with a Qwen4 family descriptor keyed by its
   declared config. Preserve original tensor bytes and separate base columns from appended neurons when loading.
   Use the same serialized family/tensor/quantization description on both backends. Reject unsupported learning
   layouts before allocation; a group-32 Flash Next transpose must never be applied to group-64/128 or other formats.
2. Add hooks at the dense MLP branch before the HC writeback in the shared Flash Next layer graph. Every prompt,
   verify and decode pass must run the same hook. Keep the empty-pocket branch free of allocations, readbacks,
   synchronization and added launches. Supply per-stream untaught masks, including mixed taught/untaught batches.
3. Reuse Flash Next's HC branch/readout backward and GDN backward with explicit Qwen4 geometry and gate mode.
   Qwen4's default SiLU must not inherit Flash Next's sigmoid derivative. Make the backward mode explicit and prove
   both modes with a planted wrong-gate control before reuse, preserving Flash Next's existing mode and arithmetic.
   Add dense SwiGLU/base-weight and pocket-input gradients, the LM head/loss
   path, and optional PLE backward. Until PLE backward is present, learning on PLE configs must refuse explicitly.
4. Record Q/K norm inputs, stored rotary frequencies, V, output gates and the exact selected IDs/counts/row ends
   from the same forward. The backward must use this tape, not reselect. Implement selected softmax backward in
   FP32, accumulating shared K/V uses without lost writes. Integrate the Q/K RMS, rotary, projections and HC chain.
   Keep each sequence's cache ownership explicit; packed training sequences must not read another sequence's keys.
5. Define and test the backward convention for every probability, context and gate cut in the actual CUDA profile.
   Finite differences of the ideal formula above cannot qualify BF16 probability stores. Check local cut offsets
   and the complete computed graph, with controls that drop a cut or transpose a GQA group.

## Metal pocket and trainer

Reuse the shared family descriptor, learned-neuron layout, fitting policy and shard writer. Follow the existing
Nemotron Metal trainer's ownership and residency structure, with Qwen4 operators and its own forward tape. Device
buffers and captured metadata must survive command completion. Keep training scratch separate from served streams;
drain outstanding work before changing pocket buffers and invalidate prompt states after any successful edit.

Port the qualified CUDA backward stages as Metal operators without replacing inference with CPU work. Reuse HC,
GDN and dense projection derivatives only where layout, precision and gate mode match. Register any new operators
and their served shapes, exceptional classes, repeats/aliasing, accumulator stress and launch limits. Pending
compiler/resource records stay unmeasured until actual images are checked. Include pre-M5 and M5 implementations.

## Gates before advertising learning

Use a tiny nonzero Qwen4-shaped model to check whole trainer/serving loss and directional gradients at early rows,
layer boundaries, GDN/QSA and PLE layers. A whole-model finite-difference probe must record whether its selected
IDs change under perturbation. Selection crossings are nondifferentiable witnesses, not permission to widen the
gradient bound or silently freeze a real serving pass. The local fixed-selection check remains a separate receipt.

Test fitting, held-out recall, unrelated-fact retention, exact take-out/delete/rollback, cancellation and save failure
through the existing shared learner/server acceptance path. Round outputs exactly as they will be stored before
judging a fit. Compare the immutable base tensor bytes before/after, and reopen each backend's saved folder on the
other. Assert that untaught streams read the original model even beside taught streams.

Each affected checkpoint and profile is then run on a CUDA GPU and on pre-M5 and M5 Macs. Match binary/runtime,
model/tokenizer revisions, prompts, seeds, quantization and cache settings. Require empty-folder tokens and speed
equal to the base, drafted/plain and concurrent/solo equality, every prompt cut, resume/restart and rejection cleanup.
Measure learning time and peak host/device memory separately. No device, real-checkpoint or performance gate is
claimed by this host step; the portable layout and complete CUDA/Metal trainers remain follow-up work.

Existing owners are the MLP HC seam in `families/flashnext/forward_graph.zig:34`, the CUDA HC/GDN trainer chain in
`families/flashnext/cuda_train.zig:369`, shared GDN backward in `cuda/gdn_train.zig:299`, and Qwen4's indexed-attention
and output-gate adapters in `families/qwen4/gpu_ops.zig:169` and `families/qwen4/metal_gpu.zig:196` under `zig/src/`.
