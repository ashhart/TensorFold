# Gemma 4 Living Weights: host backward and implementation plan

This is the host mathematical first step for #1803. It enables no native learner, pocket, admission flag or save
path. Existing serving sources and all CUDA/Metal shaders remain byte-identical. The new module is reachable only
from a host test target, included in `zig build test` on every host.

`families/gemma4/attention_backward.zig` differentiates prepared attention with scale **1**:

```text
P = softmax(Q K^T) over the captured window
O = P V
dP[j] = dot(dO, V[j])
dS[j] = P[j] * (dP[j] - sum_i P[i] dP[i])
dQ += dS[j] K[j]; dK[j] += dS[j] Q; dV[j] += P[j] dO
```

Gemma's logit softcap of 30 belongs to the LM head. Attention scores have no such softcap or inverse-root-width
factor. The current CUDA forward sets `scale = 1` and `p32 = true` in `gemma4/cuda_forward.zig`; the Metal prompt
kernel likewise takes the prepared dot product. The source of those conventions remains unchanged.

Q, dO and dQ are token-major `[rows, heads, width]`; K/V and their separate adjoints are canonical token-major
`[capacity, kv_heads, width]`. Head h reads KV head `h / (heads / kv_heads)`. All uses accumulate, including prefix
keys that have no answer loss themselves. The native head-major/paged caches need explicit adapters in a trainer.

Each query supplies its captured half-open logical range `[begin, end)`. It can describe global causal attention,
a sliding window or the upper bound that exposes all rows of an image block. These integer masks are fixed tape.
A linear snapshot starts at token 0. A ring snapshot owns `[first, first + capacity)`, with logical token t stored at
`t % capacity`. All query ranges must fit that one generation. A trainer retains earlier generations or a full
sequence tape before eviction; the host API does not reconstruct overwritten keys.

The function admits shapes, byte counts, windows, output/input/metadata aliases and selected input finiteness before
writing. It overwrites adjoints; unused cache rows stay zero and masked nonfinite inputs are never read.
Admission and allocation failures leave outputs unchanged. Nonfinite arithmetic during differentiation clears
all adjoints and returns an error.
Input arrays and mask metadata stay unchanged. K and V input aliases are allowed; their returned gradients remain
separate for the caller's chain rule.

```sh
zig build test-gemma4-attention-backward-host -j1 --summary all
```

The independent dense-mask binary64 objective has its own physical-to-logical mapping and forward loss.
Tiny linear and wrapped cases check every Q/K/V coordinate at three central step sizes in both f64 and f32.
Directional cases use the 31B's 32/16 heads at width 256 and 32/4 at width 512, and the 12B's 16/8 and 16/1 forms.
They cross 1,024-key windows, a 1,025-key extended upper bound, ring wrap and incomplete global tails.
Single-key, shared-head accumulation, synthetic input ties, repeat, ownership and allocation failures are separate
checks. FP64 slopes use `1e-8 + 1e-6 * abs(slope)`; FP32 uses `2e-7 + 3e-5 * abs(slope)`.
These are local host derivative-check limits. Whole-model acceptance and kernel/resource limits are unchanged.

This continuum model starts after Q/K/V preparation. Central slopes differentiate those continuous input nodes,
rather than a BF16 quantizer. CUDA's wide probability halves, split reductions, Metal's probability storage and
M5 operand cuts still need backend-specific forward tapes and contracts. The host check qualifies none of those
precision profiles, RMS/RoPE, the complete loss graph or checkpoint learning.

## Shared family, fitting and persistence

Use the shared Living Weights family descriptor, learner and shard writer, as for the dense Qwen pocket. The gate
in `living/pocket_save.zig` checks that saved gate/up/down values equal the accepted live values and retains pending
save state on failure. Extend the shared two-input neuron activation and its fit/rounding helpers to Gemma's
`gelu_pytorch_tanh` GeGLU before declaring the family. Tensor names and config scope must come from the admitted
Gemma namespace and geometry. This first step leaves that shared family table and save gate unchanged.

Start native learning qualification with MLX affine4/group64 and whole-group pocket widths. Reuse the dense widened
reader and original-column copy logic; keep the original packed weight, scale and bias bytes, tied head and config
fields exact. Give CUDA and Metal the same serialized family/tensor/quantization description and neuron ordering.
Reject unsupported learning formats by name before allocation. Serving keeps every format it already admits.

The future shared gate must round candidate neurons exactly as the writer will, evaluate the stored values, then
require held-out recall, earlier-fact retention, unrelated-fact checks and exact take-out/delete/rollback before
commit. Check saveBegin/saveFinish, cancellation, allocation and write failures through the real shared learner.
Its internal save reader handles pending state without reopening that folder through normal public admission.
Reopen each successful CUDA save on Metal and each Metal save on CUDA, with original tensor hashes retained.

## CUDA pocket and trainer

1. Separate shipped MLP columns from appended neurons at load. At the normalized pre-feedforward input, compute
   extra GeGLU neurons and add their down projection before post-feedforward normalization. Preserve sandwich
   norms and the learned layer scalar. Apply the same seam in prompt chunks, verification windows and decode.
   An empty pocket performs no new allocation, synchronization, readback or launch. An untaught stream reads
   the shipped columns even beside a taught stream.
2. Retain the actual serving forward's pre/post-normalization rows, scalar, GeGLU inputs, final normalized rows
   and capped-head nodes. Implement mean answer loss once and the tied head's input adjoint, including the exact
   softcap stores. Follow the full residual/sandwich-norm chain back to each requested pocket subject row.
3. Capture prepared Q/K/V and each query's actual lower/upper bound, position, rotary frequencies, cache owner
   and logical generation. Accumulate Q/K/V adjoints with the host formula as an independent mathematical check.
   Add the actual CUDA probability-half, context and projection cut conventions and independent local controls.
4. Pull Q/K back through learned RMS and the stored rotary table. Pull V back through unweighted RMS. On tied
   global layers, add both distinct preparation pullbacks into the shared raw K projection before its affine
   input adjoint. A prepared K buffer and a prepared V buffer represent different nodes despite that raw tie.
5. Qualify 31B and 12B separately. Per-layer-input and shared-KV-owner variants need their additional backward
   paths and owner accumulation before admission. Keep those learning configurations refused until then.

## Metal pocket and trainer

Use the same descriptor, fitting policy, saved tensor layout and acceptance path. Follow the dense Metal trainer's
buffer ownership and command-completion structure with Gemma operators. Keep training scratch separate from served
streams; retain captured rows until GPU completion, drain before publishing a replacement pocket, and invalidate
standing, draft and prompt-cache state after every edit. Carry each stream's untaught selection through every pass.

Implement native Metal backward stages for the saved Gemma forward. The host module remains a test oracle, never
an inference or training CPU fallback. Register each new operator's precision nodes, actual geometry, exceptional
classes, repeats, guards, accumulator witnesses and planted source controls. Keep unmeasured compiler/resource
records pending. Qualify pre-M5 and M5 independently, including their probability-product conventions.

## Gates before enabling native learning

Use a tiny nonzero Gemma-shaped model to compare the complete trainer and serving loss and directional gradients,
including prefix subject rows with no own loss, sliding/global transitions, GeGLU, sandwich norms, scalar and capped
head. Retain the existing whole-model directional rule, every failure and every refusal. A local ideal-attention
pass is only one component of that receipt.

Run CUDA and pre-M5/M5 cells with pinned binaries, runtime, model/tokenizer revisions, quantization, seeds and cache
settings. Require original-folder tokens and speed against main, drafted/plain equality, mixed taught/untaught
streams, all prompt cuts, image-boundary serving, resume/restart, rejection cleanup and the save round trip.
Measure learning time and peak host/device memory separately. This host step claims no real-model learning,
retention, device fidelity, speed or portability result.
