# Qwen4 dense native preparation

The `qwen4` and `qwen4_text` adapters prepare a dense version of the related Flash Next architecture.
No released Qwen4-27B checkpoint was available for this preparation.
The supplied Intel kernels establish supported shapes, not the complete model configuration.
Real checkpoint, Metal and CUDA device qualification remain unchecked.

Config admission requires explicit hidden, vocabulary, layer, MLP, HC, GDN, QSA, RoPE and PLE geometry.
It accepts `linear_attention`, `indexed_attention` and the `full_attention` spelling for indexed layers.
It refuses MoE, vision, unknown fields and an undeclared positive MTP head by field name.
Copy proposals use the shared lane core; they do not imply a checkpoint MTP head exists.
Native kernels currently require hidden size divisible by 256, GDN head dimensions 128, convolution length 4,
attention head dimension 256 and index dimension 128.
Each stream window admits at most 16 rows and a shared batch at most 128 rows.
Unsupported geometry is refused before device allocation.

The shared graph is `families/flashnext/forward_graph.zig`.
It owns HC, the GDN/QSA boundary, optional PLE and the dense or routed feed-forward boundary.
Qwen4 uses the shared projection and pointwise operators and geometry specializations of maintained mixer sources.
Existing Flash Next factories retain their fixed checkpoint dimensions and their arithmetic.
The new native adapters use the existing engine API, lane host, prompt cache and verification machinery.

Weights remain as stored: BF16, MLX affine 4/8-bit groups 32/64/128, ModelOpt FP8 or ModelOpt NVFP4.
Each projection keeps its own recipe, scales and bias metadata.
Activations stay BF16 and reductions stay FP32.
Small BF16/F16 vectors widen exactly to FP32; original F32 vectors retain their dtype-specific rounding policy.
HF centered norms use `1 + gamma`.
An MLX namespace must explicitly state whether its norm weights are gamma or prepared scale.
The loader validates every tensor header and PLE hash vector before uploading weights.

Synthetic host tests use an independent binary64 forward with explicit BF16 nodes.
Nonzero native-compatible fixtures exercise the actual loader, packed graph, state, cache and shared lane core
through CPU operator instruments, including every prompt cut, concurrent streams and copied proposals.
These tests establish host integration and state semantics; they do not qualify device arithmetic.
Generated native source tests and normal contract-runner Cases cover the individual operators separately.
Attention uses the shared binary64 softmax reference and its stated BF16-probability error allowance.
The maintained Metal split-softmax retains FP32 probabilities; the CUDA tensor-core path rounds P to BF16.
No cross-backend byte equality or whole-checkpoint HF fidelity is claimed from those bounds.

On a host without GPU toolchains, the shader-free checks are:

```sh
zig build test-qwen4-host test-qwen4-projection-host test-qwen4-gdn-host
zig build test-qwen4-reference-host
zig build test-qwen4-qsa-host test-qwen4-native-mixers-host
zig build test-qwen4-projection-bodies test-qwen4-cuda-bodies
```

The normal Metal and CUDA contract runners collect the new Cases.
New CUDA launch records retain pending measurements until actual compiler receipts exist for each SM.
Before advertising a released checkpoint, check its config and tensor names, run the complete backend contract,
and run fresh/drafted/concurrent/resumed/prompt-cut model comparisons with its actual tokenizer and weights.
Measure cold prefill, decode and memory on each claimed GPU; no rate or memory-fit claim is established here.

The explicit synthetic device gates generate the same nonzero safetensors through the real checkpoint writer.
They load the serving native engine and compare all 384 BF16 logits and 12 GPU greedy picks with a binary64 fixture.
Metal uses its FP32 split-softmax probability profile; CUDA retains its BF16 tensor-core probability profile.
The Metal reference declares FP32 RMS scalar and rotary scalar nodes, including its stored FP32 frequency table.
No comparison tolerance or logit mask is used. These backend profiles do not claim HF or cross-backend byte parity.
They require byte equality for five repeats, every prompt cut, saved-state resume, concurrent streams,
partial acceptance and abort/retry. The shared lane adapter also checks copied proposals against plain decoding.
A learned MTP head is not declared by this fixture and remains refused by config admission.

On a Mac with the Metal toolchain:

```sh
zig build test-qwen4-metal
```

This step compiles the production shaders at runtime and executes the GPU. It is separate from default host tests.
`zig build compile-qwen4-metal-device` emits the same executable without executing it or producing shaders.
For CUDA on a GB10, build the runner and embedded images, then run both gates:

```sh
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
./zig-out/bin/tf-cuda-test qwen4
zig build kernel-contract -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
```

`zig build compile-cuda-tests` with the same options emits the runner without installing or executing it.
The normal CUDA contract collects the Qwen4 operators and their planted controls, and resolves all four images
for the 32 new launch registrations. Resource measurements are still required; pending records must fail limits.
Synthetic device success does not qualify an unreleased checkpoint or establish cross-backend equality.

Compile the generated runtime libraries and control variants before opening a device on a Mac with the Metal toolchain:

```sh
zig build test-qwen4-metal-sources -j8
```

This uses the runtime's Metal language and math settings and opens no GPU. The explicit `test-qwen4-metal` device gate
also depends on this compiler check. `emit-qwen4-metal-sources` writes the exact source inputs under
`zig-out/qwen4-metal/` for inspection without invoking a shader compiler.

The source inventory rejects buffer indices outside Metal's 0–30 range before emission. Operator repeat controls
share a four-byte constant struct at slot 17, after GDN's production bindings through slot 16. The CPU Metal
instruments also require explicit float-to-bfloat construction, so the generated HC control stores are type-checked
without a GPU; this supplements the actual Metal compiler check.

Before the full Metal comparison, the gate runs a fresh two-token case and prints `QWEN4_NODE` values for embedding,
PLE, HC sums and boundaries, GDN projections/state/output, QSA preparation/pooling/scores/parts/output, and the head.
The same captured two rows first face the legacy BF16-probability oracle as a labelled diagnostic, then the corrected
Metal oracle. No second forward or changed input is used for that before/after comparison.
`QWEN4_FIRST_DIFFERENCE` identifies the first unequal BF16 storage node per row. FP32 reductions and recurrent state
are diagnostic raw bits beside their binary64 values; they are not asserted equal to unrounded binary64 arithmetic.
The two-row QSA range is dense, so no sparse selection result is consumed. The full 12-row gate still checks logits,
tokens and every state seam strictly. The diagnostic copies stay on the inference queue and finish before readback.
Normal native inference has no trace sink and performs no diagnostic copies or host readbacks.

Regenerate the Metal reference with:

```sh
python3 tools/zig/qwen4_reference.py --config zig/src/families/qwen4/fixtures/native-dense.json \
  --probability f32 --native-scalars
```

Add `--tokens 3,5 --nodes` for the two-row node ledger. The default generator profile and CUDA fixtures are unchanged.
