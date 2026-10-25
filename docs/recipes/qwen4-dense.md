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
zig build test-qwen4-qsa-host test-qwen4-native-mixers-host
zig build test-qwen4-projection-bodies test-qwen4-cuda-bodies
```

The normal Metal and CUDA contract runners collect the new Cases.
New CUDA launch records retain pending measurements until actual compiler receipts exist for each SM.
Before advertising a released checkpoint, check its config and tensor names, run the complete backend contract,
and run fresh/drafted/concurrent/resumed/prompt-cut model comparisons with its actual tokenizer and weights.
Measure cold prefill, decode and memory on each claimed GPU; no rate or memory-fit claim is established here.
