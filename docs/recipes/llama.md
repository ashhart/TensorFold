# Llama on MLX

This recipe adds text-only `model_type=llama` checkpoints to the existing lane engine.
It does not add a CUDA backend or a basal prompt/API adapter.

```bash
tensorfold info Remek/basal-1.5-mini-MLX-8bit
tensorfold serve Remek/basal-1.5-mini-MLX-8bit --backend mlx
```

## Supported layout

- Full causal attention in every layer, GQA or MHA, RMSNorm and SwiGLU.
- Default, full-dimension RoPE; Transformers 5 `rope_parameters` is normalized to mlx-lm's `rope_theta`.
- Optional attention and MLP biases, distinct from affine quantization offsets.
- bf16 decoder projections or MLX affine 8-bit/group-64 projections, including mixed dense/quantized layers.
- Unquantized bf16 embedding and untied output head; bf16 norm, bias and affine scale/offset tensors.
- Native context of at least 576 tokens when declared, so server admission fits the 256-token minimum prompt chunk.

Scaled or partial RoPE, sliding attention, tied embeddings, quantized embeddings/heads, other quantization
formats and external draft heads are refused. Configuration checks run before downloading weights;
safetensors headers and loaded tensors are checked before kernel preparation.

The real-weight checkpoint exercised on Apple M4 is `Remek/basal-1.5-mini-MLX-8bit`, revision
`6ea29c486e39e1a217932ab56b222c0d1bc4ba82`, with MLX 0.32.3 and mlx-lm 0.32.0. Basal 4.5B projection
geometry is covered by a synthetic layer test. Its real 8-bit weights, revision
`79f5a7cbbd028758c506f9aca5fec1ef431ee893`, were also exercised for startup and basal label scoring,
but not the full generation/snapshot qualification. Other Apple Silicon GPU generations remain untested.
Recognition of a configuration is not qualification of its weights, quality or speed.

## Arithmetic and state

`mlx_lm.utils.load_model` reads the checkpoint. Prefill uses mlx-lm's native Llama forward and the engine's
planned chunks. Quantized projection instances retain native prefill arithmetic even if another family
installs a process-wide quantized-linear wrapper.

Decode uses Llama attention and MLP logic, not Qwen's decoder. Quantized projections share the existing
generic affine row kernels across verification rows and streams. Norms, bf16 projections, head and
attention preserve one-query arithmetic. Each stream has an independent KV cache and RoPE offset.

Load-time probes check every width up to 16, live KV state, partial rollback and continuation. Shared
probes cover every admitted aggregate row count and unequal stream lengths. Failed probes restrict the
window width or disable shared forwards. Non-chain parents are not supported. The snapshot fingerprint
covers source dependencies and the instance's frozen quantized-kernel selection.

Drafting uses the engine's existing context-copy proposer; no MTP or DFlash head is required. Verification
commits only the accepted prefix. Resume and disk snapshots keep the same prompt-chunk plan.

## Qualification boundaries

Exactness is an engine contract: drafted/serial, resumed/fresh and concurrent/solo replies must match.
It does not require logit equality with stock mlx-lm, whose quantized reductions differ. Use the same
weights for a separate teacher-forced stock-forward comparison and report logit error, top-token
agreement and NLL. A short fixture does not establish task accuracy or calibrated probabilities.

The real-weight checks exercised greedy and seeded sampling, accepted and rejected draft tokens,
shared rounds and a disk snapshot at a 2,048-token prompt boundary. They are small-fixture checks,
not a full benchmark or qualification of every supported checkpoint.

`score_labels` already reads label logits through the lane engine without generation. This alone does
not make `/v1/decisions` compatible with basal. Basal requires its trained prompt and answer prefill,
option-order handling and calibration. Those remain a separate contribution; no generic decision prompt
is changed here.

A separate test-only comparison kept the original basal 1.5 server, prompt, tokenizer, option orders and
calibration, replacing only its MLX backend with sequential `score_labels` calls. On each of mini and
4.5B, 88 valid public/example requests covered 114 typed questions and 264 order prompts. All 105
non-multi argmax answers matched, but calibrated probabilities and confidence boundaries did not.
Real HTTP boundary cases changed an `act` action for each model and a `multi` selection for 4.5B.
This does **not** qualify TensorFold as a drop-in replacement for basal's native MLX backend.

The sequential scoring probe reused no state prefix and was about three times slower than basal's
native shared-prefix path with 5–12 questions. It is not an implementation or speed qualification of
basal's state-once/ask-many optimization. A layout bisect on identical loaded weights reproduced the
short-prompt differences through native shared-prefix/suffix batching; independent whole-prompt native
scoring matched TensorFold on those short cases. Recalibration or an exact shared-prefix scoring path
requires a separate contribution and new qualification.
