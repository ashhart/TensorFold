# Spark-X2.6-4B, native CUDA

This recipe covers text generation with [`XHToken/Spark-X2.6-4B`](https://huggingface.co/XHToken/Spark-X2.6-4B),
revision `66e3ae7bee54e0a523d692cb8ef841f35409807b` (model_type `spark2_5`, the same architecture as Spark-X2.5-4B),
on the CUDA engine. It was brought up on an RTX 4090 (`sm_89`) under Windows; other CUDA GPUs are untested.

## Convert, build, serve

The release is bf16 only; the family reads MLX affine checkpoints, so convert it first (numpy, no MLX needed):

```sh
hf download XHToken/Spark-X2.6-4B --revision 66e3ae7bee54e0a523d692cb8ef841f35409807b --local-dir models/spark-x2.6-4b
python tools/zig/spark25_convert.py models/spark-x2.6-4b models/spark-x2.6-4b-8bit --bits 8
zig build native install -Dsm=89 -Dstrip=true -Dnvcc="$CUDA_PATH/bin/nvcc"
zig-out/native/bin/tensorfold-native serve models/spark-x2.6-4b-8bit --name spark --port 8888 --context 32768 --parallel 4
```

`--bits 4` writes a 4-bit checkpoint instead (smaller, lower fidelity). Thinking is on by default, as in the chat
template; a request sends `"chat_template_kwargs": {"enable_thinking": false}` to turn it off.

## What the family is

`zig/src/families/spark25/` follows Gemma 4's CUDA family, on the same shared operators:

- 36 layers, three sliding-window layers (window 512) to each full one, 16 query heads of 256 over 4 key heads.
- `q_k_v_proj` and the per-head `g_proj` are packed into one 6,160-row projection at load.
- RoPE on all 256 dims (theta 10,000) in sliding layers and on the first 64 (pairs d, d + 32; theta 5,000,000) in full
  ones, through `qk_rows`; no QK norm; attention scale 1/16.
- The output gate is one sigmoid a head, rounded to bf16 before the product (`tf_head_gate_rows`), and the MLP is
  `bf16(gelu_erf(gate)) * up` with PyTorch's exact GELU (`tf_gelu_erf_rows`), both new in `act_rows.cu`.
- The residual stays fp32 from the embedding on, as the reference keeps it (`norms.addRms`), with fp32 norm weights.
- Tied head, no soft-cap. Prompts reuse kept states (`snap.zig`); copy drafts come from the lane core.

The tokenizer needed `Digits` (`individual_digits`) in `core/tokenizer/steps.zig`.

## Measured (RTX 4090, 8-bit, Windows, 25 Oct 2026)

- Tokenizer: equal to the HF tokenizer's IDs on mixed English, Chinese, Japanese, code, emoji and digit text, and on
  rendered chat prompts up to 2,319 tokens.
- Greedy replies against the HF reference (transformers 4.57.1) on the same weights dequantized to bf16: 3 of 4
  prompts equal over 96 tokens, among them a 2,319-token prompt across the window, the ring and five chunks; the
  fourth equal for 137 characters, then a near tie.
- One stream: 136 tokens/s on prose, 144 on code, greedy or sampled. Four streams: 414 tokens/s in all, each reply
  equal to its solo run.
- Time to first token: 0.24 s at 2k tokens, 0.93 s at 8k, 1.93 s at 16k, 4.65 s at 29k.

## Not done

The two new kernels are under the CUDA contract (`zig/tests/cuda/contract_spark_act.zig`, its float64 ends in
`core/act_ref.zig`, a control for each kernel's rounding point) with launch records in
`core/contract/spark_cuda_records.zig`. There is no fixture-based exactness test against dumped reference rows yet,
and no drafter or structured output is served.
