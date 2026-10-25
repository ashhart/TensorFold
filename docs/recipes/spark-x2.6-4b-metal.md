# Spark-X2.6-4B, native Metal

This recipe covers text generation with [`XHToken/Spark-X2.6-4B`](https://huggingface.co/XHToken/Spark-X2.6-4B),
revision `66e3ae7bee54e0a523d692cb8ef841f35409807b` (model_type `spark2_5`), on the Metal engine, served from the bf16
release as published: no conversion, nothing lost. It was brought up on an M5 Ultra (256 GB); other Macs are untested.
The CUDA side of the family is [its own page](spark-x2.6-4b-cuda.md).

## Build and serve

```sh
hf download XHToken/Spark-X2.6-4B --revision 66e3ae7bee54e0a523d692cb8ef841f35409807b --local-dir models/spark-x2.6-4b
zig build native -Doptimize=ReleaseFast -Dcpu=native
zig build grammar -Dxgrammar=PATH_TO_XGRAMMAR_v0.2.8 -Dcpu=native   # structured output
SPARK_LOAD_LIMIT_GB=64 zig-out/native/bin/tensorfold-native serve models/spark-x2.6-4b --name spark --port 8888 \
  --context 262144 --parallel 4 --prompt-cache-gib 8
```

`SPARK_LOAD_LIMIT_GB` caps weights, scratch and caches together (75% of RAM when unset). The weights take 7.7 GiB; a
stream's caches take 36 KiB a position in the nine full layers plus 270 MiB of sliding rings, so 64 GB holds four
262,144-token streams or one of 1,048,576 tokens (`--context 1048576 --parallel 1`). Thinking is on by default, as in
the chat template; a request sends `"chat_template_kwargs": {"enable_thinking": false}` to turn it off. The CUDA
page's MLX affine conversions also load (the same row and prompt kernels Gemma 4 uses); they are not needed on a Mac,
and the 8-bit one costs a mean KL of 0.0021 against the release over 1,500 positions.

## What the port is

`zig/src/families/spark25/metal_*.zig` follow Flint's Metal family and share `config.zig` with the CUDA side (a
checkpoint without a quantization table is the bf16 release; CUDA refuses it by name):

- The reference's rounding points (its PyTorch code in bf16): the residual fp32 from the embedding on, each norm
  `bf16(w * (x * rsqrt(mean(x^2) + eps)))` in fp32, RoPE in fp32 from the same inverse-frequency table as CUDA
  (all 256 dims in sliding layers, the first 64 in full ones), the head gate `bf16(att * bf16(sigmoid(g)))` and the MLP
  `bf16(bf16(gelu(g)) * u)` with an erf to under one ulp (`kernels/metal/spark25/glue.metal`).
- Projections over the release's bf16 rows: prompt chunks of up to 2,048 rows on `gemm_nax`'s tensor-unit GEMM
  (`gemm.metal`); decode windows on `sp_matvec`, a simdgroup an output row, fp32 sums rounded once as torch's bf16
  linear rounds them. `q_k_v_proj` and `g_proj` are stacked into one projection at load, gate and up into another.
- Attention: prompt chunks on the tensor units (gemma4's prompt attention, `attention_nax.metal`); decode rows on
  sdpa_vector lanes over key blocks at absolute positions, so a row's bits never depend on the rows beside it. Sliding
  layers keep a ring of the window and one chunk.
- The lane host's shared rounds, a lone stream's steps drawn on the GPU a step ahead, copy drafts, the prompt cache.

The Metal glue kernels are not registered with the kernel contract, as with the other Metal family glue.

## Measured (M5 Ultra, the bf16 release, 25 Oct 2026)

Against HF's own code (transformers 4.57.1) on the release:

- 1,500 positions teacher-forced, in 16-row windows: mean KL 0.00076 against HF in fp32, top-1 99.6%; HF's own bf16
  run is 0.00048 from the same fp32 run, top-1 99.3%. One position (a near tie, the worst on every comparison made)
  holds the maximum, 0.57; no other is above 0.05. After a 9,000-token prompt (five chunks), 64 forced steps: mean KL
  0.0005, no row above 0.006.
- Greedy replies on four chat prompts (code, prose, Chinese, a 3,547-token document), 96 tokens each: all four equal to
  HF's in fp32 (the document's ends at HF's end token after 89, every one equal).
- Windows of 16 and of 7 rows give the one-row steps' logits bit for bit after prompts of 64, 1,500, 2,600 and 4,100
  tokens (the ring wraps at 2,560).

Through the recipe harness (`--context 262144 --parallel 4`):

- `quick`: structured output 18/18 (drafted equal to plain 12/12), tools 5/5, seeded sampling 9/9, a resumed turn
  equal to the cold one (5,499 of 5,624 tokens cached, 0.065 s to the first token against 0.59 s), a 60-turn soak and
  concurrent streams each equal to the request alone, and the suite's streaming, stop, copy and max-tokens checks. Its
  multi-turn check recalled 7 of 8 facts: asked for five short lines, the model gave five and left one fact out.
- Decode: 106 tokens/s with one stream (9.5 ms a step: 7.7 GiB of weights a step, the memory bandwidth), 179-190 in
  all with two, 296-326 with four (12.8 ms a round: the weights read once for every row).
- Cold prompts: 2,350 tokens 9,438 tokens/s, 9,354 tokens 10,940, 37,174 tokens 9,058, 148,588 tokens 5,002 (29.7 s).
- Needles: a code at 10%, 50% and 90% of a 254,345-token prompt, each found (73.1-76.3 s); at 50% of a
  1,032,662-token prompt on a `--context 1048576` server, found in 1,035 s with 76% of the Mac's memory still free.

## Performance options

Each keeps the replies' bits; the numbers are the M5 Ultra's.

- **Streams (`--parallel`).** One stream decodes at 106 tokens/s, bound by reading the 7.7 GiB of weights each step.
  The weights are read once a round for every stream's row: two streams give 179-190 tokens/s in all, four 296-326
  (12.8 ms a round against 9.6), each reply equal to its run alone; the first token moves from 0.05-0.07 s to
  0.13-0.17 s at four. The row kernel holds up to 32 rows a launch.
- **Weights.** The release's bf16 is exact. The CUDA page's MLX affine conversions also load: the 8-bit one decodes at
  166 tokens/s (4.1 GiB a step) for a mean KL of 0.0021 against the release and greedy replies that part within a
  hundred tokens on two of four chats; 4-bit was not measured. Neither is needed on a Mac.
- **Context and memory (`--context`, `SPARK_LOAD_LIMIT_GB`).** 36 KiB a token in the nine full layers plus 270 MiB of
  rings a stream: 64 GB holds four 262,144-token streams or one of 1,048,576. Cold prompts run 9.4k-10.9k tokens/s to
  37k tokens, 5.0k at 149k and about 1.0k averaged over a million (the full layers' attention grows with the prompt).
- **Prompt cache (`--prompt-cache-gib`).** A resumed turn (5,499 of 5,624 tokens kept) starts in 0.065 s against
  0.59 s cold, with the same reply; kept states give way to streams that need the room.
- **Copy drafts.** On by default from the lane core, replies equal to `drafts: false`; they accepted 7-15% of their
  tokens on the harness's prompts and gave no measurable decode gain there. The family has no draft head.
- **Thinking.** On by default (the template's); it lengthens replies, not the cost a token; prompt speed with it on
  is within 1% of off.
- **Not offered.** No int8 or fp8 cache, no fast-math, no reduced-precision sums.

## Not done

No quality suite was run. No drafter is served (the family has no draft head; copy drafts come from the lane core).
Only the M5's tensor units were checked; M1-M4 Macs and smaller RAM are untested.
