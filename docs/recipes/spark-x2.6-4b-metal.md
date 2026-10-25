# Spark-X2.6-4B, native Metal

This recipe covers text generation with [`XHToken/Spark-X2.6-4B`](https://huggingface.co/XHToken/Spark-X2.6-4B),
revision `66e3ae7bee54e0a523d692cb8ef841f35409807b` (model_type `spark2_5`), on the Metal engine. It was brought up on
an M5 Ultra (256 GB); other Macs are untested. The CUDA side of the family is [its own page](spark-x2.6-4b-cuda.md).

## Convert, build, serve

The checkpoint is the CUDA page's 8-bit conversion (numpy only):

```sh
hf download XHToken/Spark-X2.6-4B --revision 66e3ae7bee54e0a523d692cb8ef841f35409807b --local-dir models/spark-x2.6-4b
python3 tools/zig/spark25_convert.py models/spark-x2.6-4b models/spark-x2.6-4b-8bit --bits 8
zig build native -Doptimize=ReleaseFast -Dcpu=native
zig build grammar -Dxgrammar=PATH_TO_XGRAMMAR_v0.2.8 -Dcpu=native   # structured output
SPARK_LOAD_LIMIT_GB=48 zig-out/native/bin/tensorfold-native serve models/spark-x2.6-4b-8bit --name spark --port 8888 \
  --context 262144 --parallel 4 --prompt-cache-gib 8
```

`SPARK_LOAD_LIMIT_GB` caps weights, scratch and caches together (75% of RAM when unset). The weights take 4.1 GiB; a
stream's caches take 36 KiB a position in the nine full layers plus 270 MiB of sliding rings, so 48 GB holds four
262,144-token streams or one of 1,048,576 tokens (`--context 1048576 --parallel 1`). Thinking is on by default, as in
the chat template; a request sends `"chat_template_kwargs": {"enable_thinking": false}` to turn it off.

## What the port is

`zig/src/families/spark25/metal_*.zig` follow Flint's Metal family and share `config.zig` with the CUDA side:

- The reference's rounding points (its PyTorch code in bf16): the residual fp32 from the embedding on, each norm
  `bf16(w * (x * rsqrt(mean(x^2) + eps)))` in fp32, RoPE in fp32 from the same inverse-frequency table as CUDA
  (all 256 dims in sliding layers, the first 64 in full ones), the head gate `bf16(att * bf16(sigmoid(g)))` and the MLP
  `bf16(bf16(gelu(g)) * u)` with an erf to under one ulp (`kernels/metal/spark25/glue.metal`).
- `q_k_v_proj` and `g_proj` stacked into one projection at load (6,160 rows, zero rows to 6,208 for the prompt
  matmul's tiles), gate and up into another.
- Prompt chunks of up to 2,048 rows on the tensor units (gemma4's prompt attention, `attention_nax.metal`); decode
  rows on sdpa_vector lanes over key blocks at absolute positions, so a row's bits never depend on the rows beside it.
  Sliding layers keep a ring of the window and one chunk.
- The lane host's shared rounds, a lone stream's steps drawn on the GPU a step ahead, copy drafts, the prompt cache.

The Metal glue kernels are not registered with the kernel contract, as with the other Metal family glue.

## Measured (M5 Ultra, 8-bit, 25 Oct 2026)

Against HF's own code (transformers 4.57.1) on the same 8-bit weights dequantized to bf16:

- 1,500 positions teacher-forced, in 16-row windows: mean KL 0.0011 against HF in fp32, top-1 98.9%; HF's own bf16
  run is 0.0012 from the same fp32 run. After 2,600- and 9,000-token prompts (one and five prompt chunks), 64 forced
  steps: mean KL 0.0004 and 0.0003, no row above 0.007.
- Greedy replies on four chat prompts (code, prose, Chinese, a 3,547-token document), 96 tokens each: all four equal to
  HF's in fp32 (the document's ends at HF's end token after 89, every one equal).
- Windows of 16 and of 7 rows give the one-row steps' logits bit for bit after prompts of 64, 1,500, 2,600 and 4,100
  tokens (the ring wraps at 2,560).

The 8-bit conversion against the bf16 release (both through HF in fp32, the same 1,500 positions): mean KL 0.0021,
top-1 98.0%; two of the four greedy replies above equal the release's over 96 tokens, the others part at tokens 5
and 48.

Through the recipe harness (`--context 262144 --parallel 4`):

- `quick`: structured output 18/18 (drafted equal to plain 12/12), tools 5/5, seeded sampling 9/9, a resumed turn
  equal to the cold one (5,499 of 5,630 tokens cached, 0.05 s to the first token against 0.65 s), a 60-turn soak and
  concurrent streams each equal to the request alone, and the suite's streaming, stop, copy and max-tokens checks.
- Decode: 166 tokens/s with one stream, 258-268 in all with two.
- Cold prompts: 2,362 tokens 8,201 tokens/s, 9,269 tokens 9,241, 37,121 tokens 7,753, 148,552 tokens 4,512 (32.9 s).
- Needles: a code at 10%, 50% and 90% of a 254,036-token prompt, each found (81.9-84.5 s); at 50% of a 1,032,810-token
  prompt on a `--context 1048576` server, found in 1,071 s.

## Not done

No quality suite was run. No drafter is served (the family has no draft head; copy drafts come from the lane core).
Only the M5's tensor units were checked; M1-M4 Macs and smaller RAM are untested.
