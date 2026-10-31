# Nanbeige4.3-33B, native Metal

This recipe covers text generation with [`Nanbeige/Nanbeige4.3-33B`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B)
(model_type `llama`: a dense 33B Llama-architecture model, 512K context, thinking and tool calls in its chat template)
on the Metal engine, served from the bf16 release as published: no conversion, nothing lost. It was brought up on an
M5 Ultra (256 GB); other Macs are untested. The family serves any `llama` checkpoint with heads of 128, default RoPE,
no biases and its own output head, in bf16; it refuses others by name.

## Build and serve

```sh
hf download Nanbeige/Nanbeige4.3-33B --local-dir models/nanbeige4.3-33b
zig build native -Doptimize=ReleaseFast -Dcpu=native
zig build grammar -Dxgrammar=PATH_TO_XGRAMMAR_v0.2.8 -Dcpu=native   # structured output
LLAMA_LOAD_LIMIT_GB=150 zig-out/native/bin/tensorfold-native serve models/nanbeige4.3-33b --name nanbeige --port 8888 \
  --context 131072 --parallel 2 --prompt-cache-gib 8
```

`LLAMA_LOAD_LIMIT_GB` caps weights, scratch and caches together (75% of RAM when unset). The weights take 61.7 GiB; a
stream's caches take 208 KiB a position (every one of the 52 layers keeps bf16 keys and values), so a 131,072-token
stream holds 26 GiB and the full 524,288-token context 104 GiB: 150 GB holds two 131,072-token streams, or one of
262,144; the whole context needs a Mac with the weights and 104 GiB free (192 GB, the 75% default on a 256 GB Mac,
holds one 524,288-token stream). Thinking is on by default, as in the chat template; a request sends
`"chat_template_kwargs": {"enable_thinking": false}` to turn it off. The server forwards only the thinking switch to
the template: earlier turns' reasoning stays out of the prompt (the template's `preserve_thinking` default) and tool
calls take the template's XML form (its `tool_call_format` default), which the server parses.

## What the family is

`zig/src/families/llama/` follows the Spark-X2.6 Metal family, with HF's `modeling_llama` as the reference:

- The reference's rounding points, its PyTorch code in bf16: the residual stays bf16 and each add rounds to it; the
  norm is `bf16(w * bf16(x * rsqrt(mean(x^2) + eps)))` with the mean in fp32; RoPE is `bf16(bf16(x * c) + bf16(o * s))`
  with the fp32 angle's cos and sin rounded to bf16 as HF rounds them, pairs (d, d + 64); the MLP is
  `bf16(bf16(silu(g)) * u)` (`kernels/metal/llama/glue.metal`).
- Projections over the release's bf16 rows, read in place from the mapped shards (no copies, no stacking): prompt
  chunks of up to 2,048 rows on `gemm_nax`'s tensor-unit GEMM (`gemm.metal`); decode windows on `ll_matvec`, a
  simdgroup an output row, fp32 sums rounded once as torch's bf16 linear rounds them. q, k and v write side by side
  into one row, gate and up into another.
- Attention: prompt chunks on the tensor units (Gemma 4's prompt attention over a position-major cache,
  `attention_nax.metal`); decode rows on sdpa_vector lanes over key blocks at absolute positions, so a row's bits
  never depend on the rows beside it.
- The lane host's shared rounds, a lone stream's steps drawn on the GPU a step ahead (argmax, or `tf_sample_full`),
  copy drafts and the prompt cache.

The Metal glue kernels are not registered with the kernel contract, as with the other Metal family glue.

## Measured (M5 Ultra, the bf16 release, 26 Oct 2026)

Against Hugging Face's `modeling_llama` on the release (transformers 4.57.1, bf16 on the CPU): 1,500 positions
teacher-forced, mean KL 0.0015, top-1 99.1%; 64 positions after a 9,000-token prompt, mean KL 0.0013, top-1 100%. Six
greedy chats of 64 tokens (code, prose, Chinese, a document, with and without thinking): three equal HF's token for
token; the other three part at a position where the engine's two best logits are the same bf16 value. One-row steps
and windows of 7 or 16 rows give the same logits bit for bit after prompts of 64 and 1,500 tokens.

Served through the recipe harness (an idle 97 GiB server of another model was loaded beside it, so memory readings
include it), `--context 131072 --parallel 2`, greedy 256-token replies:

| streams | prose | code | structured | list | first token | a round |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.4 | 16.3 | 16.4 | 16.4 | 0.23-0.26 s | 61.4 ms |
| 2 | 31.8 | 28.0 | 31.7 | 31.7 | 0.34-0.38 s | 66.4 ms |

Tokens a second, all streams together. One stream reads the 61.7 GiB of weights each step; two streams read them once
for both rows. Sampled replies (temperature, top-p, seeds) decode at 16.2-16.3 tokens/s, the same as greedy.

| cold prompt | 2,109 | 8,254 | 32,827 | 61,502 | 131,132 tokens |
|---|---:|---:|---:|---:|---:|
| tokens a second | 1,259 | 1,313 | 1,074 | 866 | 563 |
| time to first token | 1.68 s | 6.29 s | 30.6 s | 71.0 s | 233 s |

A fresh server, each prompt starting with words no earlier one did; the 131,132-token prompt on a
`--context 262144 --parallel 1` server, the others on the 131,072-token one. With thinking on, within 4%. A multi-turn
chat's third turn, resumed from 5,493 of its 5,649 kept tokens, starts in 0.34 s against 4.5 s cold, with the same reply.

- Structured output (json_object, json_schema, regex, choice, grammar) 18/18, drafted == plain 12/12; tool calls
  (auto, required, named, thinking, streamed) 5/5; seeded sampling repeats 9/9.
- A 60-turn soak (3 chats of 20 turns beside a decoding stream) and 12 concurrent replies each equal the request
  alone; streaming 4/4, multi-turn recall 8/8, stop strings 3/3, thinking 10/10, max_tokens 3/3.
- A passphrase at 10%, 50% and 90% of a 128,550-token prompt comes back exactly on the 131,072-token server.

## Performance options

Each keeps the replies' bits.

- **Streams (`--parallel`).** One stream decodes at 16.4 tokens/s, bound by reading the weights each step; two give
  28-32 tokens/s in all (66 ms a round against 61), each reply equal to its run alone. The row kernel holds up to 32
  rows a launch; the caches set the count (26 GiB a 131,072-token stream).
- **Context and memory (`--context`, `LLAMA_LOAD_LIMIT_GB`).** 208 KiB a token. Cold prompts run 1.1-1.3k tokens/s
  to 33k tokens, 866 at 61k and 563 at 131k (attention grows with the prompt).
- **Prompt cache (`--prompt-cache-gib`).** A resumed turn starts in 0.34 s against 4.5 s cold; kept states give way to
  streams that need the room.
- **Copy drafts.** On from the lane core, replies equal to drafts off; they accepted 0-30% of their tokens on the
  harness's prompts and gave no measurable decode gain (drafted 16 tokens/s, undrafted 16).
- **Sampling.** Draws run `tf_sample_full` over the whole 166,912-token vocabulary (the 1,024-candidate kernel is
  compiled for a 131,072-token vocabulary): sampled replies decode at the greedy speed.
- **Not offered.** No int8 or fp8 cache, no fast-math, no reduced-precision sums, no quantized weights.

## Not done

No quality suite was run. No drafter is served (the family has no draft head; copy drafts come from the lane core).
Logprobs are refused by name: the release's tokenizer decodes through a `Strip` step, which the engine's tokenizer does
not count as token-local, so the server cannot give a token's bytes on its own.
Only the M5's tensor units were checked; M1-M4 Macs and smaller RAM are untested. Tied heads, other head sizes,
Llama 3's scaled RoPE and biased projections are refused at load rather than served untested.
