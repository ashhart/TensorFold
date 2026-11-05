# Ripple-65B-Preview, native Metal

This recipe covers text generation with ricursiveAI's Ripple-65B-Preview (model_type `ricursive`: a dense 65B model,
80 layers, contexts up to 1M tokens by its model card, thinking and tool calls in its chat template) on the Metal
engine. Two of the publisher's releases are served as published, read in place from their shards with nothing
converted:

- [`ricursiveAI/Ripple-65B-Preview-NVFP4`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4) (revision
  `99eea4477a001a41b86914e41feb4b2ed86105e5`, 53 GiB): ModelOpt NVFP4 MLPs, attention, embeddings and head in bf16.
  The recommended one on a 256 GB Mac: it leaves 135 GiB for caches.
- [`ricursiveAI/Ripple-65B-Preview`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) (revision
  `240d5966cbe00f0627c699e59ef58560b60303cf`, 121 GiB): the bf16 release, the exact model.

It was brought up on an M5 Ultra (256 GB); other Macs are untested. Prompts run on Metal 4 tensor units, so it needs
an M5 or later: M1 to M4 Macs are refused by name at load. Other `ricursive` checkpoints, and either release quantized
any other way, are refused by name.

## Build and serve

```sh
hf download ricursiveAI/Ripple-65B-Preview-NVFP4 --revision 99eea4477a001a41b86914e41feb4b2ed86105e5 \
  --local-dir models/ripple-65b-nvfp4
zig build native -Doptimize=ReleaseFast -Dcpu=native
zig build grammar -Dxgrammar=PATH_TO_XGRAMMAR_v0.2.8 -Dcpu=native   # structured output
zig-out/native/bin/tensorfold-native serve models/ripple-65b-nvfp4 --name ripple --port 8888 \
  --context 262144 --parallel 2
```

The bf16 release serves the same way from its own directory. A stream's caches take 16.75 GiB at 262,144
tokens: 64 of the 80 layers slide over 1,024 keys and keep a 3,072-slot ring each (the window and a 2,048-row prompt
chunk); every fifth layer is global and keeps every key, in bf16. Under the default load limit (75% of RAM) a 256 GB
Mac holds the NVFP4 weights (54.8 GiB, wired between requests) and 135 GiB of caches: eight 262,144-token streams.
Thinking follows the chat template: Ripple writes `<think>` itself when it thinks and answers bare otherwise, and the
server files a reply without the block as answer, not reasoning, whatever `enable_thinking` says (the template has no
switch). `--context` takes up to 1,048,576 tokens, the card's 1M; 262,144 is the default and the longest checked
(below).

## What the family is

`zig/src/families/ricursive/`, with the release's `modeling_ricursive.py` and ricursiveAI's SGLang model as the
reference:

- **Layers.** PeriNorm (a norm before and after each sublayer, bf16 residual, each add rounded to it), per-head q
  and k norms, RoPE (theta 10,000) on sliding layers and none on global ones, a sigmoid output gate in fp32, the value
  residual (each layer's value input blends the first layer's scaled embeddings with its own normed input), and the
  clamped SwiGLU (gate at most 20, up within ±20; a NaN stays NaN, as `torch.clamp` keeps it). The glue is
  `kernels/metal/ricursive/glue.metal`.
- **Projections.** bf16 rows read in place from the mapped shards. Prompt chunks of up to 2,048 rows run on llama's
  tensor-unit GEMM; decode windows (up to 32 rows) run llama's `ll_matvec`. Every chunk is planned on the same
  2,048-row grid, so a resumed prompt equals the prompt served fresh.
- **NVFP4 MLPs** (`fp4.metal`, `fp4_gemm.metal`). E2M1 nibbles (low first), an unsigned E4M3 scale per 16 values,
  one fp32 global per tensor. A nibble times its scale is exact in bf16. Decode windows sum in fp32 and apply the global
  before the one bf16 rounding; prompt chunks first write the exact fp4 × scale values to a scratch and run the GEMM
  over them, each fp32 sum times the global before it rounds (CUTLASS's alpha epilogue).
- **Attention.** Prompt chunks on the tensor units over the sliding ring or every key (`attention_nax.metal`); decode
  rows over key blocks at absolute positions, so a row's bits never depend on the rows beside it.
- **Serving.** The lane host's shared rounds, greedy or `tf_sample_full` draws, copy drafts, the prompt cache.

**Activations stay bf16.** The publisher serves the NVFP4 release with SGLang's `flashinfer_cutlass`, which also
quantizes activations to fp4 (W4A4). TensorFold does not run reduced-precision activations, so the engine serves the
release's NVFP4 weights with bf16 activations and fp32 sums: what is measured below is that model, not the
publisher's W4A4 kernels.

Every kernel is registered with the kernel contract (`zig/tests/contract/metal_ricursive*.zig`, the roll call in
`contracts.zig`): the glue, decode and prompt attention for both layer kinds, the NVFP4 row kernels, dequantization
and GEMM, and llama's row projections, merge, argmax and GEMM again at Ripple's shapes. Fifteen planted controls each
edit one kernel's source, and each must turn its gate red. `tf-kernel-contract --only ricursive` on the M5 Ultra:

```
SUMMARY 36 kernels: 36 pass gates 1-4, 0 fail, 0 known findings; 15 controls: 15 red as planted, 0 missed; 0 limit problems; gates 571778 ms, limits 298 ms
```

## Correctness (M5 Ultra, 4 Nov 2026)

`tools/zig/ricursive_reference.py` runs a checkpoint on the CPU, one layer's weights at a time: in float64 (the
weights cast up exactly; the NVFP4 MLPs dequantized exactly), and with `--bf16-points` rounded to bf16 at every point
the release's bf16 forward rounds. Teacher-forced over a 2,586-token chat:

| engine logits | against | mean KL | notes |
|---|---|---:|---|
| bf16 release, 2,048-row prompt chunks | its float64 pass | 0.00086 | |
| bf16 release, 16-row decode windows | its float64 pass | 0.00073 | |
| NVFP4 release, 2,048-row prompt chunks | its float64 pass | 0.00069 | |
| NVFP4 release, 16-row decode windows | its float64 pass | 0.00107 | |
| NVFP4 release, prompt chunks | the bf16 release's float64 pass | 0.0060 | top-1 96.1% |
| NVFP4 float64 pass (no engine) | the bf16 release's float64 pass | 0.0059 | top-1 95.8% |

The last two rows are **what the NVFP4 release costs**: the publisher's quantization, not the engine's, about eight
times the engine's own rounding. The engine adds nothing measurable to it.

- Prompt chunks of 2,048, 1,000 and 333 rows give bit-identical logits; decode windows of 16 and 7 rows equal one-row
  steps bit for bit (both releases).
- A greedy bf16 reply of 77 tokens equals the float64 pass's argmax token for token; on a short NVFP4 prompt, 21 of
  21 positions take the float64 pass's top token (mean KL 0.0005).
- One NVFP4 position of 5,200 in a longer prompt flips. After "every segment gets 4," the float64 pass gives `000`
  (0.645), decode windows `000` (0.621), and the prompt path `096` (0.855). Moving the NVFP4 projections to the row
  kernels still flips it; moving every projection there does not. The flip comes from the shared bf16 tensor-unit
  GEMM on the attention projections, which matches MLX bit for bit and passes the contract at these shapes, at a
  position where the NVFP4 model is nearly torn between two answers. bf16 does not flip there. Leaving that position
  out, the prompt path's mean KL equals the decode path's.

## Measured

Served through the recipe harness (4 Nov 2026, the NVFP4 release, no other model loaded; speed and prompt runs on
build 588e7d2a2, the serving checks repeated on d26cd59ad with the same results and speeds), `--context 262144
--parallel 2`, greedy 256-token replies:

| streams | prose | code | structured | list | first token | a round |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.1 | 10.1 | 10.2 | 10.1 | 0.61-0.74 s | 99.2 ms |
| 2 | 18.8 | 18.6 | 19.1 | 18.8 | 0.93-1.10 s | 107.0 ms |

Tokens a second, all streams together. One stream reads the 53 GiB of weights each step; two read them once for both
rows. The same ruler after the serving checks gives the same speeds.

| cold prompt | 2,220 | 8,776 | 35,015 | 139,723 tokens |
|---|---:|---:|---:|---:|
| tokens a second | 569 | 605 | 587 | 434 |
| time to first token | 3.9 s | 14.5 s | 59.6 s | 322 s |

A fresh server, each prompt starting with words no earlier one did. With thinking requested (new prompts of the same
lengths), 488, 458, 503 and 410 tokens a second.
A multi-turn chat's third turn, resumed from 4,513 of its 4,597 kept tokens, starts in 0.67 s against 8.2 s cold,
with the same reply as on a restarted server.

- Structured output (json_object, json_schema, regex, choice, grammar) 18/18, drafted == plain 12/12; tool calls
  (auto, required, named, thinking, streamed) 5/5; seeded sampling 9/9 equal on repeat and undrafted.
- A 60-turn soak and 12 concurrent replies each equal to the request alone; streaming 4/4, multi-turn recall 8/8,
  stop strings 3/3, verbatim copy 2/2, max_tokens 3/3.
- Thinking 5/10 by the suite's rule, which wants reasoning in every reply: Ripple decides for itself whether to think
  (its template has no switch). Sent again, five replies thought and five answered directly, nine of ten answers right
  (the miss, a clock angle, answered directly); no think tags leaked into a reply.
- A passphrase a third of the way into a 120,041-token and a 258,039-token prompt (greedy, on the same server) comes
  back exactly; the cold prompts take 273 s and 790 s to the first token. The bf16 release found it too, at 120k and
  258k tokens.

## Not done

No quality suite was run. Contexts past 258,039 tokens were not run, though `--context` admits up to 1,048,576 (64.8
GiB of caches a stream). Ripple's think markers are spelled from several tokens, not one each, so a thinking budget
(which forces a close) and logprobs (which cut the reply at its close token) are refused by name, and a reply's
`reasoning_tokens` reads 0. The family has no draft head: drafts come only from the lane core's copies. No images:
Ripple has no vision tower. The bf16 release's correctness is measured above, but its serving numbers did not go
through the recipe harness. Only the M5's tensor units were checked; M1 to M4 Macs are refused at load, and smaller
RAM is untested. The publisher's DFlash2 drafter (`ricursiveAI/Ripple-65B-Preview-DFlash2`) is not served yet.

## Licence

Ripple-65B-Preview and its NVFP4 release are published under Apache-2.0 at the revisions measured here.
