# Ripple-65B-Preview, CUDA

This recipe covers text generation with ricursiveAI's Ripple-65B-Preview (model_type `ricursive`; see
[the Metal recipe](ripple-65b-metal.md) for the model) on the CUDA engine. One release is served as published,
read in place from its shards with nothing converted:

- [`ricursiveAI/Ripple-65B-Preview-NVFP4`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4) (revision
  `99eea4477a001a41b86914e41feb4b2ed86105e5`, 53 GiB): ModelOpt NVFP4 MLPs, attention, embeddings and head in bf16.

The bf16 release is refused by name (`RicursiveCudaNeedsNvfp4`): at 121 GiB it does not fit a GB10, and the CUDA
family reads only the NVFP4 MLPs. The publisher's DFlash2 drafter,
[`ricursiveAI/Ripple-65B-Preview-DFlash2`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2) (revision
`35609b3be3dd08a822b3ba1257c1ea5dd6f87632`, 7.5 GiB, bf16), drafts with `--drafter`; a drafted reply equals the plain
one.

It was brought up on a DGX Spark (GB10, sm_121, 121.7 GiB shared with the host, CUDA 13). The kernels build for
sm_80, 86, 89, 120 and 121 and their launches fit each, but only GB10 was run.

## Build and serve

```sh
hf download ricursiveAI/Ripple-65B-Preview-NVFP4 --revision 99eea4477a001a41b86914e41feb4b2ed86105e5 \
  --local-dir models/ripple-65b-nvfp4
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
zig-out/native/bin/tensorfold-native serve models/ripple-65b-nvfp4 --name ripple --port 8888 --context 32768
```

At `--context 32768` admission gives eight streams of 2.78 GiB and a 16.0 GiB prompt cache, after a 12.2 GiB
reserve. A stream's caches: 64 of the 80 layers slide over 1,024 keys and keep a 3,072-row ring each; every fifth
layer is global and keeps every key, in 512-row pages.

With the drafter (`hf download ricursiveAI/Ripple-65B-Preview-DFlash2 --revision
35609b3be3dd08a822b3ba1257c1ea5dd6f87632 --local-dir models/ripple-65b-dflash2`), add `--drafter
models/ripple-65b-dflash2`. It keeps 10.09 GiB for its weights, scratch and eight streams' rings; at `--context 32768`
admission still gives eight streams (3.01 GiB each), and the prompt cache drops to 11.2 GiB.

Thinking follows the chat template, as on Metal: Ripple writes `<think>` itself when it thinks and answers bare
otherwise, and the server files a reply without the block as answer. Its think markers are spelled from several
tokens, so a thinking budget and logprobs are refused by name.

## What the family is

`zig/src/families/ricursive/cuda*.zig` and `zig/kernels/cuda/ricursive*.cu`, with the same layer math as Metal and
the release's `modeling_ricursive.py` as the reference:

- **Projections.** NVFP4 MLPs on the `.nvfp4` W4A16 lane (nibbles and E4M3 scales repacked at load, the fp32 global
  per tensor; `input_scale` is not used); attention projections, embeddings and head in bf16 through `qmmf`. Q, K and
  the gate share one projection; V reads the value-residual blend.
- **Glue** (`ricursive.cu`): the embedding scale in fp32, the add-and-norm with the value-residual lerp, the gate
  (bf16 sigmoid before the multiply) and the clamped SwiGLU. `ricursive_prep.cu` does the Q/K norm, RoPE (bf16 cos
  and sin tables; none on the global layers) and the K/V write into a ring or pages in one kernel.
- **Attention.** Sliding decode on `tf_window_decode_tc` over each layer's ring; global decode on the paged
  `tree_attention` kernels; prompt chunks on the window kernel (paged for the global layers) with two query heads a
  block, for Ripple's group of 10.
- **Serving.** The lane host's shared rounds as CUDA graphs, greedy or sampled draws, copy drafts, verify windows,
  kept states and the prompt cache.
- **DFlash2 drafter** (`cuda_dflash2*.zig`, `ricursive_dflash2.cu`, `ricursive_dflash2_attn.cu`), in the Metal
  drafter's op order and bf16 rounding points. The target writes the residual after layers 1, 20, 39, 58 and 77 into a
  2,064-row tap ring a stream, inside the round graphs. A round runs the 16-row block through the five layers for
  every stream in one launch a step: the 4-tap convolution in Metal's rounding order; draft attention that reads each
  of the 8 KV heads once for its 10 query heads and all 16 rows; the target's head; a top-32 a row; then the selector
  walk on the host. Chain verify only (one draft a stream), as on Metal.

**Activations stay bf16**, as on Metal: the engine serves the NVFP4 weights with bf16 activations and fp32 sums, not
the publisher's W4A4 kernels.

Every kernel is registered with the kernel contract, with planted controls that each must turn its gate red. On
GB10 (nvcc 13.3, sm_121):

```
SUMMARY ricursive: 22 kernels, 22 pass gates 1-4, 0 fail; 27 controls: 27 red as planted, 0 missed
SUMMARY ricursive dflash2: 10 kernels, 10 pass gates 1-4, 0 fail; 19 controls: 19 red as planted, 0 missed
SUMMARY ricursive dflash2 attention: 3 kernels, 3 pass gates 1-4, 0 fail; 6 controls: 6 red as planted, 0 missed
```

## Correctness (GB10)

`tools/zig/ricursive_reference.py` (from the Metal PR) on the Spark's CPU, teacher-forced over the same 2,586-token
chat as the Metal recipe, against the NVFP4 release's own passes:

| engine logits | against | mean KL | top-1 |
|---|---|---:|---:|
| 2,048-row prompt chunks | the float64 pass | 0.00069 | 98.72% |
| 2,048-row prompt chunks | the `--bf16-points` pass | 0.00078 | 98.84% |
| one-row decode | the float64 pass | 0.00076 | 98.92% |
| one-row decode | the `--bf16-points` pass | 0.00080 | 99.07% |
| `--bf16-points` pass (no engine) | the float64 pass | 0.00073 | 98.96% |

The engine sits as close to the float64 pass as the bf16-rounded reference does (Metal: 0.00069 prompt, 0.00107
decode). The float64 pass equals the release's own `modeling_ricursive.py` run in float64 on the same weights
(`--release-f64`: top-1 100%, largest KL 0.0005). Every engine miss is a near-tie: the reference leads the engine's
pick by at most 0.25 nats.

- Prompt chunks of 2,048, 1,000 and 333 rows give bit-identical logits on all 2,586 rows; no non-finite logit.
- A greedy 128-token reply equals the float64 pass's argmax token for token.
- With graphs on and with `--eager`, every dump and reply hash equals the first run's bit for bit.
- Served replies are bit-identical alone, beside seven other streams and one at a time (greedy and sampled). A
  conversation sent again, reusing 79 to 81 cached tokens, equals its first reply, and 60 repeats of a reply equal the
  first. Seeded replies equal their repeats, and streamed text equals the reply sent whole.
- With `--drafter`, every reply equals the plain server's at the same seed: 22 of 22, greedy and sampled, at 1, 2
  and 8 streams. Each drafted reply also equals itself alone and with `"draft": false`, and cold 2,048- and
  8,192-token prompts give the plain reply. With the drafter's code built in, the plain engine's logits are unchanged
  bit for bit on all 2,586 rows.

## Measured (GB10, NVFP4 release, `--context 32768`, no other model busy)

One stream decodes at 248.3 ms a token (4.03 tok/s); each step reads all 53 GiB of weights. Prompts run
at 518-520 tok/s at 2,586 rows. Greedy 256-token essays, thinking off, all streams together:

| streams | tok/s |
|---:|---:|
| 1 | 4.03 |
| 4 | 15.54 |
| 8 | 29.93 |

With `--drafter`, 256 tokens, median of 3 (code is a raw text completion, prose a chat reply): one stream from
`tools/bench_openai.py`, eight streams together from `tools/bench_concurrent.py --levels 8` (all eight streams' tokens):

| reply | 1 stream, plain | 1 stream, drafted | 8 streams, plain | 8 streams, drafted |
|---|---:|---:|---:|---:|
| code, greedy | 5.2 | 26.1 | 37.5 | 49.5 |
| code, sampled (T=1.0) | 4.5 | 18.4 | 30.4 | 42.8 |
| prose, greedy | 4.4 | 10.0 | 32.6 | 40.3 |
| prose, sampled (T=1.0) | 4.0 | 11.2 | 30.4 | 35.2 |

At `--context 131072` admission gives four streams of 8.78 GiB and a 7.1 GiB prompt cache. Cold prompts there
(`tools/bench_cache.py --require-cold`), time to first token: 3.76 s at 2,048 tokens (545 tok/s), 15.6 s at 8,192
(524), 68.4 s at 32,768 (479) and 150.7 s at 65,536 (435). Each reply equals its repeat and itself beside a second.
A 114,528-token prompt with a random four-word passphrase a third of the way in: the reply names it, first token
after 300.5 s (381 tok/s).

## Not done

Structured output (`response_format`) and forced tool calls (`tool_choice` `"required"` or a named function) are not
supported on CUDA yet: the server refuses them, because the CUDA family has no masked sampling. Tool calls with
`tool_choice` `"auto"` work. No quality suite was run. Contexts past 131,072 tokens were not served on CUDA. Only
GB10 was run. Draft trees (several drafts a stream) are not done, and the drafter's kernel speed is not tuned.

## Licence

Ripple-65B-Preview and its NVFP4 release are published under Apache-2.0 at the revisions measured here.
