# Kolibri-1

This page describes the native engine. `kolibri1` serves Aleph Alpha's Kolibri-1, a research checkpoint, on one
NVIDIA GPU through the lane host. It is a draft family: plain decoding and copy drafts from the prompt's own text are
in, and `--drafter` loads the published EAGLE-3 drafter (off unless asked).

## Checkpoint

Tested revision: `Aleph-Alpha/Kolibri-1` at `e52eb4627d11516b0c01de49210ab5a4e4061444` (32 shards, 78.9 GB). The
weights are block FP8: e4m3 codes in 128 x 128 blocks with fp32 `weight_scale_inv`, the routers, norms, embeddings and
head in bf16. Publisher sha256 of `model.safetensors.index.json`:
`c6769a34820c9057b404b2a5cae1ac24fd7b4a9ca08b84eb0a06e7b5bad3005e`.

```bash
tensorfold-native pull Aleph-Alpha/Kolibri-1@e52eb4627d11516b0c01de49210ab5a4e4061444
tensorfold-native serve Aleph-Alpha/Kolibri-1 --name kolibri-1 --context 32768 --parallel 8
```

The model has 50 layers, 2,560 wide, 48 query heads over 4 key heads of 128. Every fifth layer (4, 9, ..., 49) attends
to every key without positions; the other 40 attend to a 513-key window with RoPE. Each layer routes by sigmoid with an
expert bias to 6 of 384 experts, weighted by the unnormalised sigmoid, plus one ungated shared expert.

## What the family adds

It brings no kernels of its own. Every launch is a shared CUDA operator:

- `cuda/fp8.zig` and `qlinear` for the attention projections (block FP8, `fp8g` lane tiles);
- `cuda/experts.zig` for the routed and shared experts (`fp8g`, the shared expert stacked last at weight 1);
- `qk_rows` (QK norm, RoPE, the cache write), `norms`, `embed_rows`, `moe_glue` (sigmoid top-k with bias),
  `dense_rows` (router and head);
- `window_attention` for prompt rows of every layer (the full layers with no window) and for sliding decode rows;
- `tree_attention` for the full layers' decode rows over committed keys and each window's chain.

The family supplies its config check, the tensor map (116,303 tensors, none left over), the packing into those
layouts, the layer graph and the lane backend. One-row steps replay a CUDA graph a stream; wider windows and shared
rounds run eager. Both give a row the same bits.

## Serving

- Prompts run in passes of up to 8,192 rows, as the Python engine ran them, one a step, so other streams' rounds run
  between them (a full pass holds them for about 3.7 s). A row's bits do not depend on its chunk: 1,000-, 1,536- and
  2,048-row cuts of a 5,000- and a 20,000-token prompt give the 8,192-row passes' logits byte for byte.
- Concurrent streams share rounds of up to 64 rows. Each stream's caches are allocated at load, one per `--parallel`
  stream: at a 32k window 1.29 GiB a stream, at 64k 1.92 GiB. Each sliding layer's ring holds a pass and its window
  (8,768 keys).
- The prompt cache keeps a stream's state at any row: full layers' keys up to it and each sliding layer's last 512.
- Drafts: the lane core's copy drafts verify in windows of up to 16 rows. `--drafter DIR` adds the EAGLE-3 drafter
  (below); a request for the model's own draft head is refused by name, as it has none.
- Refused at load: other quantizations, `norm_topk_prob`, other layer kinds, a config with a vision tower, shapes the
  shared kernels do not take, `--vision`, `--precision` and a captured kernel set (`TENSORFOLD_CUDA_KERNELS`).

## Checks

```bash
zig build -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc
zig-out/bin/tf-cuda-test kolibri-reference MODEL IDS   # layer 0 and the head against float64 from the checkpoint
zig-out/bin/tf-cuda-test kolibri-layers MODEL IDS      # the 513-key window edge, layer 4's full reach
zig-out/bin/tf-cuda-test kolibri-full MODEL IDS 128    # steps, drafted, sampled, shared rounds, cuts, kept states
zig-out/bin/tf-cuda-test kolibri-greedy MODEL PROMPTS N OUT   # greedy ids and top-2 margins a step
zig-out/bin/tf-cuda-test kolibri-logits MODEL IDS OUT  # teacher-forced logits of 2,048 rows, for quality
```

`IDS` is a file of real-text token ids (20,000 or more for `kolibri-full`, 3,000 for `kolibri-layers`). Random ids give
flat logits and say nothing about quality.

The native rows are not byte-equal to the Python engine's: its glue was Triton, these are the shared CUDA operators,
and their fp32 roundings differ by an ulp here and there. Kolibri-1's top-6-of-384 routing turns such differences into
different experts on near-tied rows, so greedy replies part where two tokens are nearly tied. On 2,048 rows of real
text, teacher-forced: next-token top-1 agrees on 92.3% of rows, KL 0.06; NLL 3.574 here against 3.578 for the
Python engine, top-1 35.96% against 35.47%, mean entropy 2.354 against 2.358.

## Speed

GB10, greedy, served by `tensorfold-native` with copy drafts on:

| | tok/s |
| --- | ---: |
| One stream, code / chat (`bench_openai.py`, 256 tokens) | 51.9 / 49.8 |
| 4 streams, aggregate | 124.0 / 126.0 |
| 8 streams, aggregate | 175.5 / 183.1 |

Cold prompts (`bench_cache.py --require-cold --repeats 3`, `--prompt-cache-gib 0`, median of the cold requests):
2,048 tokens 0.97 s, 8,192 tokens 3.72 s, 32,768 tokens 16.8 s, 64,000 tokens 37.8 s to the first token.

## EAGLE-3 drafter

`--drafter DIR` loads `josch15366/Kolibri-1-EAGLE3-drafter` (revision `6430e5dc74`): one decoder layer over Kolibri's
layers 45, 48 and 50, a 32,768-token draft vocabulary, bf16. It runs on the shared CUDA operators (no kernels of its
own) through the lane core's drafter interface, two drafts a round. Every draft is verified by the target, so replies
are the same tokens as without it, greedy and seeded, at any stream count. `--drafter-bits` takes 0 or 4 (the
default); both serve the release as published.

Refused at load, by name, before any request is queued: a release of another architecture (the block drafter
`josch15366/Kolibri-1-block-drafter` is not served on CUDA), a release for another target, a draft vocabulary or
tensor shape that does not match the release's config, a stream opened before the drafter, and a CUDA memory budget
that leaves less than the drafter's weights, scratch and caches need (computed from its config before anything is
allocated; about 0.44 GiB with one stream).

GB10, greedy, 24 prompts of 256 tokens, A-B-B-A servers, the same replies in all four:

| streams | copy drafts, tok/s | with `--drafter`, tok/s | tokens a round |
| --- | ---: | ---: | ---: |
| 1 | 51.5 | 66.5 (+29%) | 1.08 -> 1.84 |
| 4 | 120.4 | 127.8 (+6%) | 1.08 -> 2.05 |
| 8 | 157.6 | 155.7 (-1%) | 1.08 -> 2.04 |

At 8 streams the drafter is slower: the target verifies three times the rows, and the depth rule does not yet cut the
drafts there.
