# Kolibri-1

This page describes the native engine. `kolibri1` serves Aleph Alpha's Kolibri-1, a research checkpoint, on one
NVIDIA GPU through the lane host. It is a draft family: plain decoding and copy drafts from the prompt's own text are
in; the EAGLE-3 drafter is the next step and is not loaded here.

## Checkpoint

Tested revision: `Aleph-Alpha/Kolibri-1` at `e52eb4627d11516b0c01de49210ab5a4e4061444` (32 shards, 74 GB). The
weights are block FP8: e4m3 codes in 128 x 128 blocks with fp32 `weight_scale_inv`, the routers, norms, embeddings and
head in bf16. Publisher sha256 of `model.safetensors.index.json`:
`c6769a34820c9057b404b2a5cae1ac24fd7b4a9ca08b84eb0a06e7b5bad3005e`.

```bash
tensorfold-native serve /path/to/Kolibri-1 --name kolibri-1 --context 32768 --parallel 8
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
- Drafts: the lane core's copy drafts verify in windows of up to 16 rows. There is no draft head yet, so a request
  for head drafts and `--drafter` are refused by name.
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

## Next

- The EAGLE-3 drafter through the lane core's drafter interface.
