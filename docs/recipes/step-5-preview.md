# Step-5-Preview

This recipe describes the native Zig engine on one 256 GB Mac. Step-5-Preview (`stepfun-ai/Step-5-Preview`) has 92
layers of sliding and full GQA attention, a sparse CSA indexer on the full layers, 352 routed experts (8 a token),
three MTP layers and a perception-encoder image tower. StepFun ships it in FP8, which one Mac cannot hold, so this
page converts it to MLX affine weights first.

## Converting

`tools/step5_fp8_to_mlx.py` streams the FP8 checkpoint into an MLX affine one. The routed experts become 2-bit in
groups of 128, each group's range searched and refit by least squares; every other linear weight becomes 8-bit in
groups of 64, and norms, the router and the indexer's weights keep their dtype. It needs mlx and a few GiB of RAM.

```bash
python tools/step5_fp8_to_mlx.py --src ~/models/step5-preview-fp8 --out ~/models/Step-5-Preview-MLX-2bit --mtp --vision
```

The text model is 166.9 GiB; `--mtp` adds 2.5 GiB and `--vision` 3.7 GiB (bf16). `--only-mtp` and `--only-vision`
add them to a checkpoint already converted. `--layers N` writes a bring-up slice of the first N layers.

## Serving

```bash
tensorfold-native serve ~/models/Step-5-Preview-MLX-2bit --port 8888 --vision
```

Structured output (`response_format`, guided JSON, regex, choice and grammars) needs `libtfgrammar` beside the binary:
release archives ship it; from source, `zig build grammar -Dxgrammar=DIR` with an xgrammar v0.2.8 checkout.

- **Memory.** Weights, scratch and every stream's caches stay under 75% of RAM, below the point where macOS stalls;
  `STEP5_LOAD_LIMIT_GB` sets another limit. With MTP and vision, 15.8 GiB is left for caches on a 256 GB Mac: up to
  227,025 tokens of context by default, or `--context` for less. The model supports 1M tokens; a Mac's caches reach
  what the limit leaves. A 212k-token prompt ran with free memory never below 19%.
- **Prompts** run in 2,048-row chunks on the tensor units: 868, 863, 840 and 800 tokens a second at 4k, 8k, 16k and
  32k tokens. A 129k-token prompt takes about 200 s, a 212k-token one about 376 s.
- **MTP drafts** (`--mtp` checkpoints): three chained drafts a round, verified in one window. Replies are the same as
  undrafted ones. Greedy replies of 200-400 tokens, timed with their prompt: 41.2 tokens a second on code (25.8
  undrafted), 36.6 on prose (27.0), 48.8 on math (26.9).
- **Prompt cache.** Kept states hold the full layers' keys and values to the mark, the last 512 positions of each
  sliding layer and the indexer's open region, copied on the GPU. They live in what the streams' caches leave of the
  budget and give way when a stream needs the room; `--prompt-cache-gib` caps them and `0` turns them off. Replies are
  the same with it on and off. An 11.6k-token prompt sent again took 1.8 s instead of 15.3 s; later turns of the
  chat resume there. A kept state cannot sit beside its own stream past about 110k tokens, so longer prompts are not
  kept. States sit on 2,048-token boundaries and chat-history cuts, so a new question on the same image in a new
  chat reads the image again.
- **Images** (`--vision`) follow vLLM's Step3-VL processor as of vLLM `dd944845777b`: 504-pixel crops and a
  728-pixel overview, each through the encoder on the tensor units. A 1200x800 image is 688 tokens; its first reply
  took 3.8 s with an 80-token answer.
- **Streams.** `--parallel N` serves N streams at once, each with its own caches under the same budget.
- **Idle.** Weights and scratch sit in a residency set that the idle keepalive touches with a tiny kernel, so the
  first request after a pause is as fast as the rest (0.17 s for a one-token reply).

Load takes about 40 s, including a warm-up pass so the first request does not pay the GPU's first touch.

## Checking

`tools/zig/step5_ref.py` is an MLX reference of the text model (`tools/zig/step5_check_ref.py` checks it against
StepFun's torch code), and `tools/zig/step5_vision_ref.py` runs StepFun's image tower in torch. Served numbers and the
quality suite's results for this checkpoint are in
[recipe-harness](https://github.com/BobClawblaw/recipe-harness) (target `step5-m5`).

On the 4-layer slice, a 209,814-token prompt gives the reference's top 5 next tokens in its order, logprobs within
0.02. On the full model, long-context retrieval at 212k tokens (one record's code out of 10,500) answered 2 of 3
questions right and at 129k 2 of 2; the miss, an early record, is the 2-bit model's, not the engine's.

The recipe-harness runs on this build: structured output 18/18, seeded sampling 9/9, a resumed multi-turn chat
identical to a cold one (first token 0.35 s against 3.28 s), 60/60 soak turns, tools 58/60, JSON 30/30, GSM8K 49/50,
IFEval 41/50; one stream decodes 39-51 tokens a second (prose to code) with the first token in 0.24-0.28 s.
