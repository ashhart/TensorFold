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

- **Memory.** Weights, scratch and every stream's caches stay under 75% of RAM, below the point where macOS stalls;
  `STEP5_LOAD_LIMIT_GB` sets another limit. With MTP and vision, 16.8 GiB is left for caches on a 256 GB Mac: about
  240k tokens of context by default, or `--context` for less. The model supports 1M tokens; a Mac's caches reach
  what the limit leaves.
- **Prompts** run in 512-row chunks on the tensor units: 690, 668, 647 and 622 tokens a second at 4k, 8k, 16k and
  32k tokens. A 131k-token retrieval question was answered correctly in 264 s.
- **MTP drafts** (`--mtp` checkpoints): three chained drafts a round, verified in one window. Replies are the same as
  undrafted ones. Greedy decode measured 44.6 tokens a second on code (1.74x), 34.8 on prose and 46.0 on math.
- **Prompt cache.** Kept states hold the full layers' keys and values to the mark and every ring cache, copied on
  the GPU. They live in what the streams' caches leave of the budget and give way when a stream needs the room;
  `--prompt-cache-gib` caps them and `0` turns them off. An 11.6k-token prompt sent again took 1.8 s instead of
  21.2 s, with the same reply.
- **Images** (`--vision`) follow vLLM's Step3-VL processor: 504-pixel crops and a 728-pixel overview, each through the
  encoder on the tensor units. A 1200x800 image is 688 tokens, encoded in about 2 s.
- **Streams.** `--parallel N` serves N streams at once, each with its own caches under the same budget.

Load takes about 37 s, including a warm-up pass so the first request does not pay the GPU's first touch.

## Checking

`tools/zig/step5_ref.py` is an MLX reference of the text model (`tools/zig/step5_check_ref.py` checks it against
StepFun's torch code), and `tools/zig/step5_vision_ref.py` runs StepFun's image tower in torch. Served numbers and the
quality suite's results for this checkpoint are in
[recipe-harness](https://github.com/BobClawblaw/recipe-harness) (target `step5-m5`).
