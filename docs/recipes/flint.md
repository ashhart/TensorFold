# Recursive Flint

This page describes the native Zig engine serving Recursive's Flint (`RecursiveSI/flint`) on a Mac. Flint is a
40B-A2.5B mixture of experts: 49 layers of GQA (32 query heads, 4 key heads, head size 128), 35 of them sliding
(a 513-token window, with RoPE) and 14 full (no positional encoding), every head with a learned attention sink and
per-head q/k norms; two dense SwiGLU layers, then 47 layers of 256 routed experts (6 a token, sigmoid scores
renormalized) beside an always-on shared expert. Text in, text out, with reasoning and tool calls.

## Weights

The engine serves Recursive's own MLX export, `RecursiveSI/flint-int4`: affine 4-bit groups of 64 for the
projections, the experts and the head, 8-bit for the embedding (22.6 GB on disk, 21.4 GiB resident).

```bash
hf download RecursiveSI/flint-int4 --revision 3b93f9053b530488405ae9fca51d4900d6ce21f0 --local-dir ~/models/flint-int4
```

## Serving

```bash
FLINT_LOAD_LIMIT_GB=48 tensorfold-native serve ~/models/flint-int4 --port 8888 --parallel 4 --prompt-cache-gib 16 \
  --temperature 1.0 --top-p 0.95 --top-k 20
```

The sampling flags are the model card's defaults for requests that set none. Structured output and required tool
calls need `libtfgrammar` beside the binary (release archives ship it; from source, `zig build grammar
-Dxgrammar=DIR` with an xgrammar v0.2.8 checkout).

- **Context.** 262,144 tokens, the checkpoint's window. Sliding layers keep a ring of their window and one prompt
  chunk; the 14 full layers keep every position, 28 KiB a token (7 GiB a stream at the full window).
- **Memory.** Weights, scratch and every stream's caches stay under 75% of RAM, or `FLINT_LOAD_LIMIT_GB`; the prompt
  cache takes what the streams leave, up to `--prompt-cache-gib`. On a Mac that serves other models, set both.
- **Streams.** `--parallel N` streams share each decode forward, each reply equal to its run alone.
- **Prompt cache.** Kept states hold the full layers' keys and values and the sliding layers' last window; a resumed
  prompt gives the same reply as the same prompt sent fresh.
- **Reasoning.** Flint always reasons: its chat template has no switch, so `enable_thinking: false` does not turn it
  off, and the reasoning then reaches `content`. Give hard requests a large `max_tokens`; the card suggests 32,768.
- **Tools.** The template's `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>` calls are parsed
  as GLM-style tool calls, and required or named tool choices are constrained by the grammar.
- **Drafts.** The checkpoint has no draft head; the lane engine's copy drafts apply.

## Checks

`tf-flint-run` runs the model outside the server: greedy tokens, every position's logits in 16-row windows, a
prompt's last logits, or forced steps along given tokens, for comparison with a reference forward; `bench` times
decode steps. `FLINT_LAYERS=N` runs the first N layers only.

Only the M5's tensor units were checked; M1-M4 are untested.
