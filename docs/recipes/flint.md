# Recursive Flint

This page describes the native Zig engine serving Recursive's Flint (`RecursiveSI/flint`) on a Mac. Flint is a
40B-A2.5B mixture of experts: 49 layers of GQA (32 query heads, 4 key heads, head size 128), 35 of them sliding
(a 513-token window, with RoPE) and 14 full (no positional encoding), every head with a learned attention sink and
per-head q/k norms; two dense SwiGLU layers, then 47 layers of 256 routed experts (6 a token, sigmoid scores
renormalized) beside an always-on shared expert. Text in, text out, with reasoning and tool calls.

## Weights

The engine serves Recursive's own MLX export, `RecursiveSI/flint-int4`: affine 4-bit groups of 64 for the
projections, the experts and the head, 8-bit for the embedding (22.6 GB on disk, 21.4 GiB resident).

Licenses: Recursive publishes Flint's weights (`RecursiveSI/flint` and `RecursiveSI/flint-int4`) under Apache-2.0.
The kernels round where Recursive's MLX engine does (`rec_mlx` in `recursive-oss/rec`, release v0.1.0 at e53663b2,
Apache-2.0).

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
- **Memory.** Weights, scratch and every stream's caches stay under 75% of RAM, or `FLINT_LOAD_LIMIT_GB` (GB, at most
  the Mac's RAM, capped at the GPU's working set); the prompt cache takes what the streams leave, up to
  `--prompt-cache-gib`. On a Mac that serves other models, set both.
- **Streams.** `--parallel N` streams share each decode forward, each reply equal to its run alone.
- **Prompt cache.** Kept states hold the full layers' keys and values and the sliding layers' last window; a resumed
  prompt gives the same reply as the same prompt sent fresh.
- **Reasoning.** Flint always reasons: its chat template has no switch, so the server advertises `thinking_switch:
  false` and refuses a request's `enable_thinking: false` with a 400 (thinking off is not supported by this chat
  template); leave the flag out. Give hard requests a large `max_tokens`; the card suggests 32,768.
- **Tools.** The template's `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>` calls are parsed
  as GLM-style tool calls, and required or named tool choices are constrained by the grammar.
- **Drafts.** Copy drafts apply without an external head. Add `--drafter ~/models/Flint-DFlash2` for
  `incoai/Flint-DFlash2` at `39b0a7f7a1` (Apache-2.0). Its published BF16 tensors use the shared DFlash2 runtime;
  `--drafter-bits 4` prepares resident 4-bit draft projections, and `--drafter-bits 0` keeps BF16 draft projections.
  Target weights and verification stay unchanged. This head requires an M5 and `--parallel` from 1 to 4;
  other chips and larger stream counts are refused by name.

## Checks

`tf-flint-run` runs the model outside the server: greedy tokens, every position's logits in 16-row windows, a
prompt's last logits, or forced steps along given tokens, for comparison with a reference forward; `exact` compares
one-row steps with 16-row windows bit for bit; `bench` times decode steps.

Only the M5's tensor units were checked; M1-M4 are untested.

## External head receipt

An M5 Ultra with 256 GiB RAM on macOS 27.0.1 checked the published INT4 target and DFlash2 head above.
The server used context 8192, parallel capacity 4 and no prompt cache for cold measurements.
Requests used 128 output tokens, seed 1234, top-k 20, top-p 0.95, and Flint's normal reasoning template.
Each stream count used the first one, two or four prompts: sky colour, Python prime factors, a bicycle explanation,
and a small JSON object. The same binary ran without and with the head, with copies enabled on both.
A warm-up process was discarded, then no-head/head/head/no-head processes alternated.
Each table rate is total output tokens divided by the synchronized group's wall time, median of two processes.

| Temperature | Streams | No head tok/s | DFlash2 tok/s | No head tokens/round | DFlash2 tokens/round |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 174.4 | 231.7 | 1.01 | 2.94 |
| 0 | 2 | 220.2 | 293.2 | 1.04 | 2.78 |
| 0 | 4 | 304.2 | 338.1 | 1.04 | 2.61 |
| 0.7 | 1 | 166.0 | 191.1 | 1.01 | 2.56 |
| 0.7 | 2 | 208.5 | 259.2 | 1.02 | 2.61 |
| 0.7 | 4 | 278.6 | 273.8 | 1.03 | 2.44 |

Four sampled streams took 1.8% longer with the head, with separated process ranges.
The head is optional; these prompts do not establish a speedup on every workload.
Ready physical footprint was 22.32 GiB without the head and 22.72-22.73 GiB with it;
peak footprint during the measured processes was 23.04-23.10 and 23.97 GiB respectively.

All 110 cold replies in the balanced matrix kept complete token arrays and messages equal to solo plain execution.
Additional checks covered the new binary without a head, first-token and reply logprobs, full-vocabulary sampling,
BF16-head drafting, and a chat restored at 2402 of 2428 prompt tokens versus fresh execution.
The new tap kernel passed its exceptional-value, repeat/guard and shape gates, with its planted control red,
and its M1/M2 offline launch compiles. The head itself was exercised only on M5.
