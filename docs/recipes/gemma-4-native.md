# Gemma 4 31B, native Metal

This recipe covers text generation with
[`mlx-community/gemma-4-31B-it-4bit`](https://huggingface.co/mlx-community/gemma-4-31B-it-4bit/tree/696d436c404745a59f30e4939a658162b0a9e57f),
revision `696d436c404745a59f30e4939a658162b0a9e57f`, the dense 31B in MLX affine 4-bit, groups of 64.
Build from `main` with Zig **0.17.0** and Xcode's Metal toolchain, as described in
[the native preview](../../ZIG-PREVIEW.md#build).

```bash
hf download mlx-community/gemma-4-31B-it-4bit --revision 696d436c404745a59f30e4939a658162b0a9e57f \
  --local-dir "$HOME/models/gemma-4-31b"
zig build native -Dcpu=apple_m1
zig-out/native/bin/tensorfold-native serve "$HOME/models/gemma-4-31b" --port 8888 --parallel 4
```

The loader admits the dense 31B's geometry only: 60 layers, five sliding-window layers (window 1,024, head dim 256)
to each global one (head dim 512, keys reused as values, proportional RoPE), GeGLU, the tied 262,144-token head with
its logit soft-cap, uniform affine 4-bit in groups of 64. It refuses the 26B-A4B MoE, the 12B (`gemma4_unified`),
per-layer inputs, shared KV layers and per-module widths, each by name. The vision tower is not loaded.

Every linear, the tied head included, runs on the core lane projection (`zig/src/core/lane_projection.zig`) on M5
tensor units; the rest of a layer is `zig/kernels/metal/gemma4/glue.metal`. The glue ports the Python reference's
decode kernels (gilby's `gemma4-dense` branch, #360, built on cshintov's Gemma kernels from #10) with their rounding
points, and MLX's own ops where that reference calls them: the looped RMS norm, the 4-bit embedding times its bf16
scale, `geglu` and the soft-cap as bf16 ops, and the global layers' RoPE frequencies from MLX's precise `pow`.
Each output equals the reference's bit for bit at 1, 2 and 16 rows (`tf-gemma4-kernels`, `tf-gemma4-glue`, on
fixtures from `tools/zig/gemma4_dump.py`).

A stream keeps a ring of window + 128 slots for each sliding layer and a buffer for each global one; rolling back is
moving the cache's length. A forward takes up to 16 rows, from one stream or several, and every row keeps its bits.
Prompts run in 16-row decode windows. Drafts are context copies; the checkpoint has no draft head.

The context defaults to 32,768 tokens; `--context` raises it to the checkpoint's 262,144, at the cost of attention
scratch that grows with it. There is no cross-request prompt reuse yet.

| Checkpoint / platform | Qualification |
| --- | --- |
| The pinned 31B 4-bit, M5 Ultra, macOS 27 | Greedy decode, prompts, shared rounds, copy drafts and the server, below. |
| M1 to M4 | Refused: the lane projection needs M5 tensor units. |
| Other Gemma 4 checkpoints, widths, vision | Outside this recipe. |

## Verification

The oracle is gilby's `gemma4-dense` Python engine (`a6ff795e`, MLX 0.32.3, mlx-lm 0.32.0) with lane kernels.

```bash
python tools/zig/gemma4_gate.py oracle --model M --out O --tokens 128 --prompt decode   # in the oracle's venv
zig-out/bin/tf-gemma4-run M O N 128 16 && python tools/zig/gemma4_gate.py check --oracle O --native N
zig-out/bin/tf-gemma4-lanes M O
python tools/zig/gemma4_gate.py served --oracle O --url http://127.0.0.1:8888
```

- Four chat prompts, 128 greedy tokens: every token and every top-two logit equals the oracle's, with prompts fed one
  row at a time or in windows of 16.
- 8k and 32k prompts against the oracle's mlx-lm prompt forward: the first token is equal at both, and 49 tokens at
  32k. At 8k the two part at token 32 on a tie within two bf16 steps; there the native run equals the oracle's own
  decode-path prompt, tokens and last-row logits, and that path parts from mlx-lm at the same token.
- The lane core: plain and copy-drafted streams, and 4 and 8 streams in shared rounds, equal the oracle's tokens;
  on copy-heavy prompts (11.1 and 3.3 tokens a round) drafted replies equal the engine's plain ones.
- Served on an M5 Ultra: greedy token SHAs equal the oracle's alone, with drafts and four at once; thinking replies
  keep their reasoning apart, and tool calls parse.

Decode is about 33 tokens a second for one stream and 147 for eight together; prompts run at 240-340 tokens a second.
