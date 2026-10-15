# Gemma 4 31B, native Metal

This recipe covers text generation with
[`mlx-community/gemma-4-31B-it-4bit`](https://huggingface.co/mlx-community/gemma-4-31B-it-4bit/tree/696d436c404745a59f30e4939a658162b0a9e57f),
revision `696d436c404745a59f30e4939a658162b0a9e57f`, the dense 31B in MLX affine 4-bit, groups of 64.
Build from `main` with Zig **0.17.0** and Xcode's Metal toolchain, as described in
[the native preview](../../ZIG-PREVIEW.md#build).

```bash
hf download mlx-community/gemma-4-31B-it-4bit --revision 696d436c404745a59f30e4939a658162b0a9e57f \
  --local-dir "$HOME/models/gemma-4-31b"
hf download z-lab/gemma-4-31B-it-DFlash --revision eabd648301ce28583cc14757912e5e0f84e152e1 \
  --local-dir "$HOME/models/gemma-4-31b-dflash"
zig build native -Dcpu=apple_m1
zig-out/native/bin/tensorfold-native serve "$HOME/models/gemma-4-31b" --port 8888 --parallel 4 \
  --drafter "$HOME/models/gemma-4-31b-dflash"
```

Without `--drafter` the model decodes plain, at about 43 tokens a second for one stream. With z-lab's DFlash drafter
a stream decodes at 150-190 tokens a second (6-7 accepted tokens a round on chat and code, 10 on copying replies,
2 on open-ended prose), with the same tokens.

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

A stream keeps a ring of window + 512 slots for each sliding layer and a buffer for each global one; rolling back is
moving the cache's length. A decode forward takes up to 16 rows of one stream and 128 of several, and every row keeps
its bits. Prompts run in chunks of 512 rows on `core/affine_mm` and a tensor-unit attention kernel, at about 1,180
tokens a second at 8k and 1,000 at 32k.

The drafter is z-lab's DFlash (`zig/src/families/gemma4/dflash.zig`): five Qwen3-style layers reading six of the
target's layer outputs (1, 12, 23, 35, 46, 57) through `fc` and a norm, drafting a block of the pending token and up
to 15 mask tokens with the target's own embedding and tied head. Its bf16 weights are quantized at load to 4-bit
groups of 64 by MLX's affine rule and run on the same lane projection; the context keys and values of each stream
live in a 2,047-slot ring for the four sliding layers and a growing buffer for the full one, as in z-lab's MLX model.
It sits in front of the engine as the lane core's `Drafted` wrapper (chains, no trees): the target hands it each
prompt chunk's and each verify's tapped rows, and the depth rule prices its drafts against the forward's cost by rows
timed at load. A block costs about 3.4 ms at any depth and an absorb about 2.5 ms.

The context defaults to 32,768 tokens; `--context` raises it to the checkpoint's 262,144, at the cost of attention
scratch that grows with it. There is no cross-request prompt reuse yet.

| Checkpoint / platform | Qualification |
| --- | --- |
| The pinned 31B 4-bit, M5 Ultra, macOS 27 | Greedy decode, prompts, shared rounds, copy and DFlash drafts and the server, below. |
| `z-lab/gemma-4-31B-it-DFlash` `eabd6483` | The one drafter admitted: DFlash v1, block 16, six taps, head dim 128. |
| M1 to M4 | Refused: the lane projection needs M5 tensor units. |
| Other Gemma 4 checkpoints, widths, vision | Outside this recipe. |

## Verification

The oracle is gilby's `gemma4-dense` Python engine (`a6ff795e`, MLX 0.32.3, mlx-lm 0.32.0) with lane kernels.

```bash
python tools/zig/gemma4_gate.py oracle --model M --out O --tokens 128 --prompt decode   # in the oracle's venv
zig-out/bin/tf-gemma4-run M O N 128 16 && python tools/zig/gemma4_gate.py check --oracle O --native N
zig-out/bin/tf-gemma4-lanes M O && GEMMA_DRAFTER=D zig-out/bin/tf-gemma4-lanes M O
zig-out/bin/tf-gemma4-run M O R 128 512 && python tools/zig/gemma4_gate.py served --oracle O --native R
```

- Four chat prompts, 128 greedy tokens, prompts fed one row at a time or in windows of 16: three equal the oracle
  token for token; the code prompt parts at its third token, where the oracle's top two logits are one bf16 step apart
  and the native run ties them (its decode attention sums 256-key chunks in its own order). Every row keeps its bits
  at any window width and beside other streams' rows.
- 8k and 32k prompts against the oracle's mlx-lm prompt forward: the first token is equal at both, and 49 tokens at
  32k. At 8k the two part at token 32 on a tie within two bf16 steps; there the native run equals the oracle's own
  decode-path prompt, tokens and last-row logits, and that path parts from mlx-lm at the same token.
- The lane core: copy-drafted and DFlash-drafted streams, alone and 4 and 8 in shared rounds, equal the engine's
  plain replies token for token, on the four prompts and on two copy-heavy ones.
- Served on an M5 Ultra with the drafter: greedy token SHAs equal the engine's receipts alone, with drafts off and on,
  and four at once; thinking replies keep their reasoning apart, and tool calls parse.

With the drafter, one stream decodes at 150-190 tokens a second (6.4, 6.4, 2.1 and 6.7 tokens a round on the four
prompts; 10.2 and 7.8 on copying replies) and eight streams at 180-195 together. Plain decode is 43 tokens a second
for one stream. The drafts match z-lab's own MLX drafter's acceptance; shared rounds draft shallow because the lane
projection reads its weights once per 16 rows, so a 128-row forward costs 128 ms against 25 for 16.
