# Gemma 4 12B, native Metal

This recipe covers text generation with Google's
[`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it/tree/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7),
revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` (`gemma4_unified`), its language model converted to MLX affine
4-bit in groups of 64, with Google's MTP assistant
[`google/gemma-4-12B-it-assistant`](https://huggingface.co/google/gemma-4-12B-it-assistant/tree/46d4c6f13f0ac0ad827b915669b8df9b81c64c51)
(revision `46d4c6f13f0ac0ad827b915669b8df9b81c64c51`) as the drafter. It runs on the native Gemma 4 family that serves
[the 31B](gemma-4-native.md), on M5 tensor-unit GPUs. Build with Zig **0.17.0** and Xcode's Metal toolchain, as
described in [the native preview](../../ZIG-PREVIEW.md#build).

```bash
hf download google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 --local-dir "$HOME/models/gemma-4-12b-hf"
hf download google/gemma-4-12B-it-assistant --revision 46d4c6f13f0ac0ad827b915669b8df9b81c64c51 \
  --local-dir "$HOME/models/gemma-4-12b-assistant"
python tools/zig/gemma4_12b_convert.py "$HOME/models/gemma-4-12b-hf" "$HOME/models/gemma-4-12b"   # mlx and mlx-lm installed
zig build native -Dcpu=apple_m1
zig-out/native/bin/tensorfold-native serve "$HOME/models/gemma-4-12b" --port 8888 --parallel 4 \
  --drafter "$HOME/models/gemma-4-12b-assistant"
```

`gemma4_12b_convert.py` keeps the language model's tensors (the image and audio inputs' are dropped), quantizes them
with mlx-lm's converter to affine 4-bit in groups of 64 (6.3 GB, 4.5 bits a weight with the scales), and writes the
checkpoint's own `gemma4_unified` config back with that quantization. Run twice, it writes the same tensors bit for bit.

## The model

The 12B is Gemma 4's dense decoder at another size: 48 layers, five sliding-window layers (window 1,024, head dim 256,
16 query heads over 8 key heads) to each global one (head dim 512 over one key head, keys reused as values,
proportional RoPE), hidden 3,840, GeGLU at 15,360, per-layer scalars, and the tied 262,144-token head with its logit
soft-cap at 30. The family reads both sizes from one geometry the config picks (`zig/src/families/gemma4/config.zig`):
every dimension must match the 31B's or the 12B's exactly, and other layouts (per-layer inputs, shared KV layers, MoE,
per-module widths) are refused by name. The kernels take each size as a define, so the 12B runs the 31B's kernels.

Without `--drafter` the 12B decodes plain at 88 tokens a second for one stream.

## The MTP assistant

`--drafter` reads the drafter's `config.json`: a Gemma 4 assistant type goes to the MTP assistant
(`zig/src/families/gemma4/assistant.zig`), which admits `gemma4_unified_assistant` and refuses others by name, and a
DFlash checkpoint to DFlash (the 31B's), each refused by name on the wrong target. The assistant is four layers 1,024
wide (three sliding, one global) with no keys or values of its own: each layer's queries attend over the target's own
caches, the last sliding layer's and the last global layer's, from the last kept row's position. Its input is the
pending token's scaled embedding beside the target's final-normed row, then its own projected output; its head is tied
to its own 262,144-row embedding. Its bf16 weights are quantized at load to 4-bit groups of 64 by MLX's affine rule. A
round drafts up to six tokens (0.97 ms a draft), every stream's chain in one command buffer, the next step's token
straight from the GPU's picks. Ordered (centroid) embeddings are refused by name.

Served, one stream decodes at 140-248 tokens a second with it (prose 140, code 223, structured 205, list 248;
3.2 accepted tokens a round), 183 against 88 undrafted on the quality suite's draft check, with the same tokens.

## Serving

| Checkpoint / platform | Qualification |
| --- | --- |
| The converted 12B, M5 Ultra, macOS 27 | Greedy and sampled decode, prompts to 252k, shared rounds, MTP drafts, structured output, prompt reuse, the server, below. |
| `google/gemma-4-12B-it-assistant` `46d4c6f1` | The one MTP assistant admitted for the 12B. |
| M1 to M4 | Refused: the lane projection needs M5 tensor units. |
| Images, audio | Not read yet: the converted checkpoint has the language model only, and `--vision` is refused by name. |

Structured output (`response_format`, `guided_*`) works with drafts on: the harness's structured check obeys 18 of 18
(each field greedy, sampled and with thinking), and a bad regex gets HTTP 400. Thinking is on with
`chat_template_kwargs: {"enable_thinking": true}`; replies carry it in `reasoning_content`.

Prompts are reused between requests as for the 31B: an 11,481-token conversation took 3.0 s to its first token fresh
and 0.09 s asked again (11,264 tokens resumed). A 1-token request after 2 to 120 idle seconds took 57-74 ms, where a
busy server takes 37 ms.

Thinking replies fill the window: at `--context 32768`, three greedy prompts (a proof with a checked list, a
tested red-black tree, a question on a 25,801-token document) thought until the window was full (32,705, 32,714 and
6,959 tokens), each drafted reply equal to the same request with `"draft": false`.

## Quality

The recipe harness (`gemma4-12b-m5`), thinking off: tools 58/60, JSON 30/30, GSM8K 239/250, MMLU 264/342; with
thinking on, the behaviour check's word problems 8 of 10 (every reply with its reasoning, none leaking tags). The
harness's thinking plan (`--plan thinking`) was not run for this release.

## Verification

The references are gilby's `gemma4_unified` Python engine (`6b0ead7a`, MLX 0.32.3, mlx-lm 0.32.0, lane kernels) for
decode and mlx-lm for long prompts (`tools/zig/gemma4_gate.py ... --family gemma4_unified`).

- Decode, four chat prompts x 128 greedy tokens: three equal the reference's tokens; the fourth parts at its second
  token on a near-tie (18.25 against 17.875 there, 18.125 for both here). As for the 31B since its speed work
  (decode attention and the projections' K splits), the logits are not the reference's bit for bit, and the last
  prompt rows sit as close to mlx-lm's as the reference's do (RMS 0.10-0.18 apart for both).
- Long prompts, 49 tokens after each, against mlx-lm in 2,048-token steps: 8k and 128k equal; 32k, 64k and 252k part
  at near-ties (23.125 / 23.0, 25.625 / 25.5, and an exact tie 24.5 / 24.5 at 252k, where mlx-lm's own reply runs
  into a loop).
- `tf-gemma4-lanes`: plain and copy drafts at one, four and eight streams, and the MTP assistant greedy at one, four
  and eight and seeded sampled at one and four streams, all equal to their references.
- The 31B's decode, long-prompt, lanes, DFlash and image receipts are byte-identical to `21e8040e`'s.

Native prompts in 512-row chunks (`tf-gemma4-run ... 49 512`): 3,961 tokens a second at 8k, 3,407 at 32k, 2,907 at
64k, 2,130 at 128k and 1,429 at 252k.
