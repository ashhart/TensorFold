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

`--parallel 4` caps this example at four active requests; additional requests wait for a free slot.

Structured output and required tool calls need the release archive's `lib/` or `zig build grammar -Dxgrammar=DIR`; prepare DIR with `git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar` (use the native build's same `--prefix` for grammar).

Without `--drafter` the model decodes plain, at about 51 tokens a second for one stream at short context (mlx-lm 46)
and 40 at 32k. With z-lab's DFlash drafter
a stream decodes at 150-190 tokens a second (6-7 accepted tokens a round on chat and code, 10 on copying replies,
2 on open-ended prose), with the same tokens.

For the 31B the loader admits its dense geometry exactly: 60 layers, five sliding-window layers (window 1,024, head dim
256) to each global one (head dim 512, keys reused as values, proportional RoPE), GeGLU, the tied 262,144-token head
with its logit soft-cap, MLX affine weights at the widths [other widths](#other-widths) lists. It refuses the 26B-A4B
MoE, per-layer inputs and shared KV layers, each by name. With `--vision` it also loads the image tower
([images](#images)). The 12B (`gemma4_unified`) runs on the same family with its MTP assistant: [Gemma 4 12B, native
Metal](gemma-4-12b-native.md).

Every linear, the tied head included, runs on the core lane projection (`zig/src/core/lane_projection.zig`) on M5
tensor units; the rest of a layer is `zig/kernels/metal/gemma4/glue.metal`. The glue ports the Python reference's
decode kernels (gilby's `gemma4-dense` branch, #360, built on cshintov's Gemma kernels from #10) with their rounding
points, and MLX's own ops where that reference calls them: the looped RMS norm, the 4-bit embedding times its bf16
scale, `geglu` and the soft-cap as bf16 ops, and the global layers' RoPE frequencies from MLX's precise `pow`.
Each output equals the reference's bit for bit at 1, 2 and 16 rows (`tf-gemma4-kernels`, `tf-gemma4-glue`, on
fixtures dumped from that reference) at the reference's K split. Decode takes each shape's fastest split instead
(`weights.splitK`: 4 for o and down, 8 for the gate/up stack and the head), another fp32 summation order: at one row a
handful of outputs a linear move by one bf16 step (5 of 43,008 on gate_up, 99 of 262,144 on the head, none on qkv).

A stream keeps a ring of window + 512 slots for each sliding layer and a buffer for each global one; rolling back is
moving the cache's length. A decode forward takes up to 16 rows of one stream and 128 of several, and every row keeps
its bits. A one-row window runs five or six dispatches a sliding layer rather than ten, and seven a global one: the
previous layer's dense tail is the qkv projection's prologue, the attention tail the gate/up projection's with geglu as
its epilogue (the stack's rows interleaved in 16-row blocks, so a lane holds a gate column and its up column), and the
down projection sums its own input (`zig/kernels/metal/gemma4/fused.metal` on the lane core's `Fused` hook); a sliding
layer's attention kernel preps its own query heads and writes the row's key and value, and with one key chunk writes
the merged output itself. The same bits as the kernels apart.
Prompts run in chunks of 512 rows on Gemma's private `families/gemma4/affine_mm` and a tensor-unit attention kernel, at about 1,360
tokens a second at 8k, 1,200 at 32k and 560 at 252k.

The drafter is z-lab's DFlash (`zig/src/families/gemma4/dflash.zig`): five Qwen3-style layers reading six of the
target's layer outputs (1, 12, 23, 35, 46, 57) through `fc` and a norm, drafting a block of the pending token and up
to 15 mask tokens with the target's own embedding and tied head. Its bf16 weights are quantized at load to 4-bit
groups of 64 by MLX's affine rule and run on the same lane projection; the context keys and values of each stream
live in a 2,047-slot ring for the four sliding layers and a growing buffer for the full one, as in z-lab's MLX model.
It sits in front of the engine as the lane core's `Drafted` wrapper (chains, no trees): the target hands it each
prompt chunk's and each verify's tapped rows, and the depth rule prices its drafts against the forward's cost by rows
timed at load. A round's streams draft in one block forward (3.4 ms for one stream, 9 for eight) and absorb in one
pass, whose command buffer the block settles.

The context defaults to 32,768 tokens; `--context` raises it to the checkpoint's 262,144, at the cost of attention
scratch that grows with it.

Prompts are reused between requests (`zig/src/families/gemma4/snapshot.zig`). A kept state is each sliding layer's
last 1,024 keys and values, every global layer's keys and values so far, and the DFlash drafter's own context for the
same stream (4 KB a token in its full layer, 34 MB of sliding rings), so a resumed stream drafts as a fresh one does. A
prompt chunk's rows are their own arithmetic, so states are kept and resumed only where the request's planned chunks
start, as Step-5's are; a drafting stream does not resume a state kept without the drafter's context, and none is
resumed inside an image. The budget is what 70% of RAM leaves beside every stream's caches, 16 GiB by default
(`--prompt-cache-gib`, 0 for none). A later turn of a 3,336-token conversation resumed 3,329 tokens and answered as the
first pass did; one with an image after 2,300 tokens of system prompt resumed 3,584 of 3,607 tokens, alike. Served
with the drafter and `--vision`, an 11,481-token conversation took 7.8 s to its first token fresh and 0.23 s asked
again (11,264 tokens resumed); with an image in it, 8.1 s and 0.13 s.

Structured output (`response_format`, `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`) is enforced
through `libtfgrammar`, as [structured output](../api.md#structured-output) describes, with drafts on: before each
verify the `Drafted` wrapper hands the drafter's held drafts to the host, where the grammar cuts the ones it rejects.
The recipe harness's structured check obeys 18 of 18 (each field greedy, sampled and with thinking), and a bad regex
gets HTTP 400. Tools are called with `tool_choice: "auto"`; `"required"` and a named function get HTTP 400, as the
grammar for them reads `<tool_call>` markup and Gemma's template writes `<|tool_call>call:NAME{...}<tool_call|>`. The
weights, both scratches and the drafter's and tower's buffers stay in one residency set, which the idle keepalive's
tick keeps wired: a 1-token request after 2 to 120 idle seconds took 0.10-0.12 s to its first token, where a busy
server took 0.07 s (ten idle gaps, none slower).

| Checkpoint / platform | Qualification |
| --- | --- |
| The pinned 31B 4-bit, M5 Ultra, macOS 27 | Greedy decode, prompts, shared rounds, copy and DFlash drafts and the server, below. |
| `z-lab/gemma-4-31B-it-DFlash` `eabd6483` | The one drafter admitted: DFlash v1, block 16, six taps, head dim 128. |
| M1 to M4 | Refused: the lane projection needs M5 tensor units. |
| Images (`--vision`) | PNG and JPEG data URLs, and public HTTPS URLs with `--vision-urls`, below. |
| Structured output | `response_format` and the `guided_*` fields, drafts on and off, below. |
| Other widths | MLX affine 2-6 or 8 bits in groups of 64, uniform or per module: [other widths](#other-widths). |

## Images

`serve --vision` loads the checkpoint's image tower (`vision_tower.`, `embed_vision.`; 1.1 GB, bf16 but for the 4-bit
projection) and reads PNG and JPEG data URLs in user messages and tool results (with `--vision-urls`, public HTTPS URLs
too), decoded to Pillow's pixels by core's picture decoder, as [image input](../vision.md#the-native-server) describes.
`zig/src/families/gemma4/image.zig` prepares an image as transformers' `Gemma4ImageProcessorPil` does: the
largest aspect-preserving size whose sides are multiples of 48 and whose 16-pixel patches fit 280 tokens (PIL's bicubic
resize, then 1/255), so an image becomes 256 to 280 tokens (70 or 140 with `detail: low` or a smaller
`--vision-image-tokens`). The template's `<|image|>` expands to `boi`, the image's rows, `eoi`.

`zig/src/families/gemma4/vision.zig` runs the 27-layer encoder on gemm_nax's tensor-unit GEMMs at torch's per-op bf16
rounding: the patch projection and the x and y position rows, RMS norms in fp32 times their weights, per-head q, k and
v norms, the axial 2D RoPE (theta 100, x on dims 0-35 and y on 36-71), attention with scale 1 as two batched GEMMs and
a softmax, GELU-tanh gating, the 3 x 3 pooler, the standardization and the projection on Gemma's private affine_mm. An image takes
about 90 ms, and adds about 270 ms to a request's first token (a 290-token prompt with an image 365 ms, the text
alone 92 ms). In the language model the image's rows replace its tokens' embeddings, and on the 50 sliding layers they
see one another both ways, as `use_bidirectional_attention: "vision"` asks (the prompt attention kernel takes each
row's last key; global layers stay causal); a prompt chunk never cuts an image.

Checked with `tools/zig/gemma4_vision_fixtures.py` (transformers 5.17 and torch, bf16 and fp32 runs of
`Gemma4VisionModel`) and `tf-gemma4-vision` on five drawn images, from 20 x 15 to 1200 x 800 and 180 x 1400: the patches
equal the PIL processor's bit for bit, and the tower's rows are as close to the fp32 run as torch's own bf16 run (7.6%
relative error averaged over the images, torch's 7.8%). AutoProcessor with torch installed picks the torchvision
processor instead, whose resize differs from PIL's by a level at a few pixels in a thousand. Served with the drafter,
a described image answers token for token as mlx-vlm 0.7.2 does (80 greedy tokens, the same 287-token prompt), with
drafts on and off; `tools/zig/vision_served.py` passes its 19 checks, concurrent image requests equal their runs
alone, and text-only prompts keep their bits (the receipts above, unchanged).

## Other widths

The family reads each linear's width from the checkpoint's `quantization` table, as [the 12B
recipe](gemma-4-12b-native.md#other-widths) describes: 2, 3, 4, 5, 6 or 8 bits in groups of 64, uniform or per module,
stacks of mixed widths in runs, every other width or mode refused by name. 4-bit groups of 64 keep the reg kernel and
the fused one-row pipelines: the 4-bit decode, 512-row prompt and 8k and 32k receipts are the same bytes as before
other widths were read. Other widths run the lane kernel's general unpack, unfused.

[`unigilby/gemma-4-31B-it-qat-oQ8e`](https://huggingface.co/unigilby/gemma-4-31B-it-qat-oQ8e/tree/53dc2af091c166c8cefe5f670e744864f964ad47)
(`53dc2af0`, oMLX oQ uniform 8-bit from Google's QAT checkpoint, 33.8 GB): decode against gilby's oracle, the four
prompts x 128 tokens, two equal and two parting at near-ties (24.625 / 24.5, 21.25 / 21.125); `tf-gemma4-lanes`
plain, copy and DFlash drafts equal at one, four and eight streams, seeded at one and four; images: the tower's rows
7.6% from fp32 (torch bf16 7.8%), the 8-bit projection included, and `vision_served.py` passes.

Served with DFlash (harness target `gemma4-31b-oq8e-m5`): one stream 40, 63, 52 and 55 tokens a second (prose, code,
structured, list), four streams 110-180 together; the draft check 50 against 22 undrafted; structured output 18 of 18,
concurrent 12 of 12. On the four prompts DFlash accepts 1.6-4.1 tokens a round on this pack against 2.0-5.8 on the 4-bit
one (`tf-gemma4-lanes`), whose unfused path accepts as many as its fused one.

## Verification

The oracle is gilby's `gemma4-dense` Python engine (`a6ff795e`, MLX 0.32.3, mlx-lm 0.32.0) with lane kernels. The
scripts that run it and compare (`tools/zig/gemma4_gate.py`, `gemma4_dump.py`) import that Python engine, so they
stay on #879's branch; this tree holds the native side of each step:

```bash
zig-out/bin/tf-gemma4-run M O N 128 16   # O: the oracle's tokens and margins for four chat prompts; N: ours
zig-out/bin/tf-gemma4-lanes M O && GEMMA_DRAFTER=D zig-out/bin/tf-gemma4-lanes M O
zig-out/bin/tf-gemma4-run M O R 128 512
zig-out/bin/tf-gemma4-run M L NL 49 512  # L: the oracle's long prompts (8k to 252k tokens)
```

- Four chat prompts, 128 greedy tokens, prompts fed one row at a time or in windows of 16: three equal the oracle
  token for token; the story prompt parts at token 56, where the native run ties the oracle's top two (24.5 and 24.5,
  the oracle 24.5 and 24.375) and takes the lower id. Every row keeps its bits at any window width and beside other
  streams' rows.
- Long prompts against the oracle's mlx-lm prompt forward (fed in 2048-token chunks, as `mlx_lm.generate` feeds a
  prompt), 49 greedy tokens each: equal at 32k, 64k and 252k. At 8k and 128k the two part at tokens 25 and 30, each
  where mlx-lm's top two logits are equal (28.25 and 26.25); the native run puts them 0.125 and 0.75 apart.
- The lane core: copy-drafted and DFlash-drafted streams, alone and 4 and 8 in shared rounds, equal the engine's
  plain replies token for token, on the four prompts and on two copy-heavy ones.
- Served on an M5 Ultra with the drafter: greedy token SHAs equal the engine's receipts alone, with drafts off and on,
  and four at once; thinking replies keep their reasoning apart, and tool calls parse.

With the drafter, one stream decodes at 150-190 tokens a second (6.4, 6.4, 2.1 and 6.7 tokens a round on the four
prompts; 10.2 and 7.8 on copying replies) and eight streams at about 210 together. Plain decode is 51 tokens a second
for one stream. The drafts match z-lab's own MLX drafter's acceptance. Shared rounds draft shallow on purpose: past 16
rows the forward is bound by tensor-op compute at about 0.8 ms a row (40 ms at 32 rows, 71 at 64, 130 at 128, the
same rate as the prompt path and mlx-lm's), so the depth rule, given those costs, finds deeper shared drafts do not
pay; the drafter's own cost is 9 ms a round for eight streams.

Long prompts, one after the other on the same M5 Ultra: native in 512-row chunks (`tf-gemma4-run ... 49 512`), mlx-lm
0.32.0 in 2048-token steps into a fresh cache. Native decode is the run's 49 tokens after the prompt, timed to 0.1 s.

| Prompt | Native prompt tok/s | mlx-lm prompt tok/s | Native decode tok/s | mlx-lm decode tok/s |
|---|---|---|---|---|
| 8k | 1385 | 1305 | about 38 | 39 |
| 32k | 1241 | 1139 | about 35 | 32 |
| 64k | 1072 | 944 | about 33 | 32 |
| 128k | 882 | 651 | about 27 | 25 |
| 252k | 570 | 440 | about 20 | 16 |

Prompt attention takes 32 query rows of a head a threadgroup: each simdgroup scores 32 keys of a block over the whole
head and accumulates its own 64 output columns, the row maxes and fp32 probabilities crossing threadgroup memory 128
keys at a time. A 512-row chunk's 60 layers spend 138 ms there at 32k and 527 ms at 128k, where 16 rows a threadgroup
spent 194 and 810 (mlx-lm's fused attention about 161 at 32k): the kernel streams keys and values, so what each load
serves decides its rate, and vector loads, fetching a step ahead, and sparing interior blocks the mask and the rescale
moved nothing. The prompt GEMMs use Gemma's private affine_mm, which dequantizes each step's weights to bf16 in its tile (scale
times code plus bias in fp32, rounded once, as mlx-lm's quantized matmul does) and accumulates on the tensor unit over
the whole depth: a 512-row chunk's 60 layers take 284 ms there (mlx-lm's quantized matmuls 302), where the exact-codes
design, its per-group scale and bias epilogue and its row-sums pass took 344. So the prompt path rounds its weights as
mlx-lm does while the decode path's lane kernels keep the codes exact: over 40 tokens their last-row logits part by up
to 0.8.

A plain token at short context is 19.7 ms on the M5 Ultra (mlx-lm 21.6), where it was 23.7: the 241 projections read
17.3 GB in about 18 ms at the decode splits, 20.3 at the reference's (the 168-tile o and down projections ran two
waves of 256-thread groups at a split of 8 and one wave at 4; the gate/up stack and the head streamed 84-group chains
at 1, eight times the simdgroups at 8); the attention's 120 dispatches took 0.9 ms and the glue's 0.7, which running
as the projections' prologues shortens by 0.2 (a prologue costs what the tail kernel cost, and only the dispatch gap
goes) and the attention's prep and merge folded into its kernel by 0.6. `tf-gemma4-bench` prints the stages
(`GEMMA_DSTAGES`), each shape plain and fused (`GEMMA_FUSED`) and the glue kernels alone (`GEMMA_GLUE`).

## Shared arithmetic at landing

The shared affine and lane kernels, their Zig encoders, other-family callers and numerical tests retain the released
implementation. Gemma's rounded-weight dense source is in `zig/kernels/metal/gemma4/affine_mm.metal`. Its fused lane
variant is in `zig/kernels/metal/gemma4/lane_projection.metal`, with host encoding in `families/gemma4/fused_lane.zig`.
Ordinary Gemma lane packing and projections still use the shared lane module. The measurements above are the original
PR's receipts. They do not qualify the isolated landing tree. Re-run numerical, resource, teacher-forced, draft/plain,
resumed/fresh, concurrent/solo, vision and performance gates on that tree.
