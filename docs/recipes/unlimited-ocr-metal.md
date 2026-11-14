# Unlimited-OCR-2 on Metal

The native Metal engine serves Baidu's BF16 `baidu/Unlimited-OCR-2` at revision
`0480966938291445591f69f08a13b041c073a96a`, as published: the release's safetensors are read directly, with no
converted pack and no Python or libtorch in the served engine. It was qualified with Zig 0.17.0 on an M5 Ultra
(256 GB); the family needs the M5's tensor units and refuses other GPUs.

## Build and start

```sh
zig build native -Doptimize=ReleaseFast
zig-out/native/bin/tensorfold-native serve <Unlimited-OCR-2 checkpoint> \
  --name Unlimited-OCR-2 --host 127.0.0.1 --port 8896 --vision --vision-max-images 64 \
  --context 131072 --max-tokens 131072 --no-update-check
```

The release ships no chat template, so the server uses the family's own (`zig/src/families/uocr/chat_template.jinja`):
one page marker and image per image part, then the user's text, or the publisher's default instruction
`Multi page parsing.` when there is none. A checkpoint's own template, or `--chat-template`, takes its place.
JPEG pages need libjpeg-turbo 3.1.4.1 (`TENSORFOLD_LIBTURBOJPEG`), which keeps their pixels equal to the reference's;
PNG and WebP need nothing extra.

## Requests

Send page images as `image_url` parts (data URLs) in one user message of a `/v1/chat/completions` request, in reading
order. The reply is the publisher's layout format: one `<PAGE>` per page, then `<BLOCK>` lines naming each block's
kind, its box on a 0-999 page grid and its text, with tables as HTML and formulas as LaTeX. The model writes no
reasoning block; the reply is always `content`.

- **Page modes.** Each page takes gundam mode (the 1024 global view and 640 tiles, the publisher's server default)
  when its tokens fit the request's image budget, else base mode (the global view alone, 273 tokens). Without
  `--vision-image-tokens` the budget lets every page take gundam's widest grid (2,033 tokens a page).
- **Decoding.** Greedy, under the publisher's page guard: the reply ends after the last page's text. A request
  without `max_tokens` may use all the room its pages leave in the context, as the publisher's `infer` does.
- **Streams.** One request at a time; a second waits for the first. A `--parallel` above 1 is refused.

## Memory

The weights take 6.3 GiB and the caches 66 KiB a token (bf16 keys and values, fp32 router keys): 8.25 GiB at the full
131,072 tokens. The served process measured 17 GB resident at that context.

## Checks

`tools/zig/uocr_reference.py` is the reference. It runs the release's own code on the CPU (transformers 4.52.1, the
model card's) and reads it only through its modules' inputs and outputs:

- `pages` writes the check pages (two synthetic report pages and three pages of the paper the release ships).
- `capture` runs `infer` in torch bf16 and keeps the prompt ids, generated ids and text.
- `dump` keeps a page's stages, from torch bf16 and from float32 on the same bf16 pixels: each view's pixels and its
  SAM, CLIP and projector outputs, the decoder's input rows, every layer's prompt rows, the first logits and the reply.
- `blocks` takes one view of a dump through float32 and keeps each step's input and output (every SAM block, the neck
  and its two convolutions, CLIP's embedding and pre-norm, every CLIP layer, the projector), then runs torch bf16 on
  each step alone from the same input rounded to bf16.

Three tools gate the engine against float32. Each limit is twice torch bf16's own distance from float32 (or bf16
rounding's), in cosine error and in the worst row's relative error:

- `tf-uocr-blocks` runs each step of the image encoder alone on the reference's input, and every step must be within
  its limits.
- `tf-uocr-vision` encodes a dump's views (after checking that its own preprocessing gives the same pixels bit for
  bit). SAM's output must be within its limits; CLIP's and the projector's are reported against the same limits but
  not gated. CLIP takes SAM's output, and a few of its rows magnify bf16 rounding there: on the first gundam page
  (report-1), the tile view 1's CLIP row 79 is 0.186 from float32 in relative error, against a limit of 0.166 (twice
  torch bf16's worst row, 0.083), though every step on the way there is within its limits alone (at most 0.53 of them).
- `tf-uocr-decode` runs a page end to end: the prompt ids and the text rows must equal torch's, and with
  `--their-embeds` (the decoder alone on torch's input rows) every layer and the first logits must be within their
  limits. From our own image rows they are reported, as CLIP is. `--chunk ROWS` checks that a prompt cut into chunks
  gives the same bits, and every tool runs twice and must repeat byte for byte.

On report-1 (gundam, 909 prompt tokens) and on the same page in base mode (279):

- Each step alone passes on the tile view 1 and the global view: SAM blocks at most 0.53 of their limits, the neck
  0.27, CLIP's front 0.50, CLIP layers 0.52, the projector 0.50. A 5% error planted in one row of each kind of step
  fails it.
- Decode passes in both modes. With torch's input rows, ours picks torch bf16's token at all 197 steps. From our own
  rows, the reply equals torch float32's for all 197 tokens; torch bf16's departs from float32's at token 10. Layer 8's
  worst row is reported over its limit there (2.93 against 2.88); it is within it from torch's rows (1.44).

On the five-page document (4,525 prompt tokens) one row does the same: from our own rows, row 1934 is over the worst-row
limit at layers 8 to 10 (3.12 against 2.93 at layer 8), with every layer's cosine and the first logits within their
limits; from torch's rows it is 1.45 at layer 8 and every gate passes.

Measured on the served engine (M5 Ultra; another model's server loaded on the same GPU, idle) with the recipe
harness's `ocr` plan, each case once, again and as two requests at once:

| Pages | Prompt tokens | Reply tokens | First token | Decode | Text vs reference | Box coordinates moved |
|---|---:|---:|---:|---:|---|---:|
| report-1 | 909 | 198 | 0.48 s | 299.4 tok/s | identical | 2 / 16 (at most 2) |
| report-2 | 909 | 141 | 0.28 s | 301.1 tok/s | identical | 0 / 16 |
| paper-1 | 909 | 543 | 0.28 s | 297.1 tok/s | identical | 1 / 44 (1) |
| paper-3 | 909 | 911 | 0.28 s | 301.1 tok/s | citations written as `$ [5, 6, ...] $` | 1 / 44 (1) |
| paper-10 | 909 | 2,004 | 0.29 s | 298.3 tok/s | table headers written in LaTeX | 0 / 28 |
| all five as one document | 4,525 | 3,767 | 1.70 s | 282.5 tok/s | differs; 34 boxes against 35 | 98 / 136 (at most 824) |

The reference is the release's own `infer` in torch bf16 on the CPU; its float32 run departs from it too (paper-3 at
token 13, paper-10 at 1,660, the document at 16). **The single pages are what this recipe claims:** fed torch bf16's
reply, `tf-uocr-decode` picks otherwise at 2 of report-1's 197 steps, 1 of paper-1's 542, 2 of paper-3's 898 and 1 of
paper-10's 1,958 (report-2 has none), and each is a tie under the one-step policy, the same two contenders on both
sides with both margins within one bf16 step, which the tool now gates: report-1 step 10 (torch 2892 and 4774 equal,
ours 4774 by one step) and step 33 (torch 27738 and 29037 equal, ours 29037 by one), paper-1 step 88 (torch 10410 and
10669 equal, ours 10669 by one), paper-3 step 145 (torch 764 over 223 by one step, ours 223 by 0), step 557 (torch 27343
and 26670 equal, ours 27343 by one step) and paper-10 step 67 (torch 7233 over 3289 by one step, ours 3289 by 0).
report-1's and paper-1's other tokens render the same text, so their transcripts are identical. **The five-page document
is reported beside them, outside that policy:** of its 4 departures in 3,763 steps, 16 (torch 6420 over 6207 by one
step, ours 6207 by 0) and 757 (torch 26934 over 25282 by one step, ours 25282 by 0) are ties, but at step 300 torch
bf16 holds 16259 and 16208 equal where ours has 16259 by two steps, and at 431 torch has 13636 over 12326 by one step
where ours has 12326 by three. Float32 sits on our side of both (16259 by 1.7 steps, 12326 by 2.1); on torch's own
image rows (`--their-embeds`) only 431 departs, by one step, so the two wider margins come from our image rows (row
1934 of page 3, which `tf-uocr-blocks` finds no further from float32 than torch bf16 at any stage). The tool's
`--reply-reported` runs the document this way. A different pick sends the rest of the reply its own way; in the document
that moves a box and the coordinates after it. Repeats and concurrent requests give the same replies.

The harness pairs boxes in reading order. In the document the reference writes one box that ours does not, an empty
`other` block spanning paper-3's three contribution items after their own boxes, so from there each of our boxes is
compared with the reference's next one: that is the 824. Set that box aside and the other 34 pair up, with 74 of their
136 coordinates moved, by at most 3.

## Supported scope

BF16 on M5 GPUs, one stream, greedy, up to 131,072 tokens. A request takes 4 pages by default, and
`--vision-max-images` raises that to 64. Sampling, the publisher's optional n-gram repetition filter, a prompt cache, drafts, tools
and structured output are not offered. The family's 24 Metal kernels (12 in the image encoder, 12 in the decoder) are
registered with the kernel contract, each with a launch record, through 30 gate cases (a case per dispatch form:
`uv_attention` for SAM and for CLIP, `uv_im2col` for its three convolutions, `ud_attend` for prompt rows and steps,
`ud_norm` alone and with each of its two additions).
