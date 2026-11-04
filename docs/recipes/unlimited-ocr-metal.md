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

`tools/zig/uocr_reference.py` is the reference: `pages` writes the check pages (two synthetic report pages and three
pages of the paper the release ships), and `capture` runs the release's own `infer` on the CPU in torch bf16
(transformers 4.52.1, the model card's) and keeps the prompt ids, generated ids and text. `tf-uocr-decode` compares the
engine with a capture's stage dumps layer by layer, and `--chunk ROWS` checks that a prompt cut into chunks gives the
same bits.

Measured on the served engine (M5 Ultra, another model serving on the same GPU throughout), each case once, again and
as two requests at once:

| Pages | Prompt tokens | Reply tokens | First token | Decode | Text vs reference | Box coordinates moved |
|---|---:|---:|---:|---:|---|---:|
| report-1 | 909 | 198 | 1.10 s | 260.6 tok/s | identical | 0 / 16 |
| report-2 | 909 | 141 | 0.85 s | 248.7 tok/s | identical | 0 / 16 |
| paper-1 | 909 | 543 | 0.81 s | 260.1 tok/s | identical | 5 / 44 (at most 10) |
| paper-3 | 909 | 899 | 0.81 s | 259.4 tok/s | identical | 1 / 44 (1) |
| paper-10 | 909 | 1,959 | 0.81 s | 247.7 tok/s | identical | 2 / 28 (1) |
| all five as one document | 4,525 | 3,764 | 4.00 s | 223.7 tok/s | identical | 2 / 140 (1) |

The text matches the reference in every case; the only differences are box coordinates, mostly by one, where bf16
sums taken in another order (CPU torch against Metal) tip a rounding. Repeats and concurrent requests give the same
replies.

## Supported scope

BF16 on M5 GPUs, one stream, greedy, up to 131,072 tokens. A request takes 4 pages by default, and
`--vision-max-images` raises that to 64. Sampling, the publisher's optional n-gram repetition filter, a prompt cache, drafts, tools
and structured output are not offered. The 34 kernel launches are registered with the kernel contract.
