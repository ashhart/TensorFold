# dots.mocr contract fixtures

The numeric checkpoint settings come from `dots-studio/dots.mocr` at
revision `e539fbb52280393adc081b289ec597430a0f9031`.
The config fixture omits remote-code loader metadata. No inference code
or model weights are included here. Read the original model's license
when using or distributing its assets.

`processor-reference.json` records independent slow Qwen2VL processor
pixels. Each synthetic RGB sample at `(y, x, channel)` is
`(7*x + 13*y + 50*channel + (x*y)%31 + seed)%256`.
The hash is SHA-256 over merge-ordered pixel_values as little-endian FP32.
The image grids include time, height, and width. Cases cover an unchanged
image, non-square resize, square resize, and a long image.

Regenerate against an existing local checkpoint:

```sh
uv run --python 3.12 --with transformers==4.51.0 \
  --with pillow==11.2.1 --with numpy==2.2.5 \
  python tools/zig/dots_mocr_reference.py processor CHECKPOINT_DIR \
  zig/src/families/dots_mocr/fixtures/processor-reference.json
```

`TF_DOTS_DIR=CHECKPOINT_DIR zig build test-dots-mocr-host
test-dots-mocr-fast` checks the existing complete shard headers, their
index, all 643 tensor requirements, and exact declared weight bytes.
Header checks establish extents and layouts, not weight checksums or
GPU inference fidelity. Without `TF_DOTS_DIR`, only that live checkpoint
test skips. The synthetic admission and processor checks still run.

## Hugging Face example

`demo-reference.json` records the image and processor pixel hashes for
`demo/demo_image1.jpg` from the model author's repository at
`23f3e5612fb8066d4034d5ecfc8f33a9243533eb`. The image stays outside Git.
The example prepares 19520 encoder patches and 4880 merged image rows.
Set `TF_DOTS_DEMO_IMAGE` to the unchanged image file to run the native
JPEG-to-patches comparison in both host-test modes. This check also needs
the repository's pinned libjpeg-turbo 3.1.4.1. Use
`TENSORFOLD_LIBTURBOJPEG` to select a separate local library file.

`tools/zig/dots_mocr_reference.py oracle` runs the pinned Hugging Face
example on one HIP GPU with Transformers 4.57.6 and qwen-vl-utils 0.0.10;
`compare` checks a native greedy trace against its output:

```sh
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  python tools/zig/dots_mocr_reference.py oracle CHECKPOINT_DIR DEMO_IMAGE \
  OUTPUT_DIR --pci PCI_ID --max-new-tokens 64
python tools/zig/dots_mocr_reference.py compare OUTPUT_DIR NATIVE_PREFIX
```

`rope_inv_freq.txt` holds the 64 float32 bit patterns of the checkpoint's
rotary `inv_freq` as Transformers 4.57.6 computes them on the CPU (theta 1e6,
head dim 128), dumped by the oracle's `text` mode (`rope-inv_freq.npy`). The
CUDA recipe uploads them as they are, because Torch's CPU `powf` is not
correctly rounded: entry 37 is one ulp from the correctly rounded value.

`vision_inv_freq.txt` holds the 32 float32 bit patterns of the tower's
rotary `inv_freq` (theta 10,000, dim 64) as the checkpoint's Transformers
model holds them on the GB10, dumped by the oracle as
`vision-rope-inv_freq.npy`. They equal the correctly rounded table, which
the host test checks bit for bit.

`own-prompts.json` holds three prompts of our own as token ids
(`short`, `page` and `chunks`: 18, 578 and 2,537 ids), the GB10 runner's
`dots-text` input. Their text is our own `ownprompts.py` (in #2673's
description), each rendered as one user turn with the checkpoint's chat
template and tokenized by the pinned oracle (Transformers 4.57.6).

## OCR reader fixtures

The reader's prompt (`../prompt.txt`) and both test pages are written
for TensorFold. The `ocr-*.json` files record greedy `oracle` runs with
that prompt on those pages, and hold no model text: `source` and
`image_sha256` (the page), `original` (its width and height), `frame`
(the oracle's `prepared_images` size, the tower input the boxes are in),
`prompt_sha256`, `max_new_tokens`, `eos_reached`, `output_tokens`,
`output_ids_sha256` (SHA-256 over the output ids as little-endian
`u32`) and `text_sha256` (SHA-256 of the oracle's `text` field).

| Fixture | Page | `--max-new-tokens` | Result |
| --- | --- | --- | --- |
| `ocr-page.json` | `page.png` | 4096 | the whole page, ended by EOS |
| `ocr-page-cut.json` | `page.png` | 128 | JSON cut by the limit |
| `ocr-one-line.json` | `one-line.png` | 4096 | one text block |

`ocr_test.zig` reads written block lists in the model's form. Set
`TF_DOTS_OCR_DIR` to a directory that holds the three runs as `page`,
`page-cut` and `one-line` to check each run's page, prompt, token ids
and text against its fixture, and to read its text with the reader.

`pages` draws both pages with Pillow 12.2.0 in Liberation Sans Regular,
Liberation Sans Bold and Liberation Serif Regular 2.1.5 (font SHA-256
`baccc64becc3eb7d104b7c84d99f5314a0a1f896e2b3ea6c2f22fc08d2003bee`,
`769673c4355020b1e28a14c366a152da410ab6b16239fe883ebc35b73624835b` and
`86b9ea1c2f41bed9d7c09ccad4abc2894b33df5de60e5bbbece5d48610911870`):
`page.png` is 1240x1754 with headers, a title, text, a caption, a table
with a spanned header cell, a list and a page number; `one-line.png` is
960x96 with `TensorFold reads one line of text.` `record` writes a
fixture from a run. The runs used one gfx1201 GPU, ROCm 10.1 Torch,
Transformers 4.57.6, qwen-vl-utils 0.0.10, Pillow 12.2.0 and flash-attn
2.8.3.post1 with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`:

```sh
python tools/zig/dots_mocr_reference.py pages PAGES_DIR
HIP_VISIBLE_DEVICES=N FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  python tools/zig/dots_mocr_reference.py oracle CHECKPOINT_DIR \
  PAGES_DIR/page.png RUNS_DIR/page --pci PCI_ID --max-new-tokens 4096 \
  --prompt zig/src/families/dots_mocr/prompt.txt
python tools/zig/dots_mocr_reference.py record RUNS_DIR/page \
  PAGES_DIR/page.png zig/src/families/dots_mocr/fixtures/ocr-page.json
```
