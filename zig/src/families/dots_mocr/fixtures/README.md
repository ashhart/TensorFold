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
