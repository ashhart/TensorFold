The JSON files are unchanged metadata from the Apache-2.0 checkpoint
`lightonai/LightOnOCR-3-1B`, revision
`23d544085c886bbdace647bcfab91203299417e2`.

The image input is written from the checkpoint's `processor_config.json`
and from the processor's measured output, with no library source. With
`longest_edge` 1540, an image whose longer side exceeds 1540 is scaled
by `long / 1540` in f64, each side floored; each side then rounds up to
a multiple of `patch_size` x `spatial_merge_size` = 28 pixels, and one
image marker expands to `(h / 28) * (w / 28)` tokens. Running the full
1B processor (text and image) as a black box gave 30505 sizes, all
equal, and the CHW pixels of 40 test images, all byte-equal.
`processor_pixels.txt` pins those bytes for 40 synthetic pages, made by
`pixels_test.zig` and by `tools/zig/lighton_ocr_pixels.py`
from one integer formula: each row is the page's size, the processor's
CHW f32 shape and the SHA256 of its bytes. The image tests use
synthetic pixels and contain no model weights, generated pages or
private user data.
