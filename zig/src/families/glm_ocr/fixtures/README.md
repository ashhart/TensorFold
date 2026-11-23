# GLM-OCR fixtures

`config.json`, `preprocessor_config.json` and `processor.json` come from
the `processor` command of `tools/zig/glm_ocr_reference.py`.

## OCR reader fixtures

The `ocr-*.json` files are greedy outputs of the script's `capture`
command (official Transformers 5.5.4, checkpoint revision
`2e85a62840ccac27daa451df36c736c4636b8628`), for `ocr_test.zig`. Each
holds `text` (the receipt's `text`), `original` (the case image's width
and height), `frame` (the `image_grid_thw` height and width times the
14-pixel patch, the tower input size), `source` and `prompt_sha256` (the
SHA-256 of the case prompt).

| Fixture | `examples.json` case | Prompt |
| --- | --- | --- |
| `ocr-text.json` | `text` | `Text Recognition:` |
| `ocr-table.json` | `table` | `Table Recognition:` |

The `examples` command draws the case images with the font that
`--font` names; these runs used Liberation Sans Regular
(SHA-256 `baccc64becc3eb7d104b7c84d99f5314a0a1f896e2b3ea6c2f22fc08d2003bee`).
The command also copies the model card's car and bee photos from a local
copy of `huggingface/documentation-images` at the revision the script
pins.

```sh
python tools/zig/glm_ocr_reference.py examples HF_IMAGES EXAMPLES_DIR --font FONT
HIP_VISIBLE_DEVICES=N python tools/zig/glm_ocr_reference.py capture \
  CHECKPOINT_DIR EXAMPLES_DIR/examples.json CASE OUTPUT_DIR --pci PCI_ID
```
