The JSON files are unchanged metadata from the Apache-2.0 checkpoint
`lightonai/LightOnOCR-3-1B`, revision
`23d544085c886bbdace647bcfab91203299417e2`.

Image processing follows Hugging Face Transformers `v5.5.4`, specifically
`models/pixtral/image_processing_pixtral.py` and
`models/lighton_ocr/processing_lighton_ocr.py`. The LightOnOCR processor
expands one image marker to `floor(grid_h / 2) * floor(grid_w / 2)` tokens.
Its encoder still reads the complete raster patch grid, including unpaired
edges. The image tests use synthetic patch colors and contain no model
weights, generated pages or private user data.

The resize algorithm is adapted from the Apache-2.0 Transformers image
processor, copyright 2024 The HuggingFace Inc. team. The LightOnOCR token
count follows its Apache-2.0 processor, copyright 2026 The LightOn Team
and The HuggingFace Inc. team. The original source files include the
Apache License notice at <https://www.apache.org/licenses/LICENSE-2.0>.
