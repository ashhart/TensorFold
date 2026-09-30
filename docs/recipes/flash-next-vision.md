# Flash Next CUDA images

Run a complete local Flash Next checkpoint with `--vision --parallel 2` (or more), on one CUDA GPU.
This port supports images, including multiple images within the existing 4096 visual-token budget; video and
large-image extensions are not included. Image rows replace embeddings in every hyperconnection stream.
Multimodal RoPE follows the checkpoint's sections; text decode continues with its image-position offset.
Image prompts always prefill from the start and are never retained in the text-prefix cache, since identical
placeholder IDs can name different pixels. Grammar and ignore-EOS settings remain available.

A floating-point tower in the indexed checkpoint is discovered normally. EXL3 packs whose vision tower is a
quantized sidecar need a one-time CPU conversion, stored outside the original model snapshot:

```bash
python -m tensorfold.vision.exl3_convert /models/vision_k6.safetensors /cache/vision-f16-v1.safetensors
TENSORFOLD_VISION_WEIGHTS=/cache/vision-f16-v1.safetensors tensorfold serve /models --vision --parallel 2
```

The converter decodes represented EXL3 weights, transposes matrices and combines split Q/K/V. It records the
source SHA256 and converter/dtype version; repeat conversions reuse a matching artifact and refuse to
replace a mismatched one. Conversion never runs in the serving loader. The FP16 artifact loads into the
existing BF16 CUDA tower; rounding may differ from native quantized vision execution, so compare image
features and quality for the checkpoint in use. The original weights stay unchanged.

Admission counts an external tower separately, including its expanded resident bytes. It reserves 4 GiB of
image workspace by default; `TENSORFOLD_VISION_WORKSPACE_MIB` sets a measured override from 0 to 16384 MiB.
Image requests cannot use yieldable background lanes. Distributed vision and serial-only Flash Next image
serving are not supported by this port.

The integration is adapted from MiaAI-Lab patch 0008, with its MIT license in `LICENSES/MiaAI-Lab-MIT.txt`.
