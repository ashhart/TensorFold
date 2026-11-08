# LightOnOCR-3 on native HIP

The native HIP engine serves the three official BF16 sizes of
LightOnOCR-3 (0.8B, 1B and 4B). All three are qualified on gfx1201 only.
The 0.8B and 4B models use the Qwen3.5 family. The 1B model uses a Qwen3
text decoder and Pixtral vision encoder in `families/lighton_ocr`.
The legacy `mistral3` type is accepted only with the checked LightOnOCR
architecture and tensor shapes.

The implementation follows the existing ownership boundaries:

| Responsibility | Files |
| --- | --- |
| 1B config, image processor and weight admission | `zig/src/families/lighton_ocr/` |
| 1B Pixtral tower, Qwen3 text and lane backend | `zig/src/families/lighton_ocr/backend/hip/` |
| 0.8B/4B text loading and HIP engine | `zig/src/families/qwen3_5/` |
| Shared Qwen vision tower | `zig/src/vision/qwen_hip*.zig` |
| New HIP operators | `zig/kernels/hip/lighton_ocr.hip`, `qwen_vision.hip` |
| Native registration and image offer | `zig/src/native/hip.zig`, `hip_vision_offer.zig` |
| Device contracts | `zig/tests/hip/contract_lighton*.zig`, `contract_qwen*.zig` |
| Model capture and state checks | `zig/tests/lighton_hip/` |
| Pinned HF reference fixture capture | `tools/zig/lighton_ocr_reference.py` |
| 0.8B/4B on CUDA: BF16 chat admission and preflight | `zig/src/families/qwen3_5/cuda_decide.zig`, `cuda_bf16_bytes.zig` |
| Shared Qwen vision tower on CUDA | `zig/src/vision/qwen_cuda*.zig` |

Weights stay outside the checkout. The HIP loader widens BF16 weights to
FP32 without changing their values. Activations stay BF16. Allow room for
this storage, the vision tower, scratch buffers and each request's caches.
Do not change checkpoint files or add quantization metadata.

## Build and serve

Use Zig 0.17.0 and the local HIP compiler. Select the target GPU architecture
explicitly. This example uses a Radeon gfx1201 GPU.

```sh
zig build native-hip -Dhip=true -Dhip-arch=gfx1201 \
  -Dhipcc=/opt/rocm/core/bin/hipcc -j4
zig-out/native-hip/bin/tensorfold-native serve MODEL_DIRECTORY \
  --backend rocm --vision --context 4096 --parallel 2 --no-thinking \
  --name lighton --host 127.0.0.1 --port 8090
```

The 0.8B and 4B models load as `qwen3_5`, whose template opens a reasoning
block when `enable_thinking` is true. `--no-thinking` makes the server
default match the model cards, which render with `enable_thinking=False`;
a request can still set `chat_template_kwargs`. The 1B template has no
such switch, and its family declares no reasoning block.

JPEG inputs require the project's pinned libjpeg-turbo 3.1.4.1. Set
`TENSORFOLD_LIBTURBOJPEG` to that library when it is outside the normal
loader path. See [native preview](../../ZIG-PREVIEW.md) for image libraries.

For an isolated GPU test, identify the physical PCI address and acquire the
existing device lease. Shared tests must account for current users, the
complete planned allocation and a VRAM reserve. Set a process memory cap
and monitor remaining VRAM. Select the ROCm GPU UUID and verify that visible
HIP device 0 has that PCI address before loading. ROCm ordinals can differ
from `amd-smi` ordinals.

Use image-only user content for transcription. Add a text part containing
`grounding` for bounding boxes. Send images as OpenAI `image_url` data URLs.
PDF rendering belongs in the client. The family does not run a PDF parser.

## Native CUDA (0.8B and 4B)

The native CUDA engine serves the 0.8B and 4B through the CUDA Qwen3.5
family. Both are qualified on a GB10 (sm_121) only. The 1B is not served on
CUDA yet. The weights load as stored, in BF16. The tied output head is a
packed copy of the embedding rows, which adds 0.5 GB (0.8B) or 1.3 GB (4B)
to the weights.

```sh
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc
zig-out/native/bin/tensorfold-native serve MODEL_DIRECTORY \
  --vision --context 4096 --parallel 2 --no-thinking \
  --name lighton --host 127.0.0.1 --port 8090
```

On the model card's receipt, greedy transcription gives the HF ids to EOS on
both sizes. In grounding mode, the box digits differ from HF's greedy ids
after near-ties; HF scores each native id at its top logit or one BF16 step
below it, except two 4B ids at two steps.

A BF16 Qwen3.5 checkpoint serves chat on CUDA only when it has an image
tower that the CUDA tower reads and no `joint_head_config.json`. A BF16
checkpoint with a joint head is a Clef decision model (`/v1/systemone`).
A BF16 checkpoint without either is refused. The chat path refuses
`--drafter`, `--slide`, `--kv-cache fp8` and `--prompt-precision int8` by
name, because it is not qualified with them. Its own copy drafts and
`--no-drafts` give the same tokens.

## Reference examples and diagnostics

The model cards use this receipt in their Transformers examples:

[Official receipt fixture](https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/28fe12cdf7816b5dde94e22051b2ec8dc74267b7/SROIE-receipt.jpeg).

Its SHA256 is
`d66a3287c847823c6ddab9f8b2080582b96a66b7816d178472f697ea84e227b3`.

`tools/zig/lighton_ocr_reference.py` pins each checkpoint revision, the
SHA256 of every non-weight file in it, and the Transformers 5.22.1
model, processor and tokenizer sources each size loads. Every command
checks those pins before any model code runs. `sources` prints the
hashes and loads no model. `examples` runs the model card's
Transformers example on the receipt, in grounding and transcription
mode. It runs greedy and the card's own sampled call (1024 tokens,
`enable_thinking=False` on the 0.8B and 4B, temperature 0.2). It writes
a JSON receipt of prompt and output ids with their hashes:

```sh
python3 tools/zig/lighton_ocr_reference.py sources --size 0.8B \
  --model MODEL_DIRECTORY
python3 tools/zig/lighton_ocr_reference.py examples --size 0.8B \
  --model MODEL_DIRECTORY --image SROIE-receipt.jpeg \
  --output RECEIPT.json --device cuda:0
```

`tools/zig/lighton_ocr_pixels.py` checks the same 1B pins, then runs the
1B processor, called as the card does, on the 40 synthetic pages that
`zig/src/families/lighton_ocr/pixels_test.zig` makes. It prints each
page's CHW pixel shape and SHA256; the test holds those rows in
`fixtures/processor_pixels.txt` and compares its own prepared pixels
with them byte for byte:

```sh
python3 tools/zig/lighton_ocr_pixels.py --model MODEL_DIRECTORY \
  --output processor_pixels.txt
```

`tools/zig/lighton_ocr_reference.py capture` writes processor pixels,
vision features, decoder inputs and teacher logits with pinned local
Transformers and checkpoint files. Outputs must be outside the checkout.
The script does not compare tensors. The HIP capture tools read those
fixtures, verify their checkpoint and tensor checksums and compare the
native tensors with the reference:

```sh
python3 tools/zig/lighton_ocr_reference.py capture --model MODEL_DIRECTORY \
  --size 1B --revision REVISION --expect-transformers-version VERSION \
  --image RECEIPT.jpeg --output REFERENCE --device cpu
zig build tf-lighton-hip-capture tf-qwen-hip-capture \
  tf-hip-kernel-contract -Dhip=true -Dhip-arch=gfx1201 \
  -Dhipcc=/opt/rocm/core/bin/hipcc -j4
zig-out/bin/tf-lighton-hip-capture MODEL_1B REFERENCE OUTPUT 0
zig-out/bin/tf-qwen-hip-capture MODEL_QWEN REFERENCE OUTPUT 0
zig-out/bin/tf-hip-kernel-contract image lighton_ocr full
zig-out/bin/tf-hip-kernel-contract image qwen_vision full
```

Run device commands only after device selection and memory admission.
Contract controls must fail at their stated gate. The runner reports
those failures as controls; a passing suite has no missed controls or
production failures.

Capture tools report differences without assigning a pass from an arbitrary
tolerance. Cross-backend logits can differ. Within the native engine,
shared, drafted and resumed execution must preserve the required bytes.

## Current boundaries

Use one GPU and BF16 activations for images. Qwen vision admits at most
4096 merged image tokens, or 16384 encoder patches. LightOn 1B admits
its full 1540-pixel side, including odd patch grids; the learned merger
drops unmatched edges after the encoder.

The 1B host prompt cache compares exact image identities beside compact
cache words. Image states are not written to disk formats that lack that
metadata. On HIP, Qwen image requests currently keep per-request pages and
bypass cross-request radix reuse; text requests retain the existing cache
path. On CUDA, a Qwen image prompt resumes from a kept state on the
4,096-row chunk grid only when the images before that cut are the same.

On HIP, other AMD GPUs are not qualified. On CUDA, the 0.8B and 4B are
qualified on a GB10 only, and the 1B is not served. Metal is not qualified.
Issue #1927 tracks the rest of CUDA.
